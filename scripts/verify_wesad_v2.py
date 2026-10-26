"""WESAD verification — final version.

Pipeline:
  1. Pan-Tompkins ECG detector (adaptive threshold + searchback)
  2. Median-detrended BVP detector (dicrotic-notch rejection)
  3. Time-indexed 60 s / 30 s windows, paired by wall-clock start time
  4. 14-D HRV features from IBI (corrected for missed beats)
  5. HR estimated from median IBI — robust to residual peak failures
  6. Pooled ECG<->BVP HR agreement = the contact-sensor ceiling for §V-A

Outputs:
  cache/wesad_ecg_hrv.npz          ECG HRV features + labels + groups
  cache/wesad_bvp_hrv.npz          BVP HRV features + labels + groups
  cache/wesad_v2_summary.json      per-subject metrics
"""
from __future__ import annotations
import os, sys, json, glob, argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.wesad import (
    load_subject, FS_CHEST, FS_WRIST, CLASS_NAMES,
    align_wrist_labels, RAW_TO_CLASS,
)
from src.features.hrv import hrv_features, FEATURE_NAMES
from src.preprocessing.peaks import ibi_from_ecg, ibi_from_bvp
from src.eval.metrics import mae, rmse, pearson_r, ccc

try:
    from src.eval.metrics import coral_distance, coral_cross_subject_null
    HAVE_CORAL = True
except ImportError:
    HAVE_CORAL = False


# ----------------------------------------------------------------------
# Window iteration — start time on the 700 Hz clock for both modalities
# ----------------------------------------------------------------------
def _maj_and_purity(labels_seg):
    v, c = np.unique(labels_seg, return_counts=True)
    return int(v[np.argmax(c)]), float(c.max() / len(labels_seg))


def iter_windows_ecg(ecg, labels, fs=FS_CHEST, win_sec=60, step_sec=30,
                     min_purity=0.9, keep_raw=(1, 2, 3)):
    w, s = int(win_sec * fs), int(step_sec * fs)
    for st in range(0, len(ecg) - w, s):
        maj, pur = _maj_and_purity(labels[st:st + w])
        if pur >= min_purity and maj in keep_raw:
            yield st / fs, ecg[st:st + w], RAW_TO_CLASS[maj]


def iter_windows_bvp(bvp, labels_700, fs=FS_WRIST, win_sec=60, step_sec=30,
                     min_purity=0.9, keep_raw=(1, 2, 3)):
    labels_w = align_wrist_labels(labels_700, len(bvp))
    w, s = int(win_sec * fs), int(step_sec * fs)
    for st in range(0, len(bvp) - w, s):
        maj, pur = _maj_and_purity(labels_w[st:st + w])
        if pur >= min_purity and maj in keep_raw:
            # Same physical clock as ECG (both from 700 Hz timeline)
            yield st / fs, bvp[st:st + w], RAW_TO_CLASS[maj]


# ----------------------------------------------------------------------
# Feature extraction per subject
# ----------------------------------------------------------------------
def _hr_from_ibi(ibi):
    if len(ibi) < 2:
        return float("nan")
    return float(60.0 / np.median(ibi))


def features_one_subject(pkl_path, win_sec=60, step_sec=30):
    d = load_subject(pkl_path)
    ecg, bvp, lab = d["ecg"], d["bvp"], d["labels"]
    sid = int(os.path.basename(os.path.dirname(pkl_path)).replace("S", ""))

    # rows: (start_sec, class, hrv_feature_vector, hr_bpm)
    ecg_rows = []
    for t0, win, c in iter_windows_ecg(ecg, lab,
                                       win_sec=win_sec, step_sec=step_sec):
        ibi = ibi_from_ecg(win, FS_CHEST)
        f = hrv_features(ibi)
        if f is None:
            continue
        hr = _hr_from_ibi(ibi)
        if not np.isfinite(hr):
            continue
        ecg_rows.append((round(t0, 3), c, f, hr))

    bvp_rows = []
    if bvp is not None:
        for t0, win, c in iter_windows_bvp(bvp, lab,
                                           win_sec=win_sec, step_sec=step_sec):
            ibi = ibi_from_bvp(win, FS_WRIST)
            f = hrv_features(ibi)
            if f is None:
                continue
            hr = _hr_from_ibi(ibi)
            if not np.isfinite(hr):
                continue
            bvp_rows.append((round(t0, 3), c, f, hr))

    return sid, ecg_rows, bvp_rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/WESAD")
    ap.add_argument("--win_sec", type=float, default=60.0)
    ap.add_argument("--step_sec", type=float, default=30.0)
    ap.add_argument("--out_dir", default="cache")
    args = ap.parse_args()

    pkls = sorted(glob.glob(os.path.join(args.root, "S*", "S*.pkl")))
    if not pkls:
        print(f"[FAIL] no subjects under {args.root}")
        return
    print(f"[ OK ] {len(pkls)} subjects found\n")

    per_subject = []
    XE, XB, yE, yB, gE, gB = [], [], [], [], [], []
    hr_e_pool, hr_b_pool = [], []

    for pkl in pkls:
        sid, ecg_rows, bvp_rows = features_one_subject(
            pkl, win_sec=args.win_sec, step_sec=args.step_sec)

        e_d = {t: (c, f, hr) for t, c, f, hr in ecg_rows}
        b_d = {t: (c, f, hr) for t, c, f, hr in bvp_rows}
        common = sorted(set(e_d) & set(b_d))

        if len(common) < 5:
            print(f"S{sid:<4d} matched={len(common):3d}  (insufficient)")
            continue

        hr_e = np.array([e_d[t][2] for t in common])
        hr_b = np.array([b_d[t][2] for t in common])
        diff = hr_e - hr_b
        m = float(np.mean(np.abs(diff)))
        b = float(np.mean(diff))
        r = (float(np.corrcoef(hr_e, hr_b)[0, 1])
             if hr_e.std() > 1e-6 and hr_b.std() > 1e-6 else float("nan"))
        c_val = float(ccc(hr_e, hr_b))

        print(f"S{sid:<4d} matched={len(common):3d}  "
              f"MAE={m:5.2f}  bias={b:+6.2f}  r={r:+.3f}  CCC={c_val:+.3f}")

        per_subject.append({
            "subject": sid, "n_matched": len(common),
            "mae": m, "bias": b, "r": r, "ccc": c_val,
        })

        for t in common:
            XE.append(e_d[t][1]); yE.append(e_d[t][0]); gE.append(sid)
            XB.append(b_d[t][1]); yB.append(b_d[t][0]); gB.append(sid)
            hr_e_pool.append(e_d[t][2])
            hr_b_pool.append(b_d[t][2])

    if not XE:
        print("[FAIL] no matched windows across all subjects")
        return

    XE = np.asarray(XE, np.float32)
    XB = np.asarray(XB, np.float32)
    yE = np.asarray(yE); yB = np.asarray(yB)
    gE = np.asarray(gE); gB = np.asarray(gB)
    hr_e = np.asarray(hr_e_pool, np.float64)
    hr_b = np.asarray(hr_b_pool, np.float64)

    # ------------------------------------------------------------------
    # Pooled ECG<->BVP HR agreement — the paper's contact-sensor ceiling
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("POOLED HR AGREEMENT  (ECG vs BVP, time-paired, IBI-median)")
    print("=" * 72)
    diff = hr_e - hr_b
    loa_lo = float(np.mean(diff) - 1.96 * np.std(diff, ddof=1))
    loa_hi = float(np.mean(diff) + 1.96 * np.std(diff, ddof=1))
    print(f"  n          : {len(hr_e)}")
    print(f"  MAE        : {mae(hr_e, hr_b):.2f} bpm")
    print(f"  RMSE       : {rmse(hr_e, hr_b):.2f} bpm")
    print(f"  bias       : {np.mean(diff):+.2f} bpm")
    print(f"  r          : {pearson_r(hr_e, hr_b):+.3f}")
    print(f"  CCC        : {ccc(hr_e, hr_b):+.3f}")
    print(f"  LoA (95%)  : [{loa_lo:+.2f}, {loa_hi:+.2f}] bpm")
    print()

    # ------------------------------------------------------------------
    # Modality gap (optional)
    # ------------------------------------------------------------------
    if HAVE_CORAL:
        print("=" * 72)
        print("MODALITY GAP IN 14-D HRV SPACE (standardized CORAL)")
        print("=" * 72)
        null = coral_cross_subject_null(XE, gE, XB, gB, n_perm=200)
        print(f"  same-subject       : {null['same_subject']:.4f}")
        print(f"  cross-subject mean : {null['cross_subject_mean']:.4f}")
        print(f"  cross-subject p05  : {null['cross_subject_p05']:.4f}")
        print(f"  cross-subject p95  : {null['cross_subject_p95']:.4f}")
        if null["same_subject"] < null["cross_subject_p05"]:
            print("  [ OK ] same-subject gap < cross-subject p05")
        else:
            print("  [WARN] same-subject gap not clearly below null.")
    else:
        print("[skip] coral helpers not found in src/eval/metrics.py")

    # ------------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------------
    print("\nClass distribution (matched windows):")
    for c, name in enumerate(CLASS_NAMES):
        n = int(np.sum(yE == c))
        print(f"  {c} {name:10s}: {n:5d}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, "wesad_ecg_hrv.npz"),
             X=XE, y=yE, groups=gE,
             feature_names=np.array(FEATURE_NAMES))
    np.savez(os.path.join(args.out_dir, "wesad_bvp_hrv.npz"),
             X=XB, y=yB, groups=gB,
             feature_names=np.array(FEATURE_NAMES))
    with open(os.path.join(args.out_dir, "wesad_v2_summary.json"), "w") as f:
        json.dump(per_subject, f, indent=2)

    print(f"\n[ OK ] wrote {args.out_dir}/wesad_ecg_hrv.npz")
    print(f"[ OK ] wrote {args.out_dir}/wesad_bvp_hrv.npz")
    print(f"[ OK ] wrote {args.out_dir}/wesad_v2_summary.json")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    m = mae(hr_e, hr_b)
    if m < 2.5:
        print(f"\n*** VERDICT: Step 1 GREEN — MAE={m:.2f} bpm (excellent) ***")
    elif m < 5.0:
        print(f"\n*** VERDICT: Step 1 GREEN — MAE={m:.2f} bpm "
              f"(honest WESAD ceiling) ***")
    else:
        print(f"\n*** VERDICT: Step 1 NOT closed — MAE={m:.2f} bpm ***")


if __name__ == "__main__":
    main()