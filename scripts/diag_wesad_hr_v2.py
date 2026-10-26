"""WESAD diagnostic — final version.

Answers two questions:
  Q1. On one clean 60 s window (S10, baseline), do the v2 peak detectors
      land on the correct physiological events?
  Q2. Across all 15 subjects with time-paired windows, how well do ECG
      and BVP HR agree?  This number becomes the contact-sensor HR
      agreement ceiling reported in the paper (§V-A).

HR estimator: 60 / median(IBI).  Median is robust to a handful of missed
or spurious beats, unlike mean IBI.  Welch-PSD was tested and rejected
for BVP because the residual low-frequency swing dominates the global
spectral energy integral on Empatica E4.

Run:  python scripts/diag_wesad_hr_v2.py
Outputs:
  runs/diag/S10_window_v2.png
"""
from __future__ import annotations
import os, sys, glob, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.peaks import (
    ecg_rpeaks, bvp_peaks, ibi_from_ecg, ibi_from_bvp,
)

FS_CHEST = 700
FS_WRIST = 64
OUTDIR = "runs/diag"
os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------
def load_subject(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    ecg = np.asarray(d["signal"]["chest"]["ECG"]).squeeze().astype(np.float64)
    labels = np.asarray(d["label"]).squeeze().astype(int)
    bvp = None
    if "wrist" in d["signal"] and "BVP" in d["signal"]["wrist"]:
        bvp = np.asarray(d["signal"]["wrist"]["BVP"]).squeeze().astype(np.float64)
    return {"ecg": ecg, "bvp": bvp, "labels": labels}


def hr_from_ibi(ibi):
    if len(ibi) < 2:
        return float("nan")
    return float(60.0 / np.median(ibi))


# ----------------------------------------------------------------------
# Q1 — single clean window
# ----------------------------------------------------------------------
def q1_single_window():
    pkl = "datasets/WESAD/S10/S10.pkl"
    if not os.path.exists(pkl):
        print(f"[SKIP] {pkl} not found")
        return

    d = load_subject(pkl)
    ecg, bvp, lab = d["ecg"], d["bvp"], d["labels"]

    # Find a 60 s window that is >=99 % baseline (label 1)
    w = int(60 * FS_CHEST)
    start = 0
    for st in range(0, len(lab) - w, int(30 * FS_CHEST)):
        if (lab[st:st + w] == 1).mean() > 0.99:
            start = st
            break

    seg_ecg = ecg[start:start + w]
    bstart = int(round(start / FS_CHEST * FS_WRIST))
    blen = int(round(w / FS_CHEST * FS_WRIST))
    seg_bvp = bvp[bstart:bstart + blen]

    p_e = ecg_rpeaks(seg_ecg, FS_CHEST)
    p_b = bvp_peaks(seg_bvp, FS_WRIST)
    ibi_e = ibi_from_ecg(seg_ecg, FS_CHEST)
    ibi_b = ibi_from_bvp(seg_bvp, FS_WRIST)
    hr_e = hr_from_ibi(ibi_e)
    hr_b = hr_from_ibi(ibi_b)

    print("=" * 72)
    print(f"Q1  SINGLE-WINDOW  (S10, 60 s baseline, starts "
          f"{start / FS_CHEST:.0f}s)")
    print("=" * 72)
    print(f"  ECG peaks : {len(p_e):4d}   HR = {hr_e:6.2f} bpm")
    print(f"  BVP peaks : {len(p_b):4d}   HR = {hr_b:6.2f} bpm")
    print(f"  |ECG - BVP| : {abs(hr_e - hr_b):.2f} bpm")
    print()

    fig, ax = plt.subplots(2, 1, figsize=(14, 7))
    n_e = min(len(seg_ecg), 10 * FS_CHEST)
    n_b = min(len(seg_bvp), 10 * FS_WRIST)
    t_e = np.arange(len(seg_ecg)) / FS_CHEST
    t_b = np.arange(len(seg_bvp)) / FS_WRIST

    ax[0].plot(t_e[:n_e], seg_ecg[:n_e], lw=0.7, color="steelblue")
    pk = p_e[p_e < n_e]
    ax[0].plot(t_e[pk], seg_ecg[pk], "rv", ms=8,
               label=f"{len(pk)} peaks / 10s")
    ax[0].set_title(f"ECG (S10 baseline) — HR = {hr_e:.1f} bpm "
                    f"(IBI-median)")
    ax[0].set_xlabel("time (s)")
    ax[0].legend(fontsize=8)

    ax[1].plot(t_b[:n_b], seg_bvp[:n_b], lw=0.9, color="darkorange")
    pk = p_b[p_b < n_b]
    ax[1].plot(t_b[pk], seg_bvp[pk], "rv", ms=8,
               label=f"{len(pk)} peaks / 10s")
    ax[1].set_title(f"BVP (S10 baseline) — HR = {hr_b:.1f} bpm "
                    f"(IBI-median)")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTDIR, "S10_window_v2.png")
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"  [saved] {out}")
    print()


# ----------------------------------------------------------------------
# Q2 — all subjects, time-paired
# ----------------------------------------------------------------------
def q2_all_subjects():
    print("=" * 72)
    print("Q2  TIME-PAIRED ECG<->BVP HR AGREEMENT  (all subjects)")
    print("=" * 72)
    print(f"{'subj':6s} {'n':>4s}  {'MAE':>6s}  {'bias':>7s}  {'r':>7s}")
    print("-" * 72)

    win, step = 60, 30
    all_e, all_b = [], []

    for pkl in sorted(glob.glob("datasets/WESAD/S*/S*.pkl")):
        sid = os.path.basename(os.path.dirname(pkl))
        d = load_subject(pkl)
        ecg, bvp, lab = d["ecg"], d["bvp"], d["labels"]
        e_pts, b_pts = [], []

        # ECG windows
        for st in range(0, len(ecg) - win * FS_CHEST, step * FS_CHEST):
            seg = lab[st:st + win * FS_CHEST]
            v, c = np.unique(seg, return_counts=True)
            if c.max() / len(seg) < 0.9:
                continue
            ibi = ibi_from_ecg(ecg[st:st + win * FS_CHEST], FS_CHEST)
            hr = hr_from_ibi(ibi)
            if np.isfinite(hr):
                e_pts.append((round(st / FS_CHEST, 2), hr))

        # BVP windows
        for st in range(0, len(bvp) - win * FS_WRIST, step * FS_WRIST):
            ibi = ibi_from_bvp(bvp[st:st + win * FS_WRIST], FS_WRIST)
            hr = hr_from_ibi(ibi)
            if np.isfinite(hr):
                b_pts.append((round(st / FS_WRIST, 2), hr))

        ed = dict(e_pts)
        bd = dict(b_pts)
        common = sorted(set(ed) & set(bd))

        if len(common) < 5:
            print(f"{sid:6s} {len(common):>4d}  (insufficient overlap)")
            continue

        e = np.array([ed[t] for t in common])
        b = np.array([bd[t] for t in common])
        diff = e - b
        mae_v = float(np.mean(np.abs(diff)))
        bias_v = float(np.mean(diff))
        r_v = (float(np.corrcoef(e, b)[0, 1])
               if e.std() > 1e-6 and b.std() > 1e-6 else float("nan"))

        print(f"{sid:6s} {len(common):>4d}  {mae_v:6.2f}  "
              f"{bias_v:+7.2f}  {r_v:+7.3f}")

        all_e.append(e)
        all_b.append(b)

    if not all_e:
        print("[FAIL] no usable subject pairs")
        return

    e = np.concatenate(all_e)
    b = np.concatenate(all_b)
    diff = e - b
    r_pool = (float(np.corrcoef(e, b)[0, 1])
              if e.std() > 1e-6 and b.std() > 1e-6 else float("nan"))

    print("-" * 72)
    print(f"POOLED  n={len(e)}  "
          f"MAE={np.mean(np.abs(diff)):.2f}  "
          f"bias={np.mean(diff):+.2f}  "
          f"r={r_pool:+.3f}")
    print()

    m = float(np.mean(np.abs(diff)))
    n_good = sum(1 for ei, bi in zip(all_e, all_b)
                 if np.mean(np.abs(ei - bi)) < 4.0)
    print(f"  subjects with MAE < 4 bpm : {n_good}/{len(all_e)}")
    print()
    if m < 2.5:
        print("  [ OK ] MAE < 2.5 bpm — excellent ceiling.")
    elif m < 5.0:
        print(f"  [ OK ] MAE = {m:.2f} bpm — the honest contact-sensor")
        print("         ceiling for WESAD.  Report as-is in §V-A.")
    else:
        print(f"  [WARN] MAE {m:.2f} — investigate per-subject outliers.")


if __name__ == "__main__":
    q1_single_window()
    q2_all_subjects()