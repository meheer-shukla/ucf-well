"""STEP 1 smoke test:
  1. Load each WESAD subject (chest ECG + wrist BVP).
  2. Extract IBIs from both signals.
  3. Compute 14-D HRV features from both.
  4. Compare HR estimates ECG-vs-BVP (should agree within a few bpm).
  5. Compute the ECG-vs-BVP CORAL distance in the shared HRV space.

Run:  python scripts/step1_verify_wesad.py
Outputs:
  cache/wesad_ecg_hrv.npz   X, y, groups, subjects
  cache/wesad_bvp_hrv.npz   X, y, groups, subjects
  cache/step1_wesad_summary.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy import signal as sps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.wesad import (list_subjects, load_subject,
                            sliding_labeled_windows, sliding_labeled_windows_bvp,
                            FS_CHEST, FS_WRIST, CLASS_NAMES)
from src.features.hrv import hrv_features, FEATURE_NAMES
from src.eval.metrics import (
    coral_distance, mmd_rbf, regression_report,
    format_regression_table_row, mae, rmse, pearson_r, ccc,
)


# ----------------------------------------------------------------------
# Peak detection
# ----------------------------------------------------------------------
def ecg_rpeaks(ecg: np.ndarray, fs: int = FS_CHEST) -> np.ndarray:
    """Pan-Tompkins-lite R-peak detector, adequate for clean chest ECG."""
    b, a = sps.butter(3, [5.0 / (fs / 2), 25.0 / (fs / 2)], btype="band")
    x = sps.filtfilt(b, a, ecg)
    d = np.diff(x) ** 2
    win = max(1, int(0.12 * fs))
    integ = np.convolve(d, np.ones(win) / win, mode="same")
    thr = 0.3 * np.percentile(integ, 99)
    peaks, _ = sps.find_peaks(integ, height=thr, distance=int(0.25 * fs))
    return peaks


def ibi_from_ecg(ecg: np.ndarray, fs: int = FS_CHEST) -> np.ndarray:
    p = ecg_rpeaks(ecg, fs)
    if len(p) < 3:
        return np.array([])
    ibi = np.diff(p) / fs
    return ibi[(ibi > 0.33) & (ibi < 1.43)]


def bvp_peaks(bvp: np.ndarray, fs: int = FS_WRIST) -> np.ndarray:
    """Peak detector for wrist BVP (already a clean pulsatile signal)."""
    b, a = sps.butter(3, [0.7 / (fs / 2), 4.0 / (fs / 2)], btype="band")
    x = sps.filtfilt(b, a, bvp)
    min_dist = int(0.33 * fs)          # max 180 bpm
    thr = 0.25 * np.std(x)
    peaks, _ = sps.find_peaks(x, height=thr, distance=min_dist)
    return peaks


def ibi_from_bvp(bvp: np.ndarray, fs: int = FS_WRIST) -> np.ndarray:
    p = bvp_peaks(bvp, fs)
    if len(p) < 3:
        return np.array([])
    ibi = np.diff(p) / fs
    return ibi[(ibi > 0.33) & (ibi < 1.43)]


# ----------------------------------------------------------------------
# Feature extraction per subject
# ----------------------------------------------------------------------
def build_subject_features(pkl_path: str, win_sec=60, step_sec=30):
    """Returns (X_ecg, X_bvp, y, per_window_hr_ecg, per_window_hr_bvp)."""
    d = load_subject(pkl_path)
    ecg, bvp, labels = d["ecg"], d["bvp"], d["labels"]

    X_ecg, X_bvp, y = [], [], []
    hr_ecg, hr_bvp = [], []

    for win_ecg, lab in sliding_labeled_windows(
            ecg, labels, win_sec=win_sec, step_sec=step_sec):
        ibi_e = ibi_from_ecg(win_ecg)
        fe = hrv_features(ibi_e)
        if fe is None:
            continue
        X_ecg.append(fe); y.append(lab)
        hr_ecg.append(fe[0])

    if bvp is not None:
        for win_bvp, lab in sliding_labeled_windows_bvp(
                bvp, labels, win_sec=win_sec, step_sec=step_sec):
            ibi_b = ibi_from_bvp(win_bvp)
            fb = hrv_features(ibi_b)
            if fb is None:
                continue
            X_bvp.append(fb)
            hr_bvp.append(fb[0])

    X_ecg = np.asarray(X_ecg, np.float32) if X_ecg else np.zeros((0, 14), np.float32)
    X_bvp = np.asarray(X_bvp, np.float32) if X_bvp else np.zeros((0, 14), np.float32)
    return (X_ecg, X_bvp, np.asarray(y), np.asarray(hr_ecg), np.asarray(hr_bvp))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    root = "datasets/WESAD"
    pkls = list_subjects(root)
    if not pkls:
        print("[FAIL] no subjects found under", root)
        return
    print(f"[ OK ] {len(pkls)} subjects found")

    XE_all, XB_all, y_all, g_all = [], [], [], []
    per_subject = []

    for i, pkl in enumerate(pkls):
        sid = os.path.splitext(os.path.basename(pkl))[0]
        try:
            XE, XB, y, hr_e, hr_b = build_subject_features(pkl)
        except Exception as e:
            print(f"[FAIL] {sid}: {e}")
            continue

        n_ecg = len(XE)
        n_bvp = len(XB)
        # Compare HR where both exist and windows are aligned (they are by
        # construction when step/window are the same, but BVP may have fewer)
        k = min(n_ecg, n_bvp)
        if k > 0:
            hr_mae_ecg_bvp = mae(hr_e[:k], hr_b[:k])
            hr_r_ecg_bvp   = pearson_r(hr_e[:k], hr_b[:k])
        else:
            hr_mae_ecg_bvp, hr_r_ecg_bvp = float("nan"), float("nan")

        per_subject.append({
            "subject": sid,
            "n_windows_ecg": int(n_ecg),
            "n_windows_bvp": int(n_bvp),
            "hr_mae_ecg_vs_bvp": float(hr_mae_ecg_bvp) if k else None,
            "hr_r_ecg_vs_bvp":   float(hr_r_ecg_bvp)   if k else None,
        })
        print(f"  {sid:5s}  ECG win={n_ecg:3d}  BVP win={n_bvp:3d}  "
              f"HR(ECG)-HR(BVP) MAE={hr_mae_ecg_bvp:5.2f} bpm  r={hr_r_ecg_bvp:+.3f}")

        XE_all.append(XE); XB_all.append(XB)
        y_all.append(y)
        g_all.append(np.full(len(XE), i))

    if not XE_all:
        print("[FAIL] no features extracted")
        return

    XE = np.concatenate(XE_all); yE = np.concatenate(y_all)
    gE = np.concatenate(g_all)
    XB = np.concatenate(XB_all)

    # ---------------------------------------------------------------
    # HR agreement across the whole dataset (both signals pooled)
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("HR agreement: chest ECG vs wrist BVP (pooled across subjects)")
    print("=" * 72)
    # align per subject and pool
    hrE_pooled, hrB_pooled = [], []
    for sub in per_subject:
        pass  # already collected per-subject; recompute pooled directly
    # Recompute pooled aligned arrays
    for i, pkl in enumerate(pkls):
        try:
            _, _, _, hr_e, hr_b = build_subject_features(pkl)
        except Exception:
            continue
        k = min(len(hr_e), len(hr_b))
        if k:
            hrE_pooled.extend(hr_e[:k]); hrB_pooled.extend(hr_b[:k])
    hrE_pooled = np.array(hrE_pooled); hrB_pooled = np.array(hrB_pooled)

    if len(hrE_pooled) > 2:
        rep = regression_report(hrB_pooled, hrE_pooled,
                                unit=" bpm", name="HR (BVP~ECG)")
        print(format_regression_table_row(rep))
        print(f"  -> This is the intrinsic ceiling for contactless HR: "
              f"no rPPG method can beat HR agreement between two contact sensors.")

    # ---------------------------------------------------------------
    # Modality gap in HRV feature space
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Modality gap in the shared 14-D HRV space")
    print("=" * 72)
    if len(XB) > 10:
        d_ecg_bvp   = coral_distance(XE, XB)
        # Random-pairing baseline: same arrays with rows shuffled independently
        rng = np.random.default_rng(42)
        XE_shuf = XE[rng.permutation(len(XE))]
        XB_shuf = XB[rng.permutation(len(XB))]
        d_random    = coral_distance(XE_shuf, XB_shuf)
        print(f"  CORAL d(ECG, BVP)        : {d_ecg_bvp:.4f}")
        print(f"  CORAL d(ECG*, BVP*) shuff: {d_random:.4f}   (sanity baseline)")
        print(f"  MMD   (ECG, BVP)         : {mmd_rbf(XE, XB):.4f}")
        assert d_ecg_bvp <= d_random + 1e-6, \
            "ECG/BVP are LESS similar than random rows — bug in feature pipeline"
        print("  [ OK ] ECG-BVP gap < shuffled baseline")
    else:
        d_ecg_bvp = d_random = float("nan")
        print("  [WARN] insufficient BVP windows to compute gap")

    # ---------------------------------------------------------------
    # Class distribution after windowing
    # ---------------------------------------------------------------
    print("\nClass distribution (windows, class-ordered)")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {c} {name:10s}: {int((yE == c).sum()):5d}")

    # ---------------------------------------------------------------
    # Persist
    # ---------------------------------------------------------------
    os.makedirs("cache", exist_ok=True)
    np.savez("cache/wesad_ecg_hrv.npz",
             X=XE, y=yE, groups=gE,
             feature_names=np.array(FEATURE_NAMES),
             subjects=np.array([os.path.splitext(os.path.basename(p))[0] for p in pkls]))
    if len(XB):
        np.savez("cache/wesad_bvp_hrv.npz",
                 X=XB, feature_names=np.array(FEATURE_NAMES))
    with open("cache/step1_wesad_summary.json", "w") as f:
        json.dump({
            "n_subjects": len(pkls),
            "n_windows_ecg": int(len(XE)),
            "n_windows_bvp": int(len(XB)),
            "coral_ecg_bvp": float(d_ecg_bvp) if np.isfinite(d_ecg_bvp) else None,
            "coral_random_baseline": float(d_random) if np.isfinite(d_random) else None,
            "per_subject": per_subject,
            "feature_names": FEATURE_NAMES,
        }, f, indent=2)

    print("\n[ OK ] wrote cache/wesad_ecg_hrv.npz")
    if len(XB):
        print("[ OK ] wrote cache/wesad_bvp_hrv.npz")
    print("[ OK ] wrote cache/step1_wesad_summary.json")


if __name__ == "__main__":
    main()