"""Step 7 — Synthetic-rPPG WESAD from UBFC noise calibration.

Two studies:

  Study 3: UBFC same-subject modality gap.
      Extract 6-D HRV features from finger PPG and from rPPG on the
      same 42 UBFC subjects / clips.  Report per-feature deviation
      and CORAL distance.

  Study 4 setup: Synthetic-rPPG WESAD.
      Fit a robust 6-D noise model (median/MAD) from the UBFC pairs.
      Apply delta to WESAD ECG features.  Save for step5.

Why 6-D instead of 10-D:
    Initial 10-D extraction showed that 4 of the 10 features (RMSSD,
    pNN50, IBI-IQR, SD1) suffer > 50% median relative error between
    finger PPG and rPPG on the same subjects.  These are known to be
    sensitive to missed/extra peaks in the rPPG peak detector, and
    their unreliability is documented in the rPPG HRV literature.
    We retain the 6 features with < 35% median relative error:
        mean_hr, sdnn, cvnn, mean_ibi, median_ibi, sd2
    This selection is reported in the paper's Study 3.

Outputs:
  cache/ubfc_ppg_hrv.npz         6-D HRV from finger PPG
  cache/ubfc_rppg_hrv.npz        6-D HRV from rPPG
  cache/ubfc_modality_gap.json   per-feature deviation stats
  cache/wesad_synthetic_rppg.npz  WESAD ECG + sampled delta, labels
  runs/diag/ubfc_modality_gap.png
"""
from __future__ import annotations
import os, sys, glob, json
import numpy as np
from scipy import signal as sps
from scipy import interpolate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.peaks import bandpass, bvp_peaks
from src.features.hrv import hrv_features, FEATURE_NAMES
from src.eval.metrics import coral_distance


# ======================================================================
# 6-D feature selection
# ======================================================================
# Indices into the 14-D HRV vector produced by src.features.hrv
# 0:mean_hr  1:sdnn  4:cvnn  5:mean_ibi  6:median_ibi  13:sd2
KEEP_6D = [0, 1, 4, 5, 6, 13]
NAMES_6D = [FEATURE_NAMES[i] for i in KEEP_6D]
assert len(KEEP_6D) == 6


# ======================================================================
# Robust IBI filtering
# ======================================================================
def filter_ibi_outliers(ibi: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Reject IBIs that deviate more than k*1.4826*MAD from the median.

    MAD = median absolute deviation; 1.4826 scales it to an equivalent
    Gaussian std.  Robust to the small number of catastrophic peaks the
    rPPG detector picks up.  Keeps genuine physiological variability.
    """
    if len(ibi) < 5:
        return ibi
    med = float(np.median(ibi))
    mad = float(np.median(np.abs(ibi - med)))
    if mad < 1e-9:
        return ibi
    thr = k * 1.4826 * mad
    return ibi[np.abs(ibi - med) <= thr]


def iteratively_filter_ibi(ibi: np.ndarray, k: float = 3.0,
                           max_iter: int = 3) -> np.ndarray:
    prev = ibi
    for _ in range(max_iter):
        cur = filter_ibi_outliers(prev, k=k)
        if len(cur) == len(prev):
            break
        prev = cur
    return prev


# ======================================================================
# Feature extraction from an arbitrary pulsatile waveform
# ======================================================================
def hrv_from_signal(sig: np.ndarray, fs: float,
                    method: str = "bvp") -> np.ndarray | None:
    """Extract 6-D HRV features from a raw pulsatile waveform."""
    if len(sig) < int(fs * 20):
        return None

    if method == "bvp":
        p = bvp_peaks(sig, fs=fs)
    else:
        # generic peak finder for camera-derived rPPG
        x = bandpass(sig, fs, 0.7, 4.0, order=3)
        min_dist = int(fs * 60.0 / 200.0)
        prom = 0.3 * float(np.std(x))
        p, _ = sps.find_peaks(x, distance=min_dist, prominence=prom)

    if len(p) < 20:
        return None

    ibi = np.diff(p) / fs
    ibi = ibi[(ibi > 60.0 / 200.0) & (ibi < 60.0 / 40.0)]
    ibi = iteratively_filter_ibi(ibi, k=3.0)

    if len(ibi) < 20:
        return None

    f14 = hrv_features(ibi)
    if f14 is None:
        return None
    return f14[KEEP_6D]


# ======================================================================
# Study 3 — UBFC finger PPG vs rPPG
# ======================================================================
def run_study3(ubfc_root="datasets/UBFC",
               trace_dir="cache/ubfc_traces"):
    subj_dirs = sorted(
        glob.glob(os.path.join(ubfc_root, "subject*")),
        key=lambda p: int(os.path.basename(p).replace("subject", "")),
    )

    rows = []

    for subj_dir in subj_dirs:
        sid = os.path.basename(subj_dir)
        gt_path = os.path.join(subj_dir, "ground_truth.txt")
        trace_path = os.path.join(trace_dir, f"{sid}.npz")

        if not (os.path.exists(gt_path) and os.path.exists(trace_path)):
            print(f"  {sid}: missing gt or cached trace, skip")
            continue

        # ---------- Finger PPG ----------
        gt = np.loadtxt(gt_path)
        if gt.shape[0] == 3 and gt.shape[1] > 3:
            ppg_raw, ts = gt[0], gt[2]
        else:
            ppg_raw, ts = gt[:, 0], gt[:, 2]

        t_new = np.arange(ts[0], ts[-1], 1.0 / 30.0)
        f_interp = interpolate.interp1d(ts, ppg_raw, kind="linear",
                                        bounds_error=False,
                                        fill_value="extrapolate")
        ppg_u = f_interp(t_new)
        e = 30
        ppg_u = ppg_u[e:-e]
        feat_ppg = hrv_from_signal(ppg_u, 30.0, method="bvp")

        # ---------- rPPG ----------
        z = np.load(trace_path)
        fps = float(z["fps"])

        from src.preprocessing.rppg import (
            pos, detrend_moving_avg, fuse_rois, quality_weights,
        )
        traces = {
            "forehead":    z["trace_forehead"],
            "left_cheek":  z["trace_left_cheek"],
            "right_cheek": z["trace_right_cheek"],
        }
        weights = quality_weights(traces, fps)
        fused = fuse_rois(traces, weights)
        raw = pos(fused)
        raw = detrend_moving_avg(raw, int(fps))

        e2 = int(fps)
        raw = raw[e2:-e2]
        feat_rppg = hrv_from_signal(raw, fps, method="generic")

        if feat_ppg is None or feat_rppg is None:
            print(f"  {sid}: insufficient data "
                  f"(ppg_ok={feat_ppg is not None}, "
                  f"rppg_ok={feat_rppg is not None})")
            continue

        print(f"  {sid:10s}  ppg_hr={feat_ppg[0]:6.2f}  "
              f"rppg_hr={feat_rppg[0]:6.2f}  "
              f"d_hr={feat_rppg[0] - feat_ppg[0]:+6.2f}  "
              f"d_sdnn={feat_rppg[1] - feat_ppg[1]:+7.2f} ms  "
              f"d_sd2={feat_rppg[5] - feat_ppg[5]:+7.2f} ms")
        rows.append((sid, feat_ppg, feat_rppg))

    if not rows:
        raise RuntimeError("no usable subjects for Study 3")

    subjects = [r[0] for r in rows]
    X_ppg = np.stack([r[1] for r in rows]).astype(np.float32)
    X_rppg = np.stack([r[2] for r in rows]).astype(np.float32)

    # Robust statistics: median + MAD-derived std
    delta = X_rppg - X_ppg
    delta_median = np.median(delta, axis=0)
    delta_mad = np.median(np.abs(delta - delta_median), axis=0)
    delta_std_robust = 1.4826 * delta_mad

    print("\n" + "=" * 84)
    print(f"STUDY 3 — UBFC same-subject modality gap "
          f"({len(subjects)} subjects, 6-D features)")
    print("=" * 84)
    print(f"{'feature':14s}  {'PPG med':>10s}  {'rPPG med':>10s}  "
          f"{'d_med':>9s}  {'dMAD*1.48':>10s}  {'rel%':>7s}")
    print("-" * 84)
    for i, name in enumerate(NAMES_6D):
        ppg_med = float(np.median(X_ppg[:, i]))
        rppg_med = float(np.median(X_rppg[:, i]))
        rel = 100.0 * abs(delta_median[i]) / (abs(ppg_med) + 1e-9)
        print(f"{name:14s}  {ppg_med:10.3f}  {rppg_med:10.3f}  "
              f"{delta_median[i]:+9.3f}  {delta_std_robust[i]:10.3f}  "
              f"{rel:6.2f}%")

    # CORAL distance on standardized features
    Xall = np.vstack([X_ppg, X_rppg])
    mu = Xall.mean(axis=0, keepdims=True)
    sd = Xall.std(axis=0, keepdims=True) + 1e-8
    X_ppg_s = (X_ppg - mu) / sd
    X_rppg_s = (X_rppg - mu) / sd
    d_coral = coral_distance(X_ppg_s, X_rppg_s)
    print(f"\nCORAL distance (finger PPG vs rPPG, 6-D, standardized): "
          f"{d_coral:.4f}")

    # Save
    np.savez("cache/ubfc_ppg_hrv.npz",
             X=X_ppg, subjects=np.array(subjects),
             feature_names=np.array(NAMES_6D))
    np.savez("cache/ubfc_rppg_hrv.npz",
             X=X_rppg, subjects=np.array(subjects),
             feature_names=np.array(NAMES_6D))

    gap = {
        "n_subjects": len(subjects),
        "feature_names": NAMES_6D,
        "delta_median": delta_median.tolist(),
        "delta_std_robust": delta_std_robust.tolist(),
        "coral_distance": float(d_coral),
        "ppg_median": np.median(X_ppg, axis=0).tolist(),
        "rppg_median": np.median(X_rppg, axis=0).tolist(),
    }
    with open("cache/ubfc_modality_gap.json", "w") as f:
        json.dump(gap, f, indent=2)

    # Figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(NAMES_6D))
    ax[0].bar(x - 0.2, np.median(X_ppg, axis=0), 0.4, label="finger PPG")
    ax[0].bar(x + 0.2, np.median(X_rppg, axis=0), 0.4, label="rPPG")
    ax[0].set_xticks(x); ax[0].set_xticklabels(NAMES_6D, rotation=30, ha="right")
    ax[0].set_ylabel("median value")
    ax[0].set_title("UBFC HRV features — finger PPG vs rPPG")
    ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")

    ax[1].bar(x, delta_median, yerr=delta_std_robust, capsize=3)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(NAMES_6D, rotation=30, ha="right")
    ax[1].set_ylabel("rPPG - PPG (median +/- robust std)")
    ax[1].set_title(f"Per-feature rPPG error  -  CORAL d = {d_coral:.4f}")
    ax[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("runs/diag/ubfc_modality_gap.png", dpi=110)
    plt.close()
    print("[ OK ] wrote caches + runs/diag/ubfc_modality_gap.png")

    return delta_median, delta_std_robust, subjects


# ======================================================================
# Study 4 setup — synthetic-rPPG WESAD
# ======================================================================
def build_synthetic_wesad(delta_median, delta_std_robust, seed=42):
    """Apply UBFC-derived delta to WESAD ECG 6-D features."""
    rng = np.random.default_rng(seed)

    z = np.load("cache/wesad_ecg_hrv.npz")
    X14 = z["X"]
    y = z["y"]
    g = z["groups"]
    X6 = X14[:, KEEP_6D]

    n, d = X6.shape
    noise = rng.normal(
        loc=np.tile(delta_median, (n, 1)).astype(np.float32),
        scale=np.tile(delta_std_robust, (n, 1)).astype(np.float32),
        size=(n, d),
    ).astype(np.float32)
    X_syn = X6 + noise

    np.savez("cache/wesad_synthetic_rppg.npz",
             X=X_syn, y=y, groups=g,
             X_ecg_6d=X6,
             feature_names=np.array(NAMES_6D),
             delta_median=delta_median.astype(np.float32),
             delta_std_robust=delta_std_robust.astype(np.float32))

    print(f"\n[ OK ] wrote cache/wesad_synthetic_rppg.npz  "
          f"({X_syn.shape[0]} windows, {X_syn.shape[1]}-D)")

    print("\nSanity check - distribution shift induced by rPPG noise:")
    print(f"{'feature':14s}  {'ECG med':>10s}  {'synth med':>10s}  "
          f"{'d_med':>9s}  {'ECG std':>9s}  {'synth std':>10s}")
    print("-" * 72)
    for i, name in enumerate(NAMES_6D):
        ecg_med = float(np.median(X6[:, i]))
        syn_med = float(np.median(X_syn[:, i]))
        ecg_std = float(X6[:, i].std())
        syn_std = float(X_syn[:, i].std())
        print(f"{name:14s}  {ecg_med:10.3f}  {syn_med:10.3f}  "
              f"{syn_med - ecg_med:+9.3f}  {ecg_std:9.3f}  {syn_std:10.3f}")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 72)
    print("STEP 7 — UBFC modality gap + synthetic-rPPG WESAD (6-D)")
    print("=" * 72)
    os.makedirs("runs/diag", exist_ok=True)

    print("\n--- Study 3: extracting UBFC PPG / rPPG features ---")
    delta_median, delta_std_robust, subjects = run_study3()

    print("\n--- Study 4 setup: building synthetic-rPPG WESAD ---")
    build_synthetic_wesad(delta_median, delta_std_robust)

    print("\n[ OK ] step7 complete")


if __name__ == "__main__":
    main()