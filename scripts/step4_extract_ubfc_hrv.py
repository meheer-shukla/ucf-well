"""Step 4 — Extract 14-D HRV features from cached UBFC traces.

Reads cache/ubfc_traces/subjectN.npz, re-runs POS + ROI fusion,
extracts IBIs, computes 14-D HRV features. One window per subject
(the whole clip is ~50-68 s, treated as a single analysis window).

Run:  python scripts/step4_extract_ubfc_hrv.py
Output:
  cache/ubfc_hrv.npz   (X, subjects, hr_median, sqi, feature_names)
"""
from __future__ import annotations
import os, sys, glob
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.rppg import rppg_window, quality_weights, fuse_rois
from src.features.hrv import hrv_features, FEATURE_NAMES


def main():
    cache_dir = "cache/ubfc_traces"
    files = sorted(glob.glob(os.path.join(cache_dir, "subject*.npz")))
    if not files:
        print("[FAIL] no cached traces found — run step3 first")
        return

    X_all, subj_all, hr_all, sqi_all = [], [], [], []
    skipped = []

    print(f"{'subject':10s}  {'n_ibi':>5s}  {'hr_med':>7s}  {'sqi':>5s}")
    print("-" * 42)

    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        z = np.load(f)
        fps = float(z["fps"])
        traces = {
            "forehead":    z["trace_forehead"],
            "left_cheek":  z["trace_left_cheek"],
            "right_cheek": z["trace_right_cheek"],
        }
        weights = quality_weights(traces, fps)
        fused = fuse_rois(traces, weights)

        res = rppg_window(fused, fps, method="POS")
        ibi = res["ibi"]
        feat = hrv_features(ibi)
        if feat is None:
            print(f"{sid:10s}  {len(ibi):>5d}  [skip] insufficient IBIs")
            skipped.append(sid)
            continue

        hr_med = float(60.0 / np.median(ibi))
        X_all.append(feat)
        subj_all.append(sid)
        hr_all.append(hr_med)
        sqi_all.append(float(res["sqi"]))
        print(f"{sid:10s}  {len(ibi):>5d}  {hr_med:7.2f}  {sqi_all[-1]:5.3f}")

    if not X_all:
        print("\n[FAIL] no subjects produced valid features")
        return

    X = np.asarray(X_all, np.float32)
    print(f"\nExtracted HRV features for {X.shape[0]} subjects, "
          f"dim={X.shape[1]}")
    if skipped:
        print(f"Skipped ({len(skipped)}): {skipped}")

    # Sanity: feature statistics
    print("\nFeature statistics (mean ± std across subjects):")
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        print(f"  {name:14s}: {col.mean():8.2f} ± {col.std():6.2f}")

    np.savez("cache/ubfc_hrv.npz",
             X=X,
             subjects=np.array(subj_all),
             hr_median=np.array(hr_all, np.float32),
             sqi=np.array(sqi_all, np.float32),
             feature_names=np.array(FEATURE_NAMES))
    print("\n[ OK ] wrote cache/ubfc_hrv.npz")


if __name__ == "__main__":
    main()