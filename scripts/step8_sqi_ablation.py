"""Step 8 — SQI vs uniform vs forehead-only fusion on UBFC.

Compares HR MAE across three fusion strategies on 42 UBFC subjects:
  forehead_only : only the forehead ROI
  uniform       : equal weights across 3 ROIs
  sqi_fusion    : SQI-weighted (current pipeline)

Output:
  cache/ubfc_sqi_ablation.csv
  cache/ubfc_sqi_ablation.json
  runs/diag/sqi_ablation.png
"""
from __future__ import annotations
import os, sys, glob, json, csv
import numpy as np
from scipy import interpolate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.peaks import hr_from_psd
from src.preprocessing.rppg import (pos, detrend_moving_avg,
                                     quality_weights, fuse_rois)


ROI_KEYS = ["forehead", "left_cheek", "right_cheek"]
STRATEGIES = ["forehead_only", "uniform", "sqi_fusion"]


def fuse_strategy(traces, fps, strategy):
    if strategy == "forehead_only":
        return traces["forehead"].copy()
    if strategy == "uniform":
        w = {k: 1.0 / 3.0 for k in ROI_KEYS}
        return fuse_rois(traces, w)
    if strategy == "sqi_fusion":
        w = quality_weights(traces, fps)
        return fuse_rois(traces, w)
    raise ValueError(strategy)


def hr_from_rppg_trace(fused, fps):
    raw = pos(fused)
    raw = detrend_moving_avg(raw, int(fps))
    e = int(fps)                          # trim 1 s each side
    raw = raw[e:-e]
    return float(hr_from_psd(raw, fps))


def hr_from_finger_ppg(gt_path):
    gt = np.loadtxt(gt_path)
    if gt.shape[0] == 3 and gt.shape[1] > 3:
        ppg_raw, ts = gt[0], gt[2]
    else:
        ppg_raw, ts = gt[:, 0], gt[:, 2]
    t_new = np.arange(ts[0], ts[-1], 1.0 / 30.0)
    f = interpolate.interp1d(ts, ppg_raw, kind="linear",
                             bounds_error=False, fill_value="extrapolate")
    ppg_u = f(t_new)
    e = 30
    ppg_u = ppg_u[e:-e]
    return float(hr_from_psd(ppg_u, 30.0))


def main():
    subj_dirs = sorted(
        glob.glob("datasets/UBFC/subject*"),
        key=lambda p: int(os.path.basename(p).replace("subject", "")),
    )

    rows = []
    print(f"{'subject':10s}  {'gt':>6s}  "
          f"{'forehead':>9s}  {'uniform':>8s}  {'sqi':>6s}")
    print("-" * 52)

    for subj_dir in subj_dirs:
        sid = os.path.basename(subj_dir)
        gt_path = os.path.join(subj_dir, "ground_truth.txt")
        trace_path = f"cache/ubfc_traces/{sid}.npz"
        if not (os.path.exists(gt_path) and os.path.exists(trace_path)):
            continue

        z = np.load(trace_path)
        fps = float(z["fps"])
        traces = {k: z[f"trace_{k}"] for k in ROI_KEYS}

        try:
            hr_gt = hr_from_finger_ppg(gt_path)
        except Exception as e:
            print(f"  {sid}: gt fail ({e})")
            continue

        hrs = {}
        for s in STRATEGIES:
            try:
                fused = fuse_strategy(traces, fps, s)
                hrs[s] = hr_from_rppg_trace(fused, fps)
            except Exception:
                hrs[s] = float("nan")

        print(f"{sid:10s}  {hr_gt:6.2f}  "
              f"{hrs['forehead_only']:9.2f}  "
              f"{hrs['uniform']:8.2f}  {hrs['sqi_fusion']:6.2f}")
        rows.append((sid, hr_gt,
                     hrs["forehead_only"],
                     hrs["uniform"],
                     hrs["sqi_fusion"]))

    if not rows:
        print("[FAIL] no usable subjects")
        return

    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    gt = arr[:, 0]
    res = {}
    print("\n" + "=" * 64)
    print("SUMMARY — HR MAE vs finger PPG (42 UBFC subjects)")
    print("=" * 64)
    print(f"{'strategy':<16s} {'MAE':>8s} {'RMSE':>8s} {'r':>8s} {'n':>4s}")
    print("-" * 64)
    for i, s in enumerate(STRATEGIES):
        pred = arr[:, i + 1]
        mask = np.isfinite(pred) & np.isfinite(gt)
        p, g = pred[mask], gt[mask]
        mae = float(np.mean(np.abs(p - g)))
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))
        r = float(np.corrcoef(p, g)[0, 1]) if p.std() > 1e-6 else float("nan")
        print(f"{s:<16s} {mae:8.3f} {rmse:8.3f} {r:8.3f} {mask.sum():4d}")
        res[s] = {"mae": mae, "rmse": rmse, "pearson_r": r,
                  "n": int(mask.sum())}

    with open("cache/ubfc_sqi_ablation.json", "w") as f:
        json.dump(res, f, indent=2)

    with open("cache/ubfc_sqi_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "hr_gt", "hr_forehead", "hr_uniform", "hr_sqi"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.2f}", f"{r[2]:.2f}",
                        f"{r[3]:.2f}", f"{r[4]:.2f}"])

    # Bar chart
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(STRATEGIES))
    maes = [res[s]["mae"] for s in STRATEGIES]
    rmses = [res[s]["rmse"] for s in STRATEGIES]
    ax.bar(x - 0.2, maes, 0.4, label="MAE")
    ax.bar(x + 0.2, rmses, 0.4, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(["forehead only", "uniform", "SQI-weighted"])
    ax.set_ylabel("bpm")
    ax.set_title("UBFC HR error by fusion strategy")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for i, (m, r_) in enumerate(zip(maes, rmses)):
        ax.text(i - 0.2, m + 0.15, f"{m:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, r_ + 0.15, f"{r_:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("runs/diag/sqi_ablation.png", dpi=140)
    plt.close()
    print("\n[ OK ] wrote cache/ubfc_sqi_ablation.{csv,json}")
    print("[ OK ] wrote runs/diag/sqi_ablation.png")


if __name__ == "__main__":
    main()