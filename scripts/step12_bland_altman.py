"""Bland-Altman plot for rPPG HR vs finger PPG on UBFC.

Reads cache/ubfc_hr_summary.csv (produced by step3), plots:
  x-axis: mean(rPPG HR, finger-PPG HR)
  y-axis: rPPG HR - finger-PPG HR
  lines:  bias, bias +/- 1.96 * SD (95% limits of agreement)

Two subplots:
  left  -- primary reference (finger-PPG waveform PSD, matches paper Table I)
  right -- secondary reference (oximeter HR readout, subject 20 excluded)

Output:
  runs/diag/bland_altman_hr.png
  cache/bland_altman.json
"""
from __future__ import annotations
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("runs/diag", exist_ok=True)
CSV_PATH = "cache/ubfc_hr_summary.csv"


def load_hrs():
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "subject": row["subject"],
                "hr_rppg": float(row["hr_rppg_psd"]),
                "hr_ppg":  float(row["hr_ppg_psd"]),
                "hr_gt":   float(row["hr_gt_median"]),
            })
    return rows


def bland_altman(pred, ref):
    """Return (bias, sd, loa_lo, loa_hi)."""
    pred = np.asarray(pred, float)
    ref  = np.asarray(ref, float)
    diff = pred - ref
    bias = float(diff.mean())
    sd   = float(diff.std(ddof=1))
    return bias, sd, bias - 1.96 * sd, bias + 1.96 * sd


def plot_panel(ax, pred, ref, title, xlabel, ylabel):
    mean = (pred + ref) / 2.0
    diff = pred - ref
    bias, sd, loa_lo, loa_hi = bland_altman(pred, ref)

    ax.scatter(mean, diff, s=26, alpha=0.75,
               edgecolor="k", linewidth=0.4, zorder=3)
    ax.axhline(bias,   color="r", lw=1.2, zorder=2,
               label=f"bias = {bias:+.2f} bpm")
    ax.axhline(loa_lo, color="r", ls="--", lw=1.0, zorder=2,
               label=f"95% LoA = [{loa_lo:+.2f}, {loa_hi:+.2f}]")
    ax.axhline(loa_hi, color="r", ls="--", lw=1.0, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    return {"bias": bias, "sd": sd, "loa_lo": loa_lo, "loa_hi": loa_hi,
            "n": int(len(pred))}


def main():
    rows = load_hrs()
    print(f"Loaded {len(rows)} subjects")

    # Primary reference: finger-PPG waveform PSD
    pred_ppg = np.array([r["hr_rppg"] for r in rows])
    ref_ppg  = np.array([r["hr_ppg"]  for r in rows])

    # Secondary: oximeter HR readout, subject 20 excluded
    # (subject 20 has a corrupted oximeter channel; the CSV row still
    # contains the corrupted GT value)
    rows_gt = [r for r in rows if r["subject"] != "subject20"]
    pred_gt = np.array([r["hr_rppg"] for r in rows_gt])
    ref_gt  = np.array([r["hr_gt"]   for r in rows_gt])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    metrics_ppg = plot_panel(
        axes[0], pred_ppg, ref_ppg,
        "vs. finger-PPG waveform (primary reference)",
        "mean of rPPG HR and PPG HR (bpm)",
        "rPPG HR - PPG HR (bpm)",
    )
    metrics_gt = plot_panel(
        axes[1], pred_gt, ref_gt,
        "vs. oximeter HR readout (subject 20 excluded)",
        "mean of rPPG HR and GT HR (bpm)",
        "rPPG HR - GT HR (bpm)",
    )

    plt.tight_layout()
    out_png = "runs/diag/bland_altman_hr.png"
    plt.savefig(out_png, dpi=140)
    plt.close()

    print(f"\nPrimary reference (finger-PPG waveform PSD):")
    print(f"  n      = {metrics_ppg['n']}")
    print(f"  bias   = {metrics_ppg['bias']:+.3f} bpm")
    print(f"  SD     = {metrics_ppg['sd']:.3f} bpm")
    print(f"  95% LoA= [{metrics_ppg['loa_lo']:+.3f}, "
          f"{metrics_ppg['loa_hi']:+.3f}]")

    print(f"\nSecondary reference (oximeter readout, subj 20 excl.):")
    print(f"  n      = {metrics_gt['n']}")
    print(f"  bias   = {metrics_gt['bias']:+.3f} bpm")
    print(f"  SD     = {metrics_gt['sd']:.3f} bpm")
    print(f"  95% LoA= [{metrics_gt['loa_lo']:+.3f}, "
          f"{metrics_gt['loa_hi']:+.3f}]")

    with open("cache/bland_altman.json", "w", encoding="utf-8") as f:
        json.dump({"primary_vs_ppg_waveform": metrics_ppg,
                   "secondary_vs_oximeter": metrics_gt}, f, indent=2)

    print(f"\n[ OK ] wrote {out_png}")
    print(f"[ OK ] wrote cache/bland_altman.json")


if __name__ == "__main__":
    main()