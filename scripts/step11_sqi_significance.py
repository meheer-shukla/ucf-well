"""Paired tests for SQI vs uniform vs forehead-only fusion on UBFC.

Reads cache/ubfc_sqi_ablation.csv (produced by step8), computes
absolute HR errors for each strategy, and runs paired Wilcoxon
signed-rank tests between strategies.

Also reports bootstrapped 95% confidence intervals for the mean
absolute error of each strategy and for pairwise mean differences.

Output:
  cache/sqi_significance.json
  runs/diag/sqi_significance.txt
"""
from __future__ import annotations
import os, csv, json
import numpy as np
from scipy import stats

# Force UTF-8 for all print/file operations on Windows
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

CSV_PATH = "cache/ubfc_sqi_ablation.csv"
OUT_JSON = "cache/sqi_significance.json"
OUT_TXT  = "runs/diag/sqi_significance.txt"

os.makedirs("runs/diag", exist_ok=True)


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0):
    """Return (mean, low, high) of x using percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(x.mean()), lo, hi


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                      alpha: float = 0.05, seed: int = 1):
    """Bootstrap CI for mean(a) - mean(b) assuming paired samples."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return float(a.mean() - b.mean()), lo, hi


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d: mean(diff) / std(diff)."""
    diff = a - b
    sd = float(diff.std(ddof=1))
    if sd < 1e-9:
        return 0.0
    return float(diff.mean() / sd)


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"[FAIL] {CSV_PATH} not found. Run step8 first.")

    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((
                float(row["hr_gt"]),
                float(row["hr_forehead"]),
                float(row["hr_uniform"]),
                float(row["hr_sqi"]),
            ))

    arr = np.array(rows)
    gt = arr[:, 0]
    e_fore = np.abs(arr[:, 1] - gt)
    e_uni  = np.abs(arr[:, 2] - gt)
    e_sqi  = np.abs(arr[:, 3] - gt)

    n = len(gt)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 72)
    out(f"SQI FUSION SIGNIFICANCE TEST  (n = {n} subjects)")
    out("=" * 72)
    out()

    # Absolute-error summary with bootstrap CIs
    out("Absolute HR error (bpm), mean [95% bootstrap CI]:")
    summary = {}
    for name, e in [("forehead_only", e_fore),
                    ("uniform",       e_uni),
                    ("sqi_fusion",    e_sqi)]:
        m, lo, hi = bootstrap_ci(e)
        out(f"  {name:14s}  {m:5.3f}  [{lo:5.3f}, {hi:5.3f}]")
        summary[name] = {"mae": m, "mae_ci_lo": lo, "mae_ci_hi": hi}
    out()

    # Paired Wilcoxon tests
    out("Paired Wilcoxon signed-rank tests:")
    pairs = [
        ("SQI vs forehead",      e_sqi, e_fore),
        ("SQI vs uniform",       e_sqi, e_uni),
        ("uniform vs forehead",  e_uni, e_fore),
    ]
    for name, a, b in pairs:
        try:
            stat, p = stats.wilcoxon(a, b, zero_method="wilcox")
        except ValueError:
            stat, p = float("nan"), float("nan")
        d = cohens_d_paired(a, b)
        diff, lo, hi = bootstrap_diff_ci(a, b)
        sig = "*" if p < 0.05 else " "
        out(f"  {name:24s}  W={stat:7.1f}  p={p:.4f}{sig}  "
            f"d={d:+.3f}  dMAE={diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        summary[name.replace(" ", "_")] = {
            "wilcoxon_stat": float(stat) if np.isfinite(stat) else None,
            "p_value": float(p) if np.isfinite(p) else None,
            "cohens_d": d,
            "mean_diff": diff,
            "diff_ci_lo": lo,
            "diff_ci_hi": hi,
        }
    out()
    out("Significance codes: * p < 0.05; no multiple-comparison correction")
    out()

    # Verdict
    p_sqi_uni  = summary.get("SQI_vs_uniform", {}).get("p_value")
    p_sqi_fore = summary.get("SQI_vs_forehead", {}).get("p_value")

    if p_sqi_uni is not None and p_sqi_uni < 0.05:
        out("[result] SQI beats uniform at p < 0.05")
    elif p_sqi_uni is not None:
        out(f"[result] SQI vs uniform: p = {p_sqi_uni:.4f} -- "
            f"not significant at alpha = 0.05")

    if p_sqi_fore is not None and p_sqi_fore < 0.05:
        out("[result] SQI beats forehead-only at p < 0.05")
    elif p_sqi_fore is not None:
        out(f"[result] SQI vs forehead-only: p = {p_sqi_fore:.4f} -- "
            f"not significant at alpha = 0.05")

    # Save
    summary["n_subjects"] = n
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[ OK ] wrote {OUT_JSON}")
    print(f"[ OK ] wrote {OUT_TXT}")


if __name__ == "__main__":
    main()