"""Confusion matrices for source_only vs full_cmht on WESAD.

Re-runs stress training for the two extreme modes on both binary
and 3-class tasks. Pools predictions across all seeds x folds, then
plots a 2x2 grid of confusion matrices:

    row 1: binary      (source_only | full_cmht)
    row 2: 3-class     (source_only | full_cmht)

The F1 reported in each subplot title is the *pooled* macro-F1
computed from concatenated predictions across all subjects, seeds
and folds.  This differs from the *per-subject mean* macro-F1
reported in Table III, which averages per-subject F1 scores and is
used for the paired Wilcoxon test.  Both statistics are correct;
they answer different questions and are labelled explicitly to
avoid confusion.

Output:
  cache/confusion_matrices.json
  runs/diag/confusion_matrices.png
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from step5_train_cmht import load_data, zscore_fit, train_one_fold

SEEDS = 5
MODES = ["source_only", "full_cmht"]

BINARY_NAMES = ["baseline", "stress"]
MULTI_NAMES = ["baseline", "stress", "amusement"]


def pool_predictions(mode, X_ecg, X_bvp, X_syn, y, g):
    """Train 5 seeds x 5 folds, pool all held-out predictions."""
    yt_all, yp_all = [], []
    for seed in range(SEEDS):
        gkf = GroupKFold(n_splits=5)
        for fold, (tr, te) in enumerate(gkf.split(X_ecg, y, g)):
            pred = train_one_fold(
                X_ecg[tr], X_bvp[tr], X_syn[tr], y[tr],
                X_syn[te], mode=mode, seed=seed * 100 + fold,
            )
            yt_all.append(y[te])
            yp_all.append(pred)
    return np.concatenate(yt_all), np.concatenate(yp_all)


def run_task(binary: bool):
    X_ecg, X_bvp, X_syn, y, g = load_data(binary=binary)
    mu, sd = zscore_fit(X_ecg, X_bvp, X_syn)
    X_ecg = (X_ecg - mu) / sd
    X_bvp = (X_bvp - mu) / sd
    X_syn = (X_syn - mu) / sd

    n_classes = 2 if binary else 3
    class_names = BINARY_NAMES if binary else MULTI_NAMES
    task = "binary" if binary else "3class"

    results = {}
    for mode in MODES:
        print(f"  [{task}] {mode} ...")
        yt, yp = pool_predictions(mode, X_ecg, X_bvp, X_syn, y, g)
        cm = confusion_matrix(yt, yp, labels=list(range(n_classes)))
        # Pooled macro-F1: computed from concatenated predictions
        f1 = f1_score(yt, yp, average="macro",
                      labels=list(range(n_classes)), zero_division=0)
        results[mode] = {
            "confusion": cm.tolist(),
            "macro_f1_pooled": float(f1),
            "n_test": int(len(yt)),
            "class_names": class_names,
        }
        print(f"    pooled macro-F1 = {f1:.4f},  n = {len(yt)}")
        print(f"    confusion:")
        for row in cm:
            print(f"      {row.tolist()}")

    return results


def plot_matrix(ax, cm, class_names, title):
    cm = np.asarray(cm)
    n = cm.shape[0]
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10)

    # Annotate each cell with the count
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > vmax / 2 else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color=color, fontsize=10)

    # Draw grid lines between cells
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", linewidth=1)
        ax.axvline(k - 0.5, color="white", linewidth=1)

    return im


def main():
    os.makedirs("runs/diag", exist_ok=True)

    print("=" * 72)
    print("Confusion matrices -- source_only vs full_cmht")
    print("=" * 72)
    print()
    print("NOTE: F1 values shown below are POOLED macro-F1, computed from")
    print("      concatenated predictions across all subjects, seeds, and")
    print("      folds. Table III in the paper reports PER-SUBJECT MEAN")
    print("      macro-F1, which differs because subjects have different")
    print("      class balances. Both statistics are correct.")
    print()

    print("Binary task")
    bin_res = run_task(binary=True)
    print()
    print("3-class task")
    multi_res = run_task(binary=False)

    # Save JSON
    out = {
        "binary": bin_res,
        "3class": multi_res,
        "n_seeds": SEEDS,
        "statistic": "pooled macro-F1 (concatenated predictions)",
        "note": ("Pooled F1 differs from the per-subject mean F1 in "
                 "Table III of the paper."),
    }
    with open("cache/confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Figure: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.8))

    # Row 1: binary
    plot_matrix(
        axes[0, 0],
        bin_res["source_only"]["confusion"],
        BINARY_NAMES,
        f"Binary -- source_only\n"
        f"(pooled macro-F1 = {bin_res['source_only']['macro_f1_pooled']:.3f})"
    )
    plot_matrix(
        axes[0, 1],
        bin_res["full_cmht"]["confusion"],
        BINARY_NAMES,
        f"Binary -- full CMHT\n"
        f"(pooled macro-F1 = {bin_res['full_cmht']['macro_f1_pooled']:.3f})"
    )

    # Row 2: 3-class
    plot_matrix(
        axes[1, 0],
        multi_res["source_only"]["confusion"],
        MULTI_NAMES,
        f"3-class -- source_only\n"
        f"(pooled macro-F1 = {multi_res['source_only']['macro_f1_pooled']:.3f})"
    )
    plot_matrix(
        axes[1, 1],
        multi_res["full_cmht"]["confusion"],
        MULTI_NAMES,
        f"3-class -- full CMHT\n"
        f"(pooled macro-F1 = {multi_res['full_cmht']['macro_f1_pooled']:.3f})"
    )

    plt.tight_layout()
    out_png = "runs/diag/confusion_matrices.png"
    plt.savefig(out_png, dpi=140)
    plt.close()

    # Summary
    print()
    print("=" * 72)
    print("SUMMARY  (pooled macro-F1, computed from concatenated predictions)")
    print("=" * 72)
    print(f"Binary  source_only : "
          f"{bin_res['source_only']['macro_f1_pooled']:.4f}")
    print(f"Binary  full_cmht   : "
          f"{bin_res['full_cmht']['macro_f1_pooled']:.4f}")
    print(f"3-class source_only : "
          f"{multi_res['source_only']['macro_f1_pooled']:.4f}")
    print(f"3-class full_cmht   : "
          f"{multi_res['full_cmht']['macro_f1_pooled']:.4f}")
    print()

    # Recall summary (the paper-worthy finding)
    print("Stress-class recall (fraction of true stress predicted as stress):")
    for task_name, res in [("binary", bin_res), ("3-class", multi_res)]:
        for mode in MODES:
            cm = np.array(res[mode]["confusion"])
            if task_name == "binary":
                stress_row = cm[1]         # row index 1 = stress
            else:
                stress_row = cm[1]         # row 1 = stress in 3-class too
            total = stress_row.sum()
            correct = stress_row[1]        # predicted stress
            recall = correct / total if total > 0 else 0.0
            print(f"  {task_name:8s} {mode:12s} : "
                  f"{correct:4d}/{total:4d} = {recall * 100:.1f}%")
    print()

    print(f"[ OK ] wrote {out_png}")
    print(f"[ OK ] wrote cache/confusion_matrices.json")


if __name__ == "__main__":
    main()