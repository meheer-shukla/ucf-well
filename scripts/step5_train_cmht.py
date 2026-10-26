"""Step 5 — CMHT alignment ablation on synthetic-rPPG WESAD.

Design:
  - 6-D HRV features (mean_hr, sdnn, cvnn, mean_ibi, median_ibi, sd2)
  - Target domain = synthetic-rPPG WESAD (built by step7)
  - Test set     = same synthetic-rPPG WESAD, held-out subjects
  - Paired Wilcoxon signed-rank on per-subject macro-F1 (n=15)

Configurations (rows of Table III):
  A  source_only  : train WESAD ECG, no alignment
  B  wrist_only   : train WESAD wrist PPG, no alignment
  C  jitter_only  : train WESAD ECG + IBI jitter
  D  bridge_only  : train WESAD ECG, CORAL align to WESAD wrist PPG
  E  full_cmht    : train WESAD ECG, align to PPG, then to synthetic-rPPG

Run:
  python scripts/step5_train_cmht_np.py --seeds 5
  python scripts/step5_train_cmht_np.py --seeds 5 --binary

Output:
  cache/cmht_synth_3class.json
  cache/cmht_synth_binary.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
from scipy import stats as scipy_stats
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, cohen_kappa_score,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# 6-D feature indices matching step7
KEEP_6D = [0, 1, 4, 5, 6, 13]


def load_data(binary: bool = False):
    z_ecg = np.load("cache/wesad_ecg_hrv.npz")
    z_bvp = np.load("cache/wesad_bvp_hrv.npz")
    z_syn = np.load("cache/wesad_synthetic_rppg.npz")

    X_ecg = z_ecg["X"][:, KEEP_6D].astype(np.float64)
    X_bvp = z_bvp["X"][:, KEEP_6D].astype(np.float64)
    X_syn = z_syn["X"].astype(np.float64)
    y     = z_ecg["y"].astype(np.int64)
    g     = z_ecg["groups"].astype(np.int64)

    if binary:
        m = (y == 0) | (y == 1)
        X_ecg, X_bvp, X_syn, y, g = X_ecg[m], X_bvp[m], X_syn[m], y[m], g[m]

    assert len(X_ecg) == len(X_bvp) == len(X_syn)
    return X_ecg, X_bvp, X_syn, y, g


def zscore_fit(*arrays):
    X = np.vstack(arrays)
    mu = X.mean(0, keepdims=True)
    sd = np.maximum(X.std(0, keepdims=True), 1e-3)
    return mu, sd


def coral_fit(Xs, Xt, eps=1e-3):
    d = Xs.shape[1]
    ms = Xs.mean(0, keepdims=True)
    mt = Xt.mean(0, keepdims=True)
    Xs = Xs - ms; Xt = Xt - mt
    Cs = (Xs.T @ Xs) / max(1, len(Xs) - 1) + eps * np.eye(d)
    Ct = (Xt.T @ Xt) / max(1, len(Xt) - 1) + eps * np.eye(d)
    ds, Vs = np.linalg.eigh(Cs); dt, Vt = np.linalg.eigh(Ct)
    ds = np.clip(ds, eps, None); dt = np.clip(dt, eps, None)
    A = (Vs @ np.diag(1/np.sqrt(ds)) @ Vs.T) @ (Vt @ np.diag(np.sqrt(dt)) @ Vt.T)
    return A, ms, mt


def coral_apply(X, A, ms, mt):
    return (X - ms) @ A + mt


def jitter_augment(X, y, n_copies=2, sigma_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    for _ in range(n_copies):
        Xj = X.copy()
        # jitter features 0..3 (all timing-derived in 6-D: hr, sdnn, cvnn, mean_ibi)
        # leave 4 (median_ibi) and 5 (sd2) untouched — less noise-sensitive
        sd = X[:, :4].std(0, keepdims=True) + 1e-6
        Xj[:, :4] += rng.normal(size=X[:, :4].shape) * sd * sigma_frac
        Xs.append(Xj); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def oversample(X, y, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    counts = np.array([(y == c).sum() for c in classes])
    tgt = counts.max()
    Xs, ys = [X], [y]
    for c in classes:
        extra = tgt - (y == c).sum()
        if extra <= 0: continue
        idx = np.where(y == c)[0]
        ch = rng.choice(idx, size=extra, replace=True)
        Xs.append(X[ch]); ys.append(y[ch])
    return np.vstack(Xs), np.concatenate(ys)


def make_clf(seed=0):
    return MLPClassifier(
        hidden_layer_sizes=(32, 32), activation="relu", solver="adam",
        alpha=1e-3, batch_size=64, learning_rate_init=1e-3,
        max_iter=500, early_stopping=True, validation_fraction=0.2,
        n_iter_no_change=25, random_state=seed,
    )


def train_one_fold(X_tr_ecg, X_tr_bvp, X_tr_syn, y_tr,
                   X_te_syn, mode, seed=0):
    X_train = X_tr_ecg.copy()
    y_train = y_tr.copy()

    if mode == "source_only":
        pass
    elif mode == "wrist_only":
        X_train = X_tr_bvp.copy()
    elif mode == "jitter_only":
        X_train, y_train = jitter_augment(X_tr_ecg, y_tr,
                                          n_copies=2, sigma_frac=0.15,
                                          seed=seed)
    elif mode == "bridge_only":
        A, ms, mt = coral_fit(X_tr_ecg, X_tr_bvp)
        X_train = coral_apply(X_train, A, ms, mt)
        X_train, y_train = jitter_augment(X_train, y_train,
                                          n_copies=2, sigma_frac=0.15,
                                          seed=seed)
    elif mode == "full_cmht":
        # Stage 1: ECG -> wrist PPG
        A1, ms1, mt1 = coral_fit(X_tr_ecg, X_tr_bvp)
        X_train = coral_apply(X_train, A1, ms1, mt1)
        # Stage 2: bridge -> synthetic rPPG (baseline-only)
        bl = (y_tr == 0)
        if bl.sum() >= 3:
            A2, ms2, mt2 = coral_fit(X_train[bl], X_tr_syn)
            X_train = coral_apply(X_train, A2, ms2, mt2)
        X_train, y_train = jitter_augment(X_train, y_train,
                                          n_copies=2, sigma_frac=0.15,
                                          seed=seed)
    else:
        raise ValueError(mode)

    X_train, y_train = oversample(X_train, y_train, seed=seed)

    clf = make_clf(seed=seed)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_te_syn)
    return pred


def per_subject_scores(y_true, y_pred, groups, n_classes):
    scores = {}
    for sid in np.unique(groups):
        m = groups == sid
        scores[sid] = f1_score(y_true[m], y_pred[m],
                               average="macro",
                               labels=list(range(n_classes)),
                               zero_division=0)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = ("cache/cmht_synth_binary.json" if args.binary
                    else "cache/cmht_synth_3class.json")

    X_ecg, X_bvp, X_syn, y, g = load_data(binary=args.binary)
    n_classes = 2 if args.binary else 3
    class_names = (["baseline", "stress"] if args.binary
                   else ["baseline", "stress", "amusement"])

    print(f"Task        : {n_classes}-class")
    print(f"Seeds       : {args.seeds}")
    print(f"X_ecg       : {X_ecg.shape}")
    print(f"X_bvp       : {X_bvp.shape}")
    print(f"X_synth rPPG: {X_syn.shape}  (target AND test domain)")
    print(f"Labels      : {np.bincount(y)}")
    print(f"Subjects    : {len(np.unique(g))}")
    print()

    mu, sd = zscore_fit(X_ecg, X_bvp, X_syn)
    X_ecg = (X_ecg - mu) / sd
    X_bvp = (X_bvp - mu) / sd
    X_syn = (X_syn - mu) / sd

    MODES = ["source_only", "wrist_only", "jitter_only",
             "bridge_only", "full_cmht"]

    per_mode_per_subject = {m: {} for m in MODES}

    for mode in MODES:
        print("=" * 78)
        print(f"MODE: {mode.upper()}")
        print("=" * 78)

        subj_f1_by_seed = {sid: [] for sid in np.unique(g)}

        for seed in range(args.seeds):
            gkf = GroupKFold(n_splits=5)
            yt_all, yp_all, gt_all = [], [], []
            for fold, (tr, te) in enumerate(gkf.split(X_ecg, y, g)):
                pred = train_one_fold(
                    X_ecg[tr], X_bvp[tr], X_syn[tr], y[tr],
                    X_syn[te], mode=mode, seed=seed * 100 + fold,
                )
                yt_all.append(y[te]); yp_all.append(pred); gt_all.append(g[te])

            yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
            gt = np.concatenate(gt_all)
            s = per_subject_scores(yt, yp, gt, n_classes)
            for sid, val in s.items():
                subj_f1_by_seed[sid].append(val)

            pooled_f1 = f1_score(yt, yp, average="macro",
                                 labels=list(range(n_classes)),
                                 zero_division=0)
            acc = accuracy_score(yt, yp)
            print(f"  seed {seed}  pooled F1={pooled_f1:.4f}  "
                  f"acc={acc:.4f}")

        for sid in subj_f1_by_seed:
            vals = subj_f1_by_seed[sid]
            per_mode_per_subject[mode][sid] = float(np.mean(vals))

        all_sids = sorted(per_mode_per_subject[mode].keys())
        subj_vals = [per_mode_per_subject[mode][s] for s in all_sids]
        print(f"  -> per-subject mean F1 = {np.mean(subj_vals):.4f} "
              f"+/- {np.std(subj_vals, ddof=1):.4f}  "
              f"(n={len(subj_vals)} subjects)")
        print()

    # ==================================================================
    # Final comparison with paired Wilcoxon
    # ==================================================================
    print("=" * 84)
    print("FINAL COMPARISON  (per-subject macro-F1, n=15)")
    print("=" * 84)
    print(f"{'Method':<16s} {'F1 (mean+/-std)':>18s}  "
          f"{'d vs source':>12s}  {'Wilcoxon p':>12s}")
    print("-" * 84)

    source_sids = sorted(per_mode_per_subject["source_only"].keys())
    src_vals = np.array([per_mode_per_subject["source_only"][s]
                         for s in source_sids])

    results = {}
    for mode in MODES:
        vals = np.array([per_mode_per_subject[mode][s] for s in source_sids])
        delta = vals.mean() - src_vals.mean()
        if mode == "source_only":
            p_val = float("nan")
        else:
            try:
                _, p_val = scipy_stats.wilcoxon(src_vals, vals,
                                                zero_method="wilcox")
            except Exception:
                p_val = float("nan")
        marker = ("[+]" if (p_val < 0.05 and delta > 0) else
                  "[-]" if (p_val < 0.05 and delta < 0) else "[.]")
        print(f"{mode:<16s} {vals.mean():>8.4f} +/- {vals.std(ddof=1):.4f}   "
              f"{delta:>+10.4f}  {p_val:>10.4f}  {marker}")
        results[mode] = {
            "f1_mean": float(vals.mean()),
            "f1_std": float(vals.std(ddof=1)),
            "per_subject": dict(zip(map(str, source_sids), vals.tolist())),
            "delta_vs_source": float(delta),
            "wilcoxon_p": float(p_val) if not np.isnan(p_val) else None,
        }

    out = {
        "n_classes": n_classes,
        "class_names": class_names,
        "n_seeds": args.seeds,
        "n_subjects": len(source_sids),
        "feature_set": "6-D (mean_hr, sdnn, cvnn, mean_ibi, median_ibi, sd2)",
        "methods": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[ OK ] wrote {args.out}")


if __name__ == "__main__":
    main()