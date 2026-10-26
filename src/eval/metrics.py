"""Metric suite for UCF-Well.

Regression
    mae, rmse, mape, pearson_r, spearman_rho, ccc,
    bland_altman, within_loa_pct, regression_report

Classification
    accuracy, balanced_accuracy, macro_f1, weighted_f1, cohen_kappa,
    mcc, per_class_prf, confusion, auc_roc_macro, classification_report

All functions are scalar-in / scalar-out except the *_report functions,
which return dicts ready for JSON serialisation into the paper tables.
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef, roc_auc_score,
)


# ======================================================================
# Regression metrics
# ======================================================================
def _flat(x):
    return np.asarray(x, dtype=np.float64).ravel()


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Mean absolute percentage error (%).  Undefined for y_true=0; we mask."""
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def pearson_r(y_true, y_pred) -> float:
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    if len(y_true) < 2 or y_true.std() < 1e-9 or y_pred.std() < 1e-9:
        return float("nan")
    return float(stats.pearsonr(y_true, y_pred)[0])


def spearman_rho(y_true, y_pred) -> float:
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    if len(y_true) < 2:
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred)[0])


def ccc(y_true, y_pred) -> float:
    """Lin's concordance correlation coefficient.

    CCC = 2 * cov(a,b) / (var(a) + var(b) + (mean(a) - mean(b))^2)
    Range [-1, 1], penalises both scatter AND scale/offset disagreement.
    This is the standard agreement metric in rPPG papers alongside Bland-Altman.
    """
    a, b = _flat(y_true), _flat(y_pred)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = np.mean((a - ma) * (b - mb))
    denom = va + vb + (ma - mb) ** 2
    if denom < 1e-12:
        return float("nan")
    return float(2 * cov / denom)


def bland_altman(y_true, y_pred):
    """Bland-Altman analysis.

    Returns dict with:
        bias        mean(y_true - y_pred)
        sd          std(y_true - y_pred, ddof=1)
        loa_lower   bias - 1.96*sd
        loa_upper   bias + 1.96*sd
        diff        per-sample differences (array)
    """
    a, b = _flat(y_true), _flat(y_pred)
    d = a - b
    bias = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return {
        "bias": bias,
        "sd": sd,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "diff": d,
    }


def within_loa_pct(y_true, y_pred) -> float:
    """Fraction of samples inside the +-1.96 SD Bland-Altman limits (%)."""
    ba = bland_altman(y_true, y_pred)
    d = ba["diff"]
    inside = (d >= ba["loa_lower"]) & (d <= ba["loa_upper"])
    return float(inside.mean() * 100.0)


def regression_report(y_true, y_pred, unit: str = "", name: str = "") -> dict:
    """Aggregate all regression metrics into one dict."""
    ba = bland_altman(y_true, y_pred)
    return {
        "name": name,
        "n": int(len(_flat(y_true))),
        "unit": unit,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_rho": spearman_rho(y_true, y_pred),
        "ccc": ccc(y_true, y_pred),
        "ba_bias": ba["bias"],
        "ba_loa_lower": ba["loa_lower"],
        "ba_loa_upper": ba["loa_upper"],
        "within_loa_pct": within_loa_pct(y_true, y_pred),
    }


def format_regression_table_row(rep: dict) -> str:
    u = rep["unit"]
    return (f"{rep['name']:14s}  n={rep['n']:5d}  "
            f"MAE={rep['mae']:6.2f}{u}  RMSE={rep['rmse']:6.2f}{u}  "
            f"r={rep['pearson_r']:+.3f}  CCC={rep['ccc']:+.3f}  "
            f"BA=[{rep['ba_loa_lower']:+.2f},{rep['ba_loa_upper']:+.2f}]{u}  "
            f"LoA%={rep['within_loa_pct']:5.1f}")


# ======================================================================
# Classification metrics
# ======================================================================
def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def weighted_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def cohen_kappa(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred))


def mcc(y_true, y_pred) -> float:
    return float(matthews_corrcoef(y_true, y_pred))


def per_class_prf(y_true, y_pred, n_classes: int | None = None):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=None if n_classes is None else list(range(n_classes)),
        zero_division=0)
    return {"precision": p.tolist(), "recall": r.tolist(),
            "f1": f.tolist(), "support": s.tolist()}


def confusion(y_true, y_pred, n_classes: int | None = None):
    labels = None if n_classes is None else list(range(n_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def auc_roc_macro(y_true, y_proba, n_classes: int) -> float:
    """One-vs-rest macro AUC. y_proba: (N, n_classes)."""
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr",
                                   average="macro", labels=list(range(n_classes))))
    except ValueError:
        return float("nan")


def classification_report(y_true, y_pred, class_names=None,
                          y_proba=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(np.unique(y_true))
    rep = {
        "n": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "weighted_f1": weighted_f1(y_true, y_pred),
        "cohen_kappa": cohen_kappa(y_true, y_pred),
        "mcc": mcc(y_true, y_pred),
        "per_class": per_class_prf(y_true, y_pred, n_classes=n),
        "confusion": confusion(y_true, y_pred, n_classes=n).tolist(),
    }
    if class_names is not None:
        rep["class_names"] = list(class_names)
    if y_proba is not None:
        rep["auc_roc_macro"] = auc_roc_macro(y_true, y_proba, n_classes=n)
    return rep


def format_classification_summary(rep: dict) -> str:
    lines = [
        f"  n              : {rep['n']}",
        f"  accuracy       : {rep['accuracy']:.4f}",
        f"  balanced acc   : {rep['balanced_accuracy']:.4f}",
        f"  macro F1       : {rep['macro_f1']:.4f}",
        f"  weighted F1    : {rep['weighted_f1']:.4f}",
        f"  Cohen kappa    : {rep['cohen_kappa']:.4f}",
        f"  MCC            : {rep['mcc']:.4f}",
    ]
    if "auc_roc_macro" in rep:
        lines.append(f"  macro ROC AUC  : {rep['auc_roc_macro']:.4f}")
    return "\n".join(lines)


# ======================================================================
# Cross-modal feature-space distance (for the TS-CMHT paper figure)
# ======================================================================
def coral_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Squared Frobenius distance between source/target second-moment matrices.

    This is the *value* of the CORAL loss term.  We use it to quantify the
    modality gap between ECG-HRV, BVP-HRV, and (later) rPPG-HRV features:
        d(ECG, BVP)  -- "modality gap within contact"
        d(ECG, rPPG) -- "full gap to target"
        d(BVP, rPPG) -- "remaining gap after bridge"

    If TS-CMHT is the right design, we expect:
        d(ECG, BVP)   <<  d(ECG, rPPG)
        d(BVP, rPPG)  <   d(ECG, rPPG)
    """
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    d = X.shape[1]
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Cx = (Xc.T @ Xc) / max(1, X.shape[0] - 1)
    Cy = (Yc.T @ Yc) / max(1, Y.shape[0] - 1)
    return float(np.sum((Cx - Cy) ** 2) / (4 * d * d))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Maximum Mean Discrepancy with RBF kernel — a second view of the gap.

    Included because CORAL only sees second-order statistics; MMD sees the
    full kernel-embedded distribution.  If CORAL and MMD give the same
    ordering of {ECG, BVP, rPPG} pairs, that's a robustness point for the
    paper.
    """
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    if gamma is None:
        # median heuristic on pooled pairwise distances
        Z = np.vstack([X, Y])
        d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / (med + 1e-12)

    def k(a, b):
        return np.exp(-gamma * ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))

    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())