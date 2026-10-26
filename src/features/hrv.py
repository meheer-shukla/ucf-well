"""14-D HRV feature vector.

Modality-agnostic: the same function is used for chest ECG, wrist BVP,
and camera rPPG. This is what makes cross-modal transfer (CMHT) possible.

Index  Feature          Unit      Notes
-----  ---------------  --------  --------------------------------------
 0     mean_hr          bpm
 1     sdnn             ms        std of NN intervals
 2     rmssd            ms        root mean square successive diff
 3     pnn50            %         fraction of successive diff > 50 ms
 4     cvnn             %         coefficient of variation
 5     mean_ibi         ms
 6     median_ibi       ms        robust to outliers
 7     ibi_iqr          ms
 8     lf_power         ms^2      0.04-0.15 Hz
 9     hf_power         ms^2      0.15-0.40 Hz
10     lf_hf_ratio      -         sympathovagal balance
11     total_power      ms^2      0.003-0.40 Hz
12     sd1              ms        Poincare perpendicular
13     sd2              ms        Poincare parallel
"""
from __future__ import annotations
import numpy as np
from scipy import interpolate, signal as sps


# NumPy 2.0 renamed np.trapz -> np.trapezoid.  Support both.
try:
    _trapz = np.trapezoid        # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz            # NumPy < 2.0


FEATURE_NAMES = [
    "mean_hr", "sdnn", "rmssd", "pnn50", "cvnn",
    "mean_ibi", "median_ibi", "ibi_iqr",
    "lf_power", "hf_power", "lf_hf_ratio", "total_power",
    "sd1", "sd2",
]
N_FEATURES = len(FEATURE_NAMES)


# ======================================================================
# Helpers
# ======================================================================
def _bandpower(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    """Trapz-integrated power in [lo, hi).

    Uses `_trapz` (np.trapezoid on NumPy >= 2.0, np.trapz on older).
    """
    m = (f >= lo) & (f < hi)
    if m.sum() == 0:
        return 0.0
    return float(_trapz(pxx[m], f[m]))


def correct_missed_beats(ibi: np.ndarray, factor: float = 1.7) -> np.ndarray:
    """Split IBIs longer than factor * median into equal parts."""
    ibi = np.asarray(ibi, dtype=np.float64)
    if len(ibi) < 5:
        return ibi
    med = float(np.median(ibi))
    if med <= 0:
        return ibi
    out = []
    for v in ibi:
        if v > factor * med:
            k = max(2, int(round(v / med)))
            out.extend([v / k] * k)
        else:
            out.append(v)
    return np.array(out, dtype=np.float64)


def _safe_tachogram(ibi_sec: np.ndarray, fs_interp: float):
    """Interpolate the IBI sequence onto a uniform grid (fs_interp Hz).

    Returns (t_new, ibi_ms_interp) or (None, None) if resampling is
    impossible.  Robust to non-monotone inputs, duplicate cumulative
    timestamps, and very short sequences.
    """
    ibi_sec = np.asarray(ibi_sec, dtype=np.float64)
    if len(ibi_sec) < 20:
        return None, None

    t = np.cumsum(ibi_sec)
    t = t - t[0]
    if t[-1] < 20.0:
        return None, None

    # Deduplicate cumulative timestamps: enforce strictly increasing
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t = t[keep]
    ibi_ms = (ibi_sec * 1000.0)[keep]
    if len(t) < 20:
        return None, None

    try:
        f_interp = interpolate.interp1d(t, ibi_ms, kind="linear",
                                        bounds_error=False,
                                        fill_value="extrapolate")
    except Exception as e:
        print(f"    [hrv] interp1d failed: {e}")
        return None, None

    t_new = np.arange(0, t[-1], 1.0 / fs_interp)
    try:
        ibi_i = sps.detrend(f_interp(t_new))
    except Exception as e:
        print(f"    [hrv] detrend failed: {e}")
        return None, None

    return t_new, ibi_i


# ======================================================================
# Main extractor
# ======================================================================
def hrv_features(ibi: np.ndarray, fs_interp: float = 4.0):
    """ibi: inter-beat intervals in SECONDS.  Returns 14-D vector or None.

    Requires >= 20 IBIs and >= 20 s of cumulative duration.
    """
    ibi = np.asarray(ibi, dtype=np.float64)
    ibi = ibi[np.isfinite(ibi)]
    ibi = ibi[ibi > 0]
    if len(ibi) < 20:
        return None

    # Recover missed beats before computing anything
    ibi = correct_missed_beats(ibi)

    # ------------------------------------------------------------------
    # Time-domain (all in ms)
    # ------------------------------------------------------------------
    mean_ibi = float(ibi.mean())            # seconds
    mean_hr  = 60.0 / mean_ibi              # bpm

    sdnn = float(ibi.std(ddof=1) * 1000.0)
    diff_ms = np.diff(ibi) * 1000.0
    rmssd = float(np.sqrt(np.mean(diff_ms ** 2))) if len(diff_ms) else 0.0
    pnn50 = float(np.mean(np.abs(diff_ms) > 50.0) * 100.0) if len(diff_ms) else 0.0
    cvnn  = float(sdnn / (mean_ibi * 1000.0) * 100.0)

    q25, q75 = np.percentile(ibi, [25, 75])
    ibi_iqr = float((q75 - q25) * 1000.0)

    # Poincare
    x, y = ibi[:-1], ibi[1:]
    sd1 = float(np.std((x - y) / np.sqrt(2)) * 1000.0)
    sd2 = float(np.std((x + y) / np.sqrt(2)) * 1000.0)

    # ------------------------------------------------------------------
    # Frequency-domain
    # ------------------------------------------------------------------
    lf = hf = tot = lf_hf = 0.0
    t_new, ibi_i = _safe_tachogram(ibi, fs_interp)
    if ibi_i is not None and len(ibi_i) >= 32:
        try:
            f, pxx = sps.welch(ibi_i, fs=fs_interp,
                               nperseg=min(len(ibi_i), 256))
            lf  = _bandpower(f, pxx, 0.04, 0.15)     # ms^2
            hf  = _bandpower(f, pxx, 0.15, 0.40)     # ms^2
            tot = _bandpower(f, pxx, 0.003, 0.40)    # ms^2
            lf_hf = lf / hf if hf > 1e-12 else 0.0
        except Exception as e:
            print(f"    [hrv] welch failed: {e}")

    return np.array([
        mean_hr,
        sdnn, rmssd, pnn50, cvnn,
        mean_ibi * 1000.0,
        float(np.median(ibi) * 1000.0),
        ibi_iqr,
        lf, hf, lf_hf, tot,
        sd1, sd2,
    ], dtype=np.float32)


def standardize(X, mean=None, std=None):
    """Z-score standardisation with optional pre-computed moments."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std