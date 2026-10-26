"""rPPG signal extraction: POS, CHROM, SQI, IBI, ROI fusion.

Reuses bandpass / hr_from_psd from .peaks — those are the numerically
stable versions we validated on WESAD.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sps
from .peaks import bandpass, hr_from_psd


# ----------------------------------------------------------------------
# Chrominance methods
# ----------------------------------------------------------------------
def pos(rgb: np.ndarray) -> np.ndarray:
    """POS — Wang et al., IEEE TBME 2017.  rgb: (N,3) as R,G,B."""
    eps = 1e-9
    mu = rgb.mean(axis=0, keepdims=True) + eps
    Cn = rgb / mu - 1.0
    S1 = Cn[:, 1] - Cn[:, 2]                      # G - B
    S2 = Cn[:, 1] + Cn[:, 2] - 2.0 * Cn[:, 0]     # G + B - 2R
    alpha = (S1.std() + eps) / (S2.std() + eps)
    h = S1 + alpha * S2
    return h - h.mean()


def chrom(rgb: np.ndarray) -> np.ndarray:
    """CHROM — de Haan & Jeanne, IEEE TBME 2013."""
    eps = 1e-9
    mu = rgb.mean(axis=0, keepdims=True) + eps
    Cn = rgb / mu
    X = 3.0 * Cn[:, 0] - 2.0 * Cn[:, 1]
    Y = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]
    alpha = (X.std() + eps) / (Y.std() + eps)
    S = X - alpha * Y
    return S - S.mean()


def detrend_moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    w = max(3, w | 1)
    k = np.ones(w) / w
    return x - np.convolve(x, k, mode="same")


# ----------------------------------------------------------------------
# Signal Quality Index
# ----------------------------------------------------------------------
def sqi(sig: np.ndarray, fs: float, band=(0.7, 4.0)) -> float:
    """Composite SQI in [0,1] = 0.5 * spectral concentration +
    0.5 * autocorrelation-peak prominence.  Higher = cleaner."""
    if len(sig) < int(fs * 4):
        return 0.0

    f, pxx = sps.welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 8)))
    m = (f >= band[0]) & (f <= band[1])
    if m.sum() == 0 or pxx.sum() <= 0:
        return 0.0
    pb = pxx[m]
    top3 = float(np.sort(pb)[-3:].sum() / (pb.sum() + 1e-12))

    x = sig - sig.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = max(1, int(fs / band[1]))
    hi = min(len(ac) - 1, int(fs / band[0]))
    ac_peak = float(ac[lo:hi].max()) if hi > lo else 0.0

    return float(np.clip(0.5 * top3 + 0.5 * ac_peak, 0.0, 1.0))


# ----------------------------------------------------------------------
# IBI from rPPG waveform
# ----------------------------------------------------------------------
def ibi_from_signal(sig: np.ndarray, fs: float,
                    band=(0.7, 4.0),
                    min_bpm: float = 40.0, max_bpm: float = 200.0) -> np.ndarray:
    """Peak detection on a band-passed rPPG waveform."""
    x = bandpass(sig, fs, band[0], band[1], order=3)
    min_dist = int(fs * 60.0 / max_bpm)
    prom = 0.3 * float(np.std(x))
    peaks, _ = sps.find_peaks(x, distance=min_dist, prominence=prom)
    if len(peaks) < 3:
        return np.array([])
    ibi = np.diff(peaks) / fs
    lo, hi = 60.0 / max_bpm, 60.0 / min_bpm
    return ibi[(ibi >= lo) & (ibi <= hi)]


# ----------------------------------------------------------------------
# Full pipeline on one window
# ----------------------------------------------------------------------
def rppg_window(rgb_window: np.ndarray, fs: float,
                method: str = "POS") -> dict:
    """rgb_window: (N,3).  Returns dict with raw, waveform, hr, ibi, sqi."""
    if method.upper() == "POS":
        raw = pos(rgb_window)
    elif method.upper() == "CHROM":
        raw = chrom(rgb_window)
    else:
        raise ValueError(f"unknown method {method}")

    raw = detrend_moving_avg(raw, int(fs * 1.0))
    bp = bandpass(raw, fs, 0.7, 4.0, order=3)

    return {
        "raw": raw,
        "waveform": bp,
        # ---- FIXED: pass the 1-D rPPG signal, not the raw RGB array ----
        "hr_psd": hr_from_psd(raw, fs),
        # ----------------------------------------------------------------
        "ibi": ibi_from_signal(raw, fs),
        "sqi": sqi(raw, fs),
    }


def fuse_rois(traces: dict, weights: dict | None = None) -> np.ndarray:
    """Weighted fusion of ROI RGB traces.
    traces: {roi_name: (N,3) array}.
    weights: {roi_name: float}, or None for uniform."""
    keys = list(traces.keys())
    if weights is None:
        weights = {k: 1.0 / len(keys) for k in keys}
    out = None
    for k in keys:
        w = weights.get(k, 0.0)
        out = w * traces[k] if out is None else out + w * traces[k]
    return out


def quality_weights(traces: dict, fs: float) -> dict:
    """SQI-weighted ROI fusion weights (normalised)."""
    w = {}
    for k, tr in traces.items():
        if len(tr) < int(fs * 4):
            w[k] = 1e-3
            continue
        r = pos(tr)
        r = detrend_moving_avg(r, int(fs))
        w[k] = max(1e-3, sqi(r, fs))
    s = sum(w.values()) or 1.0
    return {k: v / s for k, v in w.items()}