"""Robust ECG and BVP peak detectors + HR estimators.

ECG  : Pan-Tompkins with adaptive threshold + searchback
BVP  : median-filter detrend + prominence-based peak detection
HR   : Welch-PSD centroid (robust to broad shoulders) and IBI-median

All filters use `sosfiltfilt` with second-order sections, which is
numerically stable for narrow-band IIR at high sample rates. Do NOT
use filtfilt(butter(...)) — for fs=700 with a 0.7-4 Hz band, the ba
coefficient format is ill-conditioned and filtfilt overflows.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sps
from scipy.ndimage import median_filter


# ----------------------------------------------------------------------
# Numerically stable bandpass
# ----------------------------------------------------------------------
def bandpass(x: np.ndarray, fs: float, lo: float, hi: float,
             order: int = 4) -> np.ndarray:
    """Zero-phase bandpass via second-order sections."""
    nyq = 0.5 * fs
    hi = min(hi, nyq * 0.95)
    lo = max(lo, 1e-4)
    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x)


# ----------------------------------------------------------------------
# ECG: Pan-Tompkins with adaptive threshold + searchback
# ----------------------------------------------------------------------
def ecg_rpeaks(ecg: np.ndarray, fs: float = 700.0,
               min_bpm: float = 40.0, max_bpm: float = 200.0) -> np.ndarray:
    """Detect R-peaks in a chest ECG signal."""
    # 1) Band-pass 5-25 Hz
    x = bandpass(ecg, fs, 5.0, 25.0, order=3)

    # 2) Derivative, square, moving-window integration
    d = np.gradient(x)
    s = d ** 2
    win = max(1, int(0.150 * fs))
    integ = np.convolve(s, np.ones(win) / win, mode="same")

    # 3) Adaptive threshold: 2 s local percentile with global floor
    block = int(2.0 * fs)
    global_thr = 0.35 * np.percentile(integ, 99)
    n_blocks = len(integ) // block + 1
    thr = np.zeros_like(integ)
    for k in range(n_blocks):
        lo, hi = k * block, min((k + 1) * block, len(integ))
        if hi - lo < 4:
            continue
        local = integ[lo:hi]
        thr[lo:hi] = max(0.30 * np.percentile(local, 98), 0.5 * global_thr)

    # 4) Candidate peaks, refractory = 200 ms
    min_dist = int(fs * 60.0 / max_bpm)
    peaks, _ = sps.find_peaks(integ, height=thr, distance=min_dist)

    # 5) Searchback for missed beats
    if len(peaks) >= 3:
        rr = np.diff(peaks)
        med_rr = float(np.median(rr))
        gaps = np.where(rr > 1.6 * med_rr)[0]
        extra = []
        for g in gaps:
            lo, hi = int(peaks[g]), int(peaks[g + 1])
            local_thr = 0.5 * float(thr[lo:hi].mean())
            lp, _ = sps.find_peaks(integ[lo:hi], height=local_thr)
            if len(lp):
                extra.extend(lo + lp)
        if extra:
            peaks = np.unique(np.concatenate([peaks, np.array(extra, dtype=int)]))
            keep = [int(peaks[0])]
            for p in peaks[1:]:
                if p - keep[-1] >= min_dist:
                    keep.append(int(p))
            peaks = np.array(keep, dtype=int)

    # 6) Physiologic range
    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs
        lo_ibi, hi_ibi = 60.0 / max_bpm, 60.0 / min_bpm
        mask = np.ones(len(peaks), dtype=bool)
        mask[1:] = (rr >= lo_ibi) & (rr <= hi_ibi)
        peaks = peaks[mask]

    return peaks


def ibi_from_ecg(ecg: np.ndarray, fs: float = 700.0) -> np.ndarray:
    """Inter-beat intervals (seconds) from a chest ECG window."""
    p = ecg_rpeaks(ecg, fs)
    if len(p) < 3:
        return np.array([])
    ibi = np.diff(p) / fs
    return ibi[(ibi > 60.0 / 200.0) & (ibi < 60.0 / 40.0)]


# ----------------------------------------------------------------------
# BVP: median-filter detrend + prominence-based peak detection
# ----------------------------------------------------------------------
def bvp_peaks(bvp: np.ndarray, fs: float = 64.0,
              min_bpm: float = 40.0, max_bpm: float = 200.0,
              detrend_sec: float = 3.0,
              notch_radius_sec: float = 0.35) -> np.ndarray:
    """Detect systolic peaks in a wrist BVP/PPG signal."""
    if len(bvp) < int(fs * 2):
        return np.array([], dtype=int)

    # 1) Moving-median baseline removal
    win = max(3, int(detrend_sec * fs))
    if win % 2 == 0:
        win += 1
    baseline = median_filter(bvp, size=win, mode="nearest")
    x = bvp - baseline

    # 2) Band-pass 0.7-4 Hz
    x = bandpass(x, fs, 0.7, 4.0, order=3)

    # 3) Prominence-based peaks
    min_dist = int(fs * 60.0 / max_bpm)
    prom = 0.3 * float(np.std(x))
    peaks, _ = sps.find_peaks(x, distance=min_dist, prominence=prom)
    if len(peaks) < 2:
        return peaks

    # 4) Dicrotic-notch rejection
    radius = int(notch_radius_sec * fs)
    keep = []
    for p in peaks:
        lo = max(0, p - radius)
        hi = min(len(x), p + radius + 1)
        in_win = [q for q in peaks if lo <= q < hi]
        tallest = max(in_win, key=lambda q: x[q])
        if p == tallest:
            keep.append(int(p))
    return np.array(sorted(set(keep)), dtype=int)


def ibi_from_bvp(bvp: np.ndarray, fs: float = 64.0) -> np.ndarray:
    """Inter-beat intervals (seconds) from a wrist BVP window."""
    p = bvp_peaks(bvp, fs)
    if len(p) < 3:
        return np.array([])
    ibi = np.diff(p) / fs
    return ibi[(ibi > 60.0 / 200.0) & (ibi < 60.0 / 40.0)]


# ----------------------------------------------------------------------
# HR from Welch PSD — energy-weighted centroid over top-quantile bins
# ----------------------------------------------------------------------
def hr_from_psd(sig: np.ndarray, fs: float,
                band=(0.7, 4.0), nperseg_sec: float = 8.0,
                top_frac: float = 0.15) -> float:
    """HR (bpm) from the Welch PSD of a band-passed signal.

    Uses an energy-weighted centroid over the top `top_frac` of in-band
    PSD bins rather than a single argmax bin. For sharp peaks the two
    coincide; for broad/shoulder peaks the centroid is a more stable
    central-frequency estimate.

    Returns NaN if the window is too short.
    """
    if len(sig) < int(fs * 4):
        return float("nan")

    x = bandpass(sig, fs, band[0], band[1], order=3)
    nperseg = min(len(x), int(nperseg_sec * fs))
    f, pxx = sps.welch(x, fs=fs, nperseg=nperseg)

    m = (f >= band[0]) & (f <= band[1])
    if m.sum() < 3:
        return float("nan")

    fb, pb = f[m], pxx[m]

    # Robust centroid over the top-quantile bins
    thr = float(np.quantile(pb, 1.0 - top_frac))
    mask = pb >= thr
    if mask.sum() >= 1 and pb[mask].sum() > 0:
        f_peak = float(np.sum(fb[mask] * pb[mask]) / np.sum(pb[mask]))
    else:
        f_peak = float(fb[int(np.argmax(pb))])

    return float(f_peak * 60.0)