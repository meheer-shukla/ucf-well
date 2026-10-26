import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter


def pos_algorithm(rgb_signal, fps=30, smooth=True):
    """
    Plane Orthogonal to Skin (POS) algorithm.

    CHANGED: the Savitzky-Golay smoothing window was previously
    fps * 1.5 (~1.5 seconds at 30fps) -- longer than a full heartbeat
    period, which acts as an unintended second low-pass filter with an
    effective cutoff around ~0.74 Hz. That re-attenuates almost the
    entire passband the Butterworth bandpass just preserved (0.667-2.5 Hz),
    flattening exactly the 60-150 BPM range you care about. Fixed to a
    short window that only removes high-frequency jitter.
    """
    N = rgb_signal.shape[0]
    if N < fps:
        return None

    # 1. Temporal normalization
    window_size = int(fps * 1.6)
    if window_size % 2 == 0:
        window_size += 1

    rgb_normalized = np.zeros_like(rgb_signal, dtype=float)
    for i in range(3):
        kernel = np.ones(window_size) / window_size
        padded = np.pad(rgb_signal[:, i], (window_size // 2, window_size // 2), mode='edge')
        running_mean = np.convolve(padded, kernel, mode='valid')
        rgb_normalized[:, i] = rgb_signal[:, i] / np.clip(running_mean, 1e-6, None)

    # 2. POS projection
    R, G, B = rgb_normalized[:, 0], rgb_normalized[:, 1], rgb_normalized[:, 2]
    X = 3 * R - 2 * G
    Y = 1.5 * R + G - 1.5 * B
    alpha = np.std(X) / (np.std(Y) + 1e-6)
    rppg_raw = X + alpha * Y

    # 3. Bandpass filter (40-150 BPM)
    # NOTE: methodology write-up states 0.7-2.5 Hz (42 BPM) -- keep code
    # and paper text consistent, whichever you settle on.
    lowcut, highcut = 40.0 / 60.0, 150.0 / 60.0
    nyq = 0.5 * fps
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    rppg_filtered = filtfilt(b, a, rppg_raw)

    if not smooth:
        return rppg_filtered

    # 4. Light smoothing only -- short window so we don't re-attenuate
    # the heartbeat frequencies the bandpass just preserved.
    # Old: win_len = int(fps * 1.5)  -> ~1.5s, too long, over-smooths
    win_len = max(5, int(fps * 0.15))   # ~0.15s at 30fps -> ~4-5 samples
    if win_len % 2 == 0:
        win_len += 1

    if 3 <= win_len <= len(rppg_filtered):
        return savgol_filter(rppg_filtered, window_length=win_len, polyorder=3)
    return rppg_filtered


def calculate_bpm_from_signal(signal, fps):
    if signal is None or len(signal) < fps:
        return 0.0
    min_distance = int(fps * 0.4)
    peaks, _ = find_peaks(signal, distance=min_distance, prominence=np.std(signal) * 0.3)
    if len(peaks) < 2:
        return 0.0
    return 60.0 / np.mean(np.diff(peaks) / fps)