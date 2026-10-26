"""UBFC-rPPG loader.

Layout:
    datasets/UBFC/subjectN/vid.avi
    datasets/UBFC/subjectN/ground_truth.txt

ground_truth.txt is 3 rows of equal length:
    row 0 : PPG waveform (finger oximeter)   ~30 Hz, irregular timestamps
    row 1 : HR in bpm (piecewise constant)   updates ~1 Hz
    row 2 : timestamps in seconds
"""
from __future__ import annotations
import os, glob
import numpy as np


def list_subjects(root: str) -> list[str]:
    subs = glob.glob(os.path.join(root, "subject*"))
    subs = [s for s in subs if os.path.isdir(s)]
    return sorted(subs, key=lambda p: int(
        os.path.basename(p).replace("subject", "")))


def load_ground_truth(subj_dir: str) -> dict | None:
    gt = os.path.join(subj_dir, "ground_truth.txt")
    if not os.path.exists(gt):
        return None
    arr = np.loadtxt(gt)
    if arr.shape[0] == 3 and arr.shape[1] > 3:
        ppg, hr, ts = arr[0], arr[1], arr[2]
    elif arr.shape[1] == 3 and arr.shape[0] > 3:
        ppg, hr, ts = arr[:, 0], arr[:, 1], arr[:, 2]
    else:
        raise ValueError(f"unexpected ground_truth shape {arr.shape} in {gt}")

    dt = np.diff(ts); dt = dt[dt > 0]
    fs_eff = 1.0 / float(np.median(dt)) if len(dt) else 30.0

    return {
        "ppg": ppg.astype(np.float64),
        "hr": hr.astype(np.float64),
        "timestamps": ts.astype(np.float64),
        "fs_effective": fs_eff,
        "duration_sec": float(ts[-1] - ts[0]),
    }


def resample_to_uniform(ppg: np.ndarray, ts: np.ndarray,
                        fs_target: float = 30.0):
    """Resample finger PPG onto a uniform grid for Welch/peak detection."""
    from scipy import interpolate
    t0, t1 = ts[0], ts[-1]
    t_new = np.arange(t0, t1, 1.0 / fs_target)
    f = interpolate.interp1d(ts, ppg, kind="linear",
                             bounds_error=False, fill_value="extrapolate")
    return f(t_new), t_new


def video_path(subj_dir: str) -> str | None:
    for pat in ("vid.avi", "video.avi", "*.avi", "*.mp4"):
        hits = glob.glob(os.path.join(subj_dir, pat))
        if hits:
            return hits[0]
    return None


def sanity_report(root: str, n_show: int = 5) -> None:
    """Print one line per subject (text files only — no video decode)."""
    subs = list_subjects(root)
    print(f"{len(subs)} subjects under {root}")
    for s in subs[:n_show] + subs[-1:]:
        sid = os.path.basename(s)
        try:
            gt = load_ground_truth(s)
        except Exception as e:
            print(f"  {sid:10s}  [FAIL] {e}")
            continue
        vp = video_path(s)
        sz = os.path.getsize(vp) / 1e9 if vp else 0.0
        print(f"  {sid:10s}  dur={gt['duration_sec']:6.2f}s  "
              f"fs_eff={gt['fs_effective']:5.2f}Hz  "
              f"HR={gt['hr'].min():5.1f}-{gt['hr'].max():5.1f}bpm  "
              f"video={sz:.2f}GB")


if __name__ == "__main__":
    sanity_report("datasets/UBFC")