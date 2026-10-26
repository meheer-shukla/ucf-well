"""WESAD loader.

Layout:   datasets/WESAD/SX/SX.pkl
Keys:     d['signal']['chest']['ECG']   (700 Hz)
          d['signal']['wrist']['BVP']   ( 64 Hz)
          d['label']                    (700 Hz, ids 0..4)

Label map (kept classes):
    1 -> 0  baseline
    2 -> 1  stress
    3 -> 2  amusement
    (0 transient, 4 meditation are discarded by default)
"""
from __future__ import annotations
import os, glob, pickle
import numpy as np

FS_CHEST  = 700
FS_WRIST  = 64
CLASS_NAMES = ["baseline", "stress", "amusement"]
RAW_TO_CLASS = {1: 0, 2: 1, 3: 2}
LABEL_NAMES_RAW = {0: "transient", 1: "baseline", 2: "stress",
                   3: "amusement", 4: "meditation"}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def list_subjects(root: str) -> list[str]:
    out = []
    for p in sorted(glob.glob(os.path.join(root, "S*"))):
        pkl = os.path.join(p, os.path.basename(p) + ".pkl")
        if os.path.exists(pkl):
            out.append(pkl)
    return out


def load_subject(pkl_path: str) -> dict:
    """Return a normalised dict with chest ECG, wrist BVP, and labels."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    sig = d["signal"]
    ecg = np.asarray(sig["chest"]["ECG"]).squeeze().astype(np.float64)
    labels = np.asarray(d["label"]).squeeze().astype(int)

    bvp = None
    if "wrist" in sig and "BVP" in sig["wrist"]:
        bvp = np.asarray(sig["wrist"]["BVP"]).squeeze().astype(np.float64)

    return {
        "subject": d.get("subject", os.path.basename(os.path.dirname(pkl_path))),
        "ecg": ecg,          # (N_chest,)  @ 700 Hz
        "bvp": bvp,          # (N_wrist,)  @ 64 Hz  or None
        "labels": labels,    # (N_chest,)  @ 700 Hz
        "fs_chest": FS_CHEST,
        "fs_wrist": FS_WRIST,
    }


# ----------------------------------------------------------------------
# Window generation
# ----------------------------------------------------------------------
def _majority_label(labels_seg: np.ndarray):
    vals, counts = np.unique(labels_seg, return_counts=True)
    maj = vals[np.argmax(counts)]
    purity = counts.max() / len(labels_seg)
    return maj, purity


def sliding_labeled_windows(ecg: np.ndarray, labels: np.ndarray,
                            fs: int = FS_CHEST,
                            win_sec: float = 60.0,
                            step_sec: float = 30.0,
                            min_purity: float = 0.9,
                            keep_raw_ids=(1, 2, 3)):
    """Yield (ecg_window, class_label). Only yields windows that are
    >= min_purity homogeneous and whose majority raw label is in keep_raw_ids."""
    w = int(win_sec * fs)
    s = int(step_sec * fs)
    for st in range(0, len(ecg) - w, s):
        seg_lab = labels[st:st + w]
        maj, purity = _majority_label(seg_lab)
        if purity >= min_purity and maj in keep_raw_ids:
            yield ecg[st:st + w], RAW_TO_CLASS[maj]


def align_wrist_labels(labels_700: np.ndarray, n_wrist: int,
                       fs_chest: int = FS_CHEST,
                       fs_wrist: int = FS_WRIST) -> np.ndarray:
    """Downsample the 700 Hz label stream to wrist rate.

    Uses nearest-neighbour by construction: index into labels_700 at
    round(i * fs_chest / fs_wrist).  Safe because labels are piecewise
    constant over many seconds.
    """
    idx = np.round(np.arange(n_wrist) * fs_chest / fs_wrist).astype(np.int64)
    idx = np.clip(idx, 0, len(labels_700) - 1)
    return labels_700[idx]


def sliding_labeled_windows_bvp(bvp: np.ndarray, labels_700: np.ndarray,
                                fs_wrist: int = FS_WRIST,
                                win_sec: float = 60.0,
                                step_sec: float = 30.0,
                                min_purity: float = 0.9,
                                keep_raw_ids=(1, 2, 3)):
    """Same as above but for the wrist BVP at 64 Hz."""
    if bvp is None:
        return
    labels_wrist = align_wrist_labels(labels_700, len(bvp))
    w = int(win_sec * fs_wrist)
    s = int(step_sec * fs_wrist)
    for st in range(0, len(bvp) - w, s):
        seg_lab = labels_wrist[st:st + w]
        maj, purity = _majority_label(seg_lab)
        if purity >= min_purity and maj in keep_raw_ids:
            yield bvp[st:st + w], RAW_TO_CLASS[maj]