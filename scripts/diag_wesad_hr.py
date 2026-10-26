"""Diagnose the ECG<->BVP HR disagreement on WESAD.

Self-contained: only depends on numpy, scipy, matplotlib, stdlib.
No src/ imports.

Run:  python scripts/diag_wesad_hr.py
Outputs:
  runs/diag/S10_window.png   (4-panel plot)
  console report
"""
from __future__ import annotations
import os, sys, glob, pickle
import numpy as np
from scipy import signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS_CHEST = 700
FS_WRIST = 64
OUTDIR = "runs/diag"
os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------------------------------------------------
# Inline WESAD loading (no src/ import)
# ----------------------------------------------------------------------
def load_subject(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    ecg = np.asarray(d["signal"]["chest"]["ECG"]).squeeze().astype(np.float64)
    labels = np.asarray(d["label"]).squeeze().astype(int)
    bvp = None
    if "wrist" in d["signal"] and "BVP" in d["signal"]["wrist"]:
        bvp = np.asarray(d["signal"]["wrist"]["BVP"]).squeeze().astype(np.float64)
    return {"ecg": ecg, "bvp": bvp, "labels": labels}


# ----------------------------------------------------------------------
# Inline bandpass + Welch-PSD HR (copies of what src/preprocessing/rppg.py
# will contain — same math, no import needed)
# ----------------------------------------------------------------------
def bandpass(x, fs, lo, hi, order=4):
    nyq = 0.5 * fs
    hi = min(hi, nyq * 0.99)
    lo = max(lo, 1e-4)
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x)


def hr_from_psd(sig, fs, band=(0.7, 4.0)):
    f, pxx = sps.welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 8)))
    m = (f >= band[0]) & (f <= band[1])
    if m.sum() < 3:
        return float("nan")
    fb, pb = f[m], pxx[m]
    k = int(np.argmax(pb))
    if 0 < k < len(pb) - 1:
        a, b, c = pb[k - 1], pb[k], pb[k + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        f_peak = fb[k] + delta * (fb[1] - fb[0])
    else:
        f_peak = fb[k]
    return float(f_peak * 60.0)


# ----------------------------------------------------------------------
# Peak detectors — SAME AS CURRENT PIPELINE (for diagnosis)
# ----------------------------------------------------------------------
def ecg_rpeaks_current(ecg, fs=FS_CHEST):
    b, a = sps.butter(3, [5.0 / (fs / 2), 25.0 / (fs / 2)], btype="band")
    x = sps.filtfilt(b, a, ecg)
    d = np.diff(x) ** 2
    win = int(0.12 * fs)
    integ = np.convolve(d, np.ones(win) / win, mode="same")
    thr = 0.3 * np.percentile(integ, 99)
    peaks, _ = sps.find_peaks(integ, height=thr, distance=int(0.25 * fs))
    return peaks


def bvp_peaks_current(bvp, fs=FS_WRIST):
    b, a = sps.butter(3, [0.7 / (fs / 2), 4.0 / (fs / 2)], btype="band")
    x = sps.filtfilt(b, a, bvp)
    thr = 0.25 * np.std(x)
    peaks, _ = sps.find_peaks(x, height=thr, distance=int(0.33 * fs))
    return peaks


def hr_from_ibi_median(peaks, fs):
    if len(peaks) < 3:
        return float("nan")
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi > 0.33) & (ibi < 1.43)]
    if len(ibi) < 2:
        return float("nan")
    return float(60.0 / np.median(ibi))


# ----------------------------------------------------------------------
# Q1: single-window diagnostic on S10
# ----------------------------------------------------------------------
def diagnose_one_window():
    pkl = "datasets/WESAD/S10/S10.pkl"
    if not os.path.exists(pkl):
        print(f"[SKIP] {pkl} not found")
        return

    d = load_subject(pkl)
    ecg, bvp, lab = d["ecg"], d["bvp"], d["labels"]

    # Find a 60 s window >=99 % baseline (label 1)
    w_ecg = int(60 * FS_CHEST)
    start = 0
    for st in range(0, len(lab) - w_ecg, int(30 * FS_CHEST)):
        seg = lab[st:st + w_ecg]
        if (seg == 1).mean() > 0.99:
            start = st
            break

    seg_ecg = ecg[start:start + w_ecg]
    t_ecg = np.arange(len(seg_ecg)) / FS_CHEST

    bvp_start = int(round(start / FS_CHEST * FS_WRIST))
    bvp_len = int(round(w_ecg / FS_CHEST * FS_WRIST))
    seg_bvp = bvp[bvp_start:bvp_start + bvp_len]
    t_bvp = np.arange(len(seg_bvp)) / FS_WRIST

    p_e = ecg_rpeaks_current(seg_ecg)
    p_b = bvp_peaks_current(seg_bvp)

    hr_ecg_ibi = hr_from_ibi_median(p_e, FS_CHEST)
    hr_bvp_ibi = hr_from_ibi_median(p_b, FS_WRIST)

    ecg_bp = bandpass(seg_ecg, FS_CHEST, 0.7, 4.0)
    bvp_bp = bandpass(seg_bvp, FS_WRIST, 0.7, 4.0)
    hr_ecg_psd = hr_from_psd(ecg_bp, FS_CHEST)
    hr_bvp_psd = hr_from_psd(bvp_bp, FS_WRIST)

    print("=" * 72)
    print("Q1  SINGLE-WINDOW DIAGNOSTIC  (S10, 60 s baseline)")
    print("=" * 72)
    print(f"  window starts at t = {start / FS_CHEST:.1f} s")
    print(f"  ECG duration = {len(seg_ecg) / FS_CHEST:.2f} s, "
          f"BVP duration = {len(seg_bvp) / FS_WRIST:.2f} s")
    print()
    print(f"  ECG peaks detected    : {len(p_e):4d}   "
          f"(HR via IBI-median = {hr_ecg_ibi:.2f} bpm)")
    print(f"  BVP peaks detected    : {len(p_b):4d}   "
          f"(HR via IBI-median = {hr_bvp_ibi:.2f} bpm)")
    print(f"  expected beats in 60 s: ECG ~{hr_ecg_ibi:.0f}, "
          f"BVP ~{hr_bvp_ibi:.0f}")
    print()
    print("  HR estimates:")
    print(f"    ECG via IBI-median : {hr_ecg_ibi:6.2f} bpm")
    print(f"    ECG via Welch PSD  : {hr_ecg_psd:6.2f} bpm   "
          f"(|diff| = {abs(hr_ecg_ibi - hr_ecg_psd):.2f})")
    print(f"    BVP via IBI-median : {hr_bvp_ibi:6.2f} bpm")
    print(f"    BVP via Welch PSD  : {hr_bvp_psd:6.2f} bpm   "
          f"(|diff| = {abs(hr_bvp_ibi - hr_bvp_psd):.2f})")
    print()

    # Plots — 4 panels
    fig, ax = plt.subplots(4, 1, figsize=(14, 11))
    seg_show = 10  # seconds to display
    n_e = min(len(seg_ecg), int(seg_show * FS_CHEST))
    n_b = min(len(seg_bvp), int(seg_show * FS_WRIST))

    ax[0].plot(t_ecg[:n_e], seg_ecg[:n_e], lw=0.7, color="steelblue")
    pk = p_e[p_e < n_e]
    ax[0].plot(t_ecg[pk], seg_ecg[pk], "rv", ms=8,
               label=f"{len(pk)} peaks in first {seg_show}s")
    ax[0].set_title(f"ECG raw, first {seg_show}s — red triangles = detected R-peaks")
    ax[0].set_xlabel("time (s)")
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].plot(t_bvp[:n_b], seg_bvp[:n_b], lw=0.9, color="darkorange")
    pk = p_b[p_b < n_b]
    ax[1].plot(t_bvp[pk], seg_bvp[pk], "rv", ms=8,
               label=f"{len(pk)} peaks in first {seg_show}s")
    ax[1].set_title(f"BVP raw, first {seg_show}s — red triangles = detected peaks")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(loc="upper right", fontsize=8)

    f_e, pxx_e = sps.welch(ecg_bp, fs=FS_CHEST,
                           nperseg=min(len(ecg_bp), FS_CHEST * 8))
    m = (f_e >= 0.7) & (f_e <= 4.0)
    ax[2].plot(f_e[m] * 60, pxx_e[m], color="steelblue")
    ax[2].axvline(hr_ecg_psd, color="red", ls="--",
                  label=f"peak = {hr_ecg_psd:.1f} bpm")
    ax[2].set_title("ECG Welch PSD (0.7-4 Hz)")
    ax[2].set_xlabel("bpm")
    ax[2].legend(loc="upper right", fontsize=8)

    f_b, pxx_b = sps.welch(bvp_bp, fs=FS_WRIST,
                           nperseg=min(len(bvp_bp), FS_WRIST * 8))
    m = (f_b >= 0.7) & (f_b <= 4.0)
    ax[3].plot(f_b[m] * 60, pxx_b[m], color="darkorange")
    ax[3].axvline(hr_bvp_psd, color="red", ls="--",
                  label=f"peak = {hr_bvp_psd:.1f} bpm")
    ax[3].set_title("BVP Welch PSD (0.7-4 Hz)")
    ax[3].set_xlabel("bpm")
    ax[3].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(OUTDIR, "S10_window.png")
    plt.savefig(out_png, dpi=110)
    plt.close()
    print(f"  [saved] {out_png}")
    print()

    print("  Heuristic checks:")
    if abs(hr_ecg_ibi - hr_ecg_psd) > 4:
        print(f"    [SUSPECT] ECG IBI vs PSD disagree by "
              f"{abs(hr_ecg_ibi - hr_ecg_psd):.1f} bpm "
              f"→ ECG peak detection is inconsistent")
    else:
        print(f"    [  OK  ] ECG IBI vs PSD agree within "
              f"{abs(hr_ecg_ibi - hr_ecg_psd):.2f} bpm")
    if abs(hr_bvp_ibi - hr_bvp_psd) > 4:
        print(f"    [SUSPECT] BVP IBI vs PSD disagree by "
              f"{abs(hr_bvp_ibi - hr_bvp_psd):.1f} bpm "
              f"→ BVP peak detection is inconsistent (likely dicrotic notches)")
    else:
        print(f"    [  OK  ] BVP IBI vs PSD agree within "
              f"{abs(hr_bvp_ibi - hr_bvp_psd):.2f} bpm")
    print()


# ----------------------------------------------------------------------
# Q2: all subjects, time-paired
# ----------------------------------------------------------------------
def diagnose_all_time_paired():
    pkls = sorted(glob.glob("datasets/WESAD/S*/S*.pkl"))
    print("=" * 72)
    print("Q2  TIME-PAIRED ECG<->BVP HR AGREEMENT  (all subjects)")
    print("=" * 72)
    print(f"{'subj':6s} {'n':>4s}  {'MAE':>6s}  {'bias':>7s}  {'r':>7s}")
    print("-" * 72)

    win_sec, step_sec = 60, 30
    all_e, all_b = [], []

    for pkl in pkls:
        sid = os.path.basename(os.path.dirname(pkl))
        d = load_subject(pkl)
        ecg, bvp, lab = d["ecg"], d["bvp"], d["labels"]

        e_pts, b_pts = [], []

        for st in range(0, len(ecg) - win_sec * FS_CHEST,
                        step_sec * FS_CHEST):
            seg = lab[st:st + win_sec * FS_CHEST]
            vals, counts = np.unique(seg, return_counts=True)
            if counts.max() / len(seg) < 0.9:
                continue
            t_sec = st / FS_CHEST
            p = ecg_rpeaks_current(ecg[st:st + win_sec * FS_CHEST])
            hr = hr_from_ibi_median(p, FS_CHEST)
            if not np.isnan(hr):
                e_pts.append((round(t_sec, 2), hr))

        for st in range(0, len(bvp) - win_sec * FS_WRIST,
                        step_sec * FS_WRIST):
            t_sec = st / FS_WRIST
            p = bvp_peaks_current(bvp[st:st + win_sec * FS_WRIST])
            hr = hr_from_ibi_median(p, FS_WRIST)
            if not np.isnan(hr):
                b_pts.append((round(t_sec, 2), hr))

        e_dict = dict(e_pts)
        b_dict = dict(b_pts)
        common = sorted(set(e_dict) & set(b_dict))
        if len(common) < 5:
            print(f"{sid:6s} {len(common):>4d}  (insufficient overlap)")
            continue

        e_hr = np.array([e_dict[t] for t in common])
        b_hr = np.array([b_dict[t] for t in common])
        d_hr = e_hr - b_hr
        mae_v = float(np.mean(np.abs(d_hr)))
        bias_v = float(np.mean(d_hr))
        if e_hr.std() > 1e-6 and b_hr.std() > 1e-6:
            r_v = float(np.corrcoef(e_hr, b_hr)[0, 1])
        else:
            r_v = float("nan")
        print(f"{sid:6s} {len(common):>4d}  {mae_v:6.2f}  "
              f"{bias_v:+7.2f}  {r_v:+7.3f}")
        all_e.append(e_hr); all_b.append(b_hr)

    if all_e:
        e = np.concatenate(all_e); b = np.concatenate(all_b)
        diff = e - b
        r_pool = (np.corrcoef(e, b)[0, 1]
                  if e.std() > 1e-6 and b.std() > 1e-6 else float("nan"))
        print("-" * 72)
        print(f"POOLED   n={len(e)}   MAE={np.mean(np.abs(diff)):.2f}   "
              f"bias={np.mean(diff):+.2f}   r={r_pool:+.3f}")
        print()
        print("Interpretation:")
        if np.mean(np.abs(diff)) < 4:
            print("  [ OK ] MAE < 4 bpm with time-pairing → the original "
                  "bug was window misalignment.")
        else:
            print(f"  [SUSPECT] MAE {np.mean(np.abs(diff)):.1f} bpm still "
                  f"high with time-pairing → peak detectors themselves are weak.")
        if abs(np.mean(diff)) > 4:
            print(f"  [SUSPECT] systematic bias {np.mean(diff):+.1f} bpm — "
                  f"one detector under/over-counts consistently.")


if __name__ == "__main__":
    diagnose_one_window()
    diagnose_all_time_paired()