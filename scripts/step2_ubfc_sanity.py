"""Step 2 — rPPG sanity check on one UBFC subject.

Goal: prove that POS extraction on video produces an HR that matches
the finger-PPG ground truth before we batch-process 42 subjects.

Run:  python scripts/step2_ubfc_sanity.py
      python scripts/step2_ubfc_sanity.py --subject 3 --max_frames 600
Output:
  runs/diag/ubfc_subjectN_sanity.png
  console report
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.ubfc import (list_subjects, load_ground_truth, video_path,
                           resample_to_uniform)
from src.preprocessing.face_roi import extract_rgb_traces
from src.preprocessing.rppg import (pos, chrom, rppg_window, sqi,
                                    detrend_moving_avg, quality_weights,
                                    fuse_rois)
from src.preprocessing.peaks import bandpass, hr_from_psd

OUTDIR = "runs/diag"
os.makedirs(OUTDIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/UBFC")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0,
                    help="0 = process whole video")
    ap.add_argument("--method", default="POS", choices=["POS", "CHROM"])
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()

    subj_dir = os.path.join(args.root, f"subject{args.subject}")
    if not os.path.isdir(subj_dir):
        print(f"[FAIL] {subj_dir} not found")
        return

    vp = video_path(subj_dir)
    gt = load_ground_truth(subj_dir)
    if vp is None or gt is None:
        print(f"[FAIL] video or ground truth missing for "
              f"subject{args.subject}")
        return

    print("=" * 72)
    print(f"STEP 2  UBFC sanity check — subject{args.subject}")
    print("=" * 72)
    print(f"  video         : {os.path.basename(vp)} "
          f"({os.path.getsize(vp) / 1e9:.2f} GB)")
    print(f"  duration      : {gt['duration_sec']:.1f} s  "
          f"(fs_eff = {gt['fs_effective']:.2f} Hz)")
    print(f"  GT HR range   : {gt['hr'].min():.1f} - {gt['hr'].max():.1f} bpm")
    print(f"  method        : {args.method}")
    print()

    # ----------------------------------------------------------------
    # 1. Extract RGB traces
    # ----------------------------------------------------------------
    print("Extracting RGB traces from video (this may take a minute)...")
    mf = args.max_frames if args.max_frames > 0 else None
    traces, fps, valid = extract_rgb_traces(vp, mode="geometric",
                                            max_frames=mf)
    n = len(traces["forehead"])
    print(f"  frames processed      : {n}  "
          f"(fps = {fps:.2f}, duration = {n / fps:.1f} s)")
    print(f"  faces detected        : {int(valid.sum())} / {n} "
          f"({100 * valid.mean():.1f} %)")

    if valid.mean() < 0.5:
        print(f"  [WARN] face detected in <50 % of frames — results unreliable")

    # ----------------------------------------------------------------
    # 2. Trim boundary artifacts (1 s on each side)
    # ----------------------------------------------------------------
    edge = int(round(fps * 1.0))
    if n > 2 * edge + int(fps * 4):
        traces = {k: v[edge:-edge] for k, v in traces.items()}
        valid = valid[edge:-edge]
        n = len(traces["forehead"])
        print(f"  trimmed {edge} frames per side → {n} frames used")
    else:
        print(f"  [WARN] window too short to trim ({n} frames); using as-is")

    # ----------------------------------------------------------------
    # 3. Per-ROI SQI weights, fuse
    # ----------------------------------------------------------------
    traces_w = {k: v for k, v in traces.items()}
    weights = quality_weights(traces_w, fps)
    print()
    print("  ROI SQI weights:")
    for k, w in weights.items():
        print(f"    {k:12s}: {w:.3f}")

    fused = fuse_rois(traces, weights)

    # ----------------------------------------------------------------
    # 4. rPPG pipeline on fused trace
    # ----------------------------------------------------------------
    print()
    print(f"  Running {args.method} on fused trace...")
    res = rppg_window(fused, fps, method=args.method)
    hr_rppg_psd = res["hr_psd"]
    ibi = res["ibi"]
    hr_rppg_ibi = float(60.0 / np.median(ibi)) if len(ibi) >= 2 else float("nan")
    sqi_fused = res["sqi"]

    # Per-ROI HR for comparison
    print()
    print("  Per-ROI POS HR (PSD peak):")
    roi_hrs = {}
    for k, tr in traces.items():
        try:
            r = rppg_window(tr, fps, method=args.method)
            roi_hrs[k] = r["hr_psd"]
            print(f"    {k:12s}: HR = {r['hr_psd']:6.2f} bpm  "
                  f"SQI = {r['sqi']:.3f}")
        except Exception as e:
            roi_hrs[k] = float("nan")
            print(f"    {k:12s}: [FAIL] {e}")

    # ----------------------------------------------------------------
    # 5. Finger PPG HR from ground truth
    # ----------------------------------------------------------------
    ppg_u, t_u = resample_to_uniform(gt["ppg"], gt["timestamps"], fs_target=30.0)
    hr_ppg_psd = hr_from_psd(ppg_u, 30.0)
    hr_ppg_gt_mean = float(np.median(gt["hr"]))

    # ----------------------------------------------------------------
    # 6. Report
    # ----------------------------------------------------------------
    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  rPPG HR (PSD, {args.method}, fused)    : {hr_rppg_psd:6.2f} bpm")
    print(f"  rPPG HR (IBI-median, fused)      : {hr_rppg_ibi:6.2f} bpm")
    print(f"  Finger PPG HR (PSD, resampled)   : {hr_ppg_psd:6.2f} bpm")
    print(f"  Ground-truth HR (piecewise)      : {hr_ppg_gt_mean:6.2f} bpm")
    print()
    print(f"  |rPPG(PSD) - FingerPPG(PSD)|     : "
          f"{abs(hr_rppg_psd - hr_ppg_psd):6.2f} bpm")
    print(f"  |rPPG(PSD) - GT-median|          : "
          f"{abs(hr_rppg_psd - hr_ppg_gt_mean):6.2f} bpm")
    print(f"  Fused SQI                        : {sqi_fused:.3f}")
    print()

    err = abs(hr_rppg_psd - hr_ppg_psd)
    if err < 3.0:
        print("  [ OK ] error < 3 bpm — rPPG pipeline validated")
    elif err < 5.0:
        print(f"  [ OK ] error = {err:.2f} bpm — acceptable for a first pass")
    elif err < 10.0:
        print(f"  [WARN] error = {err:.2f} bpm — borderline; inspect PNG")
    else:
        print(f"  [FAIL] error = {err:.2f} bpm — something is wrong")

    # ----------------------------------------------------------------
    # 7. Plots
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(4, 1, figsize=(14, 11))

    t_v = np.arange(len(res["waveform"])) / fps
    n_show = min(len(res["waveform"]), int(15 * fps))

    ax[0].plot(t_v[:n_show], res["waveform"][:n_show],
               color="steelblue", lw=0.8)
    ax[0].set_title(f"rPPG waveform ({args.method}, fused) — "
                    f"HR(PSD)={hr_rppg_psd:.1f} bpm, SQI={sqi_fused:.2f}")
    ax[0].set_xlabel("time (s)")

    n_ppg = min(len(ppg_u), int(15 * 30))
    ax[1].plot(np.arange(n_ppg) / 30.0, ppg_u[:n_ppg],
               color="darkorange", lw=0.8)
    ax[1].set_title(f"Finger PPG (from ground_truth.txt) — "
                    f"HR(PSD)={hr_ppg_psd:.1f} bpm")
    ax[1].set_xlabel("time (s)")

    xr = res["waveform"]
    fr, pr = sps.welch(xr, fs=fps, nperseg=min(len(xr), int(fps * 8)))
    mr = (fr >= 0.7) & (fr <= 4.0)
    ax[2].plot(fr[mr] * 60, pr[mr], color="steelblue",
               label=f"rPPG peak = {hr_rppg_psd:.1f} bpm")
    xp = bandpass(ppg_u, 30.0, 0.7, 4.0, order=3)
    fp, pp = sps.welch(xp, fs=30.0, nperseg=min(len(xp), 30 * 8))
    mp = (fp >= 0.7) & (fp <= 4.0)
    ax[2].plot(fp[mp] * 60, pp[mp] / pp[mp].max() * pr[mr].max(),
               color="darkorange", alpha=0.7,
               label=f"finger PPG peak = {hr_ppg_psd:.1f} bpm")
    ax[2].set_title("Welch PSD (0.7-4 Hz) — rPPG vs finger PPG")
    ax[2].set_xlabel("bpm")
    ax[2].legend(fontsize=8)

    for k, tr in traces.items():
        lum = tr.mean(axis=1)
        lum = detrend_moving_avg(lum, int(fps))
        ax[3].plot(np.arange(len(lum)) / fps, lum,
                   lw=0.6, alpha=0.7, label=k)
    ax[3].set_title("Per-ROI luminance traces (for reference)")
    ax[3].set_xlabel("time (s)")
    ax[3].legend(fontsize=8, ncol=3)

    plt.tight_layout()
    out = os.path.join(OUTDIR,
                       f"ubfc_subject{args.subject}_sanity{args.out_suffix}.png")
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"  [saved] {out}")


if __name__ == "__main__":
    main()