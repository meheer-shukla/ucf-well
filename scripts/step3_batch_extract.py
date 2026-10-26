"""Step 3 — Batch RGB extraction across all UBFC subjects.

For each subject:
  1. Extract per-ROI RGB traces from video
  2. Trim 1 s boundary edges
  3. Compute SQI-weighted fused trace
  4. Run POS on the fused trace
  5. Compare rPPG HR vs finger PPG HR
  6. Save (traces, fps, valid, gt) to cache/ubfc_traces/subjectN.npz

Resume-safe: subjects already cached are skipped unless --force.

Run:  python scripts/step3_batch_extract.py
      python scripts/step3_batch_extract.py --start 5 --end 10
      python scripts/step3_batch_extract.py --force --subject 3
Outputs:
  cache/ubfc_traces/subjectN.npz
  cache/ubfc_hr_summary.csv
  cache/ubfc_hr_summary.json
"""
from __future__ import annotations
import os, sys, argparse, time, json, csv
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.ubfc import (list_subjects, load_ground_truth, video_path,
                           resample_to_uniform)
from src.preprocessing.face_roi import extract_rgb_traces
from src.preprocessing.rppg import rppg_window, quality_weights, fuse_rois
from src.preprocessing.peaks import hr_from_psd

CACHE_DIR = "cache/ubfc_traces"


# ----------------------------------------------------------------------
# Process one subject
# ----------------------------------------------------------------------
def process_one(subj_dir, method="POS"):
    """Return dict of results, or None on failure."""
    sid = os.path.basename(subj_dir)
    vp = video_path(subj_dir)
    gt = load_ground_truth(subj_dir)
    if vp is None or gt is None:
        return None

    # 1. Extract RGB traces
    traces, fps, valid = extract_rgb_traces(vp, mode="geometric")
    n = len(traces["forehead"])

    # 2. Trim 1 s edges
    edge = int(round(fps * 1.0))
    if n > 2 * edge + int(fps * 4):
        traces = {k: v[edge:-edge] for k, v in traces.items()}
        valid = valid[edge:-edge]
        n = len(traces["forehead"])

    # 3. SQI-weighted fused trace
    weights = quality_weights(traces, fps)
    fused = fuse_rois(traces, weights)

    # 4. rPPG HR
    res = rppg_window(fused, fps, method=method)
    hr_rppg = res["hr_psd"]
    ibi = res["ibi"]
    hr_ibi = float(60.0 / np.median(ibi)) if len(ibi) >= 2 else float("nan")

    # 5. Finger PPG HR
    ppg_u, _ = resample_to_uniform(gt["ppg"], gt["timestamps"], fs_target=30.0)
    hr_ppg = hr_from_psd(ppg_u, 30.0)
    hr_gt_med = float(np.median(gt["hr"]))

    return {
        "subject": sid,
        "n_frames": int(n),
        "fps": float(fps),
        "duration_sec": float(gt["duration_sec"]),
        "face_detected_pct": float(100 * valid.mean()),
        "sqi_fused": float(res["sqi"]),
        "roi_weights": {k: float(v) for k, v in weights.items()},
        "hr_rppg_psd": float(hr_rppg),
        "hr_rppg_ibi": float(hr_ibi),
        "hr_ppg_psd": float(hr_ppg),
        "hr_gt_median": float(hr_gt_med),
        "err_vs_ppg": float(abs(hr_rppg - hr_ppg)),
        "err_vs_gt": float(abs(hr_rppg - hr_gt_med)),
        # Payload for later feature extraction (numpy arrays)
        "traces": {k: v.astype(np.float32) for k, v in traces.items()},
        "valid": valid,
        "gt_ppg": gt["ppg"].astype(np.float32),
        "gt_hr": gt["hr"].astype(np.float32),
        "gt_timestamps": gt["timestamps"].astype(np.float64),
    }


# ----------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------
def save_subject(result, cache_dir):
    sid = result["subject"]
    path = os.path.join(cache_dir, f"{sid}.npz")
    payload = {
        "fps": np.float32(result["fps"]),
        "valid": result["valid"],
        "gt_ppg": result["gt_ppg"],
        "gt_hr": result["gt_hr"],
        "gt_timestamps": result["gt_timestamps"],
        "roi_weights": np.array(
            [result["roi_weights"][k]
             for k in ("forehead", "left_cheek", "right_cheek")],
            dtype=np.float32),
    }
    for k, v in result["traces"].items():
        payload[f"trace_{k}"] = v
    np.savez_compressed(path, **payload)


def load_cached(sid, cache_dir):
    return os.path.exists(os.path.join(cache_dir, f"{sid}.npz"))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/UBFC")
    ap.add_argument("--method", default="POS", choices=["POS", "CHROM"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = process all")
    ap.add_argument("--force", action="store_true",
                    help="reprocess even if cached")
    ap.add_argument("--subject", type=int, default=0,
                    help="process only this subject")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    all_dirs = list_subjects(args.root)
    if not all_dirs:
        print(f"[FAIL] no subjects under {args.root}")
        return

    if args.subject > 0:
        all_dirs = [d for d in all_dirs
                    if os.path.basename(d) == f"subject{args.subject}"]
    else:
        all_dirs = all_dirs[args.start:args.end or len(all_dirs)]

    print("=" * 90)
    print(f"BATCH EXTRACTION  ({len(all_dirs)} subjects, method={args.method})")
    print("=" * 90)
    print(f"{'#':>3s} {'subject':10s} {'status':22s} "
          f"{'dur(s)':>7s} {'fps':>6s} {'face%':>6s} "
          f"{'SQI':>5s} {'rPPG':>6s} {'PPG':>6s} {'err':>5s}")
    print("-" * 90)

    results = []
    t_start = time.time()

    for i, subj_dir in enumerate(all_dirs, 1):
        sid = os.path.basename(subj_dir)
        cached = load_cached(sid, CACHE_DIR)
        if cached and not args.force:
            print(f"{i:>3d} {sid:10s} {'cached (skip)':22s}")
            continue

        t0 = time.time()
        try:
            res = process_one(subj_dir, method=args.method)
        except Exception as e:
            print(f"{i:>3d} {sid:10s} {'[FAIL] ' + str(e)[:15]:22s}")
            continue

        if res is None:
            print(f"{i:>3d} {sid:10s} {'[SKIP] missing files':22s}")
            continue

        elapsed = time.time() - t0
        save_subject(res, CACHE_DIR)
        results.append(res)

        print(f"{i:>3d} {sid:10s} "
              f"{'ok (' + f'{elapsed:.0f}s' + ')':22s} "
              f"{res['duration_sec']:7.1f} "
              f"{res['fps']:6.2f} "
              f"{res['face_detected_pct']:6.1f} "
              f"{res['sqi_fused']:5.3f} "
              f"{res['hr_rppg_psd']:6.1f} "
              f"{res['hr_ppg_psd']:6.1f} "
              f"{res['err_vs_ppg']:5.2f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if not results:
        print("\n[WARN] no new subjects processed "
              "(all cached? try --force)")
        # Try to salvage from cache if it exists
        return

    errs = np.array([r["err_vs_ppg"] for r in results])
    errs_gt = np.array([r["err_vs_gt"] for r in results])
    sqis = np.array([r["sqi_fused"] for r in results])
    faces = np.array([r["face_detected_pct"] for r in results])

    print("\n" + "=" * 90)
    print(f"SUMMARY  ({len(results)} subjects processed this run)")
    print("=" * 90)
    print(f"  HR error vs finger PPG (PSD)  : "
          f"MAE = {errs.mean():.2f} bpm  "
          f"median = {np.median(errs):.2f}  "
          f"p90 = {np.percentile(errs, 90):.2f}  "
          f"max = {errs.max():.2f}")
    print(f"  HR error vs GT median         : "
          f"MAE = {errs_gt.mean():.2f} bpm")
    print(f"  Fused SQI                     : "
          f"mean = {sqis.mean():.3f}  min = {sqis.min():.3f}")
    print(f"  Face detection rate           : "
          f"mean = {faces.mean():.1f} %  min = {faces.min():.1f} %")
    print(f"  Subjects with err < 3 bpm     : "
          f"{int((errs < 3).sum())} / {len(errs)}")
    print(f"  Subjects with err < 5 bpm     : "
          f"{int((errs < 5).sum())} / {len(errs)}")
    print(f"  Total time                    : "
          f"{(time.time() - t_start) / 60:.1f} min")

    # Worst 5 subjects
    order = np.argsort(-errs)[:5]
    print("\n  Worst 5 subjects (err vs PPG):")
    for idx in order:
        r = results[idx]
        print(f"    {r['subject']:10s}  err={r['err_vs_ppg']:5.2f} bpm  "
              f"SQI={r['sqi_fused']:.3f}  face%={r['face_detected_pct']:.1f}")

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    csv_path = "cache/ubfc_hr_summary.csv"
    # If the CSV already exists and we're not processing all subjects,
    # append rather than overwrite.  Simplest: always rewrite from
    # cache if we can find the previous rows — but to keep this simple
    # we write only the current run's rows and rely on the .npz cache.
    write_header = not os.path.exists(csv_path) or args.force or \
                   len(all_dirs) == len(list_subjects(args.root))
    with open(csv_path, "a" if not write_header else "w", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["subject", "duration_sec", "fps", "face_pct",
                        "sqi_fused", "hr_rppg_psd", "hr_rppg_ibi",
                        "hr_ppg_psd", "hr_gt_median",
                        "err_vs_ppg", "err_vs_gt",
                        "w_forehead", "w_left_cheek", "w_right_cheek"])
        for r in results:
            w.writerow([
                r["subject"], f"{r['duration_sec']:.2f}", f"{r['fps']:.2f}",
                f"{r['face_detected_pct']:.1f}", f"{r['sqi_fused']:.3f}",
                f"{r['hr_rppg_psd']:.2f}", f"{r['hr_rppg_ibi']:.2f}",
                f"{r['hr_ppg_psd']:.2f}", f"{r['hr_gt_median']:.2f}",
                f"{r['err_vs_ppg']:.2f}", f"{r['err_vs_gt']:.2f}",
                f"{r['roi_weights']['forehead']:.3f}",
                f"{r['roi_weights']['left_cheek']:.3f}",
                f"{r['roi_weights']['right_cheek']:.3f}",
            ])
    print(f"\n[ OK ] wrote {csv_path}")

    # ------------------------------------------------------------------
    # JSON — only JSON-serializable summary fields
    # ------------------------------------------------------------------
    JSON_KEYS = (
        "subject", "n_frames", "fps", "duration_sec", "face_detected_pct",
        "sqi_fused", "hr_rppg_psd", "hr_rppg_ibi", "hr_ppg_psd",
        "hr_gt_median", "err_vs_ppg", "err_vs_gt", "roi_weights",
    )
    summary_clean = [
        {k: r[k] for k in JSON_KEYS if k in r}
        for r in results
    ]
    with open("cache/ubfc_hr_summary.json", "w") as f:
        json.dump(summary_clean, f, indent=2)
    print(f"[ OK ] wrote cache/ubfc_hr_summary.json")


if __name__ == "__main__":
    main()