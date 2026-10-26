"""STEP 1 — WESAD integrity check.
Run:  python check_wesad.py
"""
import os, glob, pickle, traceback
import numpy as np

ROOT = "datasets/WESAD"
EXPECTED = [f"S{i}" for i in list(range(2, 12)) + list(range(13, 18))]  # 15 subs

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


def main():
    print("=" * 72)
    print("WESAD integrity check")
    print("=" * 72)

    if not os.path.isdir(ROOT):
        print(f"{FAIL} {ROOT} not found")
        return

    folders = sorted([os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "S*"))
                      if os.path.isdir(p)])
    print(f"\nFolders found : {len(folders)}  {folders}")
    missing = [s for s in EXPECTED if s not in folders]
    extra = [s for s in folders if s not in EXPECTED]
    if missing:
        print(f"{FAIL} missing subjects: {missing}")
    else:
        print(f"{OK} all 15 expected subjects present")
    if extra:
        print(f"{WARN} unexpected folders: {extra}")

    # Aggregate stats
    all_labels = {}
    all_lengths = {}
    wrist_bvp_ok = 0
    fails = []

    print("\n" + "-" * 72)
    print(f"{'Subject':8s} {'len(lab)':>10s} {'dur(min)':>9s}  "
          f"{'ECG shape':>14s} {'BVP shape':>14s}  label counts (1/2/3/4)")
    print("-" * 72)

    for s in folders:
        pkl_path = os.path.join(ROOT, s, f"{s}.pkl")
        if not os.path.exists(pkl_path):
            print(f"{FAIL} {s}: missing {pkl_path}")
            fails.append(s)
            continue
        try:
            with open(pkl_path, "rb") as f:
                d = pickle.load(f, encoding="latin1")
        except Exception as e:
            print(f"{FAIL} {s}: cannot unpickle ({e})")
            fails.append(s)
            continue

        try:
            ecg = np.asarray(d["signal"]["chest"]["ECG"]).squeeze()
            lab = np.asarray(d["label"]).squeeze().astype(int)
        except KeyError as e:
            print(f"{FAIL} {s}: missing key {e}")
            fails.append(s)
            continue

        wrist_keys = list(d["signal"].get("wrist", {}).keys())
        has_bvp = "BVP" in wrist_keys
        bvp_shape_str = "-"
        if has_bvp:
            bvp = np.asarray(d["signal"]["wrist"]["BVP"]).squeeze()
            bvp_shape_str = str(bvp.shape)
            wrist_bvp_ok += 1

        u, c = np.unique(lab, return_counts=True)
        counts = dict(zip(u.tolist(), c.tolist()))
        n1, n2, n3, n4 = (counts.get(k, 0) for k in (1, 2, 3, 4))

        dur_min = len(lab) / 700.0 / 60.0
        print(f"{s:8s} {len(lab):>10d} {dur_min:>9.2f}  "
              f"{str(ecg.shape):>14s} {bvp_shape_str:>14s}  "
              f"{n1}/{n2}/{n3}/{n4}")

        all_labels[s] = counts
        all_lengths[s] = len(lab)

    print("-" * 72)

    # Verdicts
    print("\nVERDICTS")
    if not fails:
        print(f"{OK} all {len(folders)} subjects unpickle cleanly")
    else:
        print(f"{FAIL} {len(fails)} subjects failed: {fails}")

    if wrist_bvp_ok == len(folders):
        print(f"{OK} wrist BVP present in all subjects "
              f"→ TS-CMHT (two-stage modality bridge) is viable")
    elif wrist_bvp_ok == 0:
        print(f"{FAIL} wrist BVP missing everywhere → "
              f"fall back to single-stage ECG→rPPG CMHT")
    else:
        print(f"{WARN} wrist BVP present in only {wrist_bvp_ok}/{len(folders)} subjects")

    # Label structure check across dataset
    print("\nLabel distribution (pooled, 700 Hz samples):")
    tot = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for c in all_labels.values():
        for k in tot:
            tot[k] += c.get(k, 0)
    grand = sum(tot.values())
    names = {0: "transient", 1: "baseline ", 2: "stress   ", 3: "amusement", 4: "meditation"}
    for k in (0, 1, 2, 3, 4):
        print(f"  {k} {names[k]:11s}: {tot[k]:>10d}  ({100*tot[k]/grand:5.1f}%)")

    class_ratio = max(tot[1], tot[2], tot[3]) / max(1, min(tot[1], tot[2], tot[3]))
    print(f"  → class imbalance ratio (1,2,3): {class_ratio:.2f}x")
    if class_ratio > 3:
        print(f"{WARN} source-domain class imbalance >3x — "
              f"use class-weighted CE / FocalLoss in the stress head")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()