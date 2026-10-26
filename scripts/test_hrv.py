"""Quick sanity check that hrv_features produces non-zero LF/HF."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.features.hrv import hrv_features, FEATURE_NAMES

np.random.seed(0)
ibi = np.random.normal(0.70, 0.05, 80)   # ~86 bpm, realistic variability

f = hrv_features(ibi)
if f is None:
    print("hrv_features returned None")
    sys.exit(1)

print("Index  Feature        Value")
print("-" * 36)
for i, name in enumerate(FEATURE_NAMES):
    print(f"{i:>3d}   {name:14s} {f[i]:>10.3f}")

print()
if f[8] > 0 and f[9] > 0 and f[10] > 0:
    print("[ OK ] LF/HF features are non-zero — fix worked")
else:
    print("[FAIL] LF/HF still zero")