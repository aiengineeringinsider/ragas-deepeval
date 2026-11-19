"""Module 12: exercise the regression gate on SYNTHETIC scores (no model).

We fabricate per-item scores on purpose here, to test the gate's logic:
  run A: the baseline
  run B: the same system re-judged (small symmetric judge noise only)
  run C: a real regression (faithfulness drops on numeric items)
The gate must pass B and fail C. These numbers are test fixtures, not results.
"""
import random
import sys

sys.path.insert(0, "src")
from evallab.regression import compare  # noqa: E402

rng = random.Random(0)
ids = [f"g{i:03d}" for i in range(1, 31)]
numeric = set(ids[8:15])  # pretend g009..g015 are the numeric items


def run(noise: float, numeric_drop: float = 0.0) -> dict:
    out = {"faithfulness": {}, "context_recall": {}}
    for i in ids:
        base_f = 0.9 if i not in numeric else 0.8
        f = base_f + rng.uniform(-noise, noise) - (numeric_drop if i in numeric else 0.0)
        out["faithfulness"][i] = min(1.0, max(0.0, f))
        out["context_recall"][i] = min(1.0, max(0.0, 0.85 + rng.uniform(-noise, noise)))
    return out


A = run(noise=0.05)
B = run(noise=0.05)
C = run(noise=0.05, numeric_drop=0.45)
floors = {"faithfulness": 0.70, "context_recall": 0.60}
print("Candidate B (re-judged, noise only)")
print(compare(A, B, floors=floors).report())
print("\nCandidate C (numeric items regressed)")
print(compare(A, C, floors=floors).report())
