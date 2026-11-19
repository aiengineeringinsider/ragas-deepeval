"""Module 19: production-style monitoring over a file of answers (frozen or logged).

Run:  python scripts/16_monitor.py --records results/responses_qwen2.5_3b.jsonl
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evallab.monitoring import online_checks, sample_for_review  # noqa: E402
from evallab.records import load_records  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--records", required=True)
args = ap.parse_args()
recs = load_records(Path(args.records))
checks = [online_checks(r) for r in recs]
lat = sorted(c["latency_s"] for c in checks)
n = len(recs)
print(f"answers={n} numeric_grounding_pass={sum(c['numeric_grounding'] == 1 for c in checks)}/{n} "
      f"citation_valid_pass={sum(c['citation_validity'] == 1 for c in checks)}/{n} "
      f"refusal_rate={sum(c['refused'] for c in checks) / n:.3f}")
print(f"latency p50={statistics.median(lat):.2f}s p95={lat[max(0, int(0.95 * n) - 1)]:.2f}s")
for r, c in zip(recs, checks):
    if c["numeric_grounding"] < 1 or c["citation_validity"] < 1:
        print(f"  flagged {r.id} {r.category:18} grounding={c['numeric_grounding']:.2f} "
              f"citations={c['citation_validity']:.2f}  {r.answer[:60]!r}")
print("review sample:", sample_for_review(recs, checks))
