"""Module 15: generate synthetic candidates with the RAGAS TestsetGenerator and triage them.

Run:  python scripts/12_testset.py --size 12
Writes data/golden/testset_candidates.jsonl (status needs_review or rejected_duplicate).
Nothing here touches golden_v1.jsonl: a person promotes candidates by hand.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evallab.config import settings  # noqa: E402
from evallab.golden import load_golden  # noqa: E402
from evallab.testset import generate_testset, review_queue  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=12)
ap.add_argument("--model", default=None)
args = ap.parse_args()

df = generate_testset(args.size, args.model)
queue = review_queue(df.to_dict("records"), load_golden(settings.golden_path))
out = settings.golden_path.parent / "testset_candidates.jsonl"
out.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in queue) + "\n", encoding="utf-8")
by = {}
for q in queue:
    by.setdefault(q["synthesizer"], 0)
    by[q["synthesizer"]] += 1
print(f"generated {len(queue)} candidates -> {out}")
print("by synthesizer:", by)
print("duplicates of existing goldens:", sum(1 for q in queue if q["duplicate_of"]))
print("with a trap (number, negation, limit):", sum(1 for q in queue if q["has_trap"]))
