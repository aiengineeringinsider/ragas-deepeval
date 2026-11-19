"""Module 19: red teaming. Deterministic probes first, then an optional DeepTeam scan.

Run:  python scripts/17_redteam.py             # five probes with exact pass/fail rules
      python scripts/17_redteam.py --deepteam  # + LLM-simulated attacks, local judge
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evallab.monitoring import deepteam_scan, redteam_probes, redteam_verdict  # noqa: E402
from evallab.rag import build_default_pipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--deepteam", action="store_true")
args = ap.parse_args()
pipeline = build_default_pipeline()
held = 0
for probe in redteam_probes:
    answer = pipeline.answer(probe["input"]).answer
    v = redteam_verdict(probe, answer)
    held += v["held"]
    print(f"{probe['id']} {probe['attack']:17} {'HELD ' if v['held'] else 'BROKE'} {answer[:70]!r}")
print(f"held {held}/{len(redteam_probes)}")
if args.deepteam:
    from evallab.deepeval_eval import make_judge

    print(deepteam_scan(pipeline, make_judge()))
