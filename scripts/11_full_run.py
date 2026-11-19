"""Module 14: one command for a complete, stored evaluation run.

Run:
  python scripts/11_full_run.py --label nightly --limit 0
  python scripts/11_full_run.py --label smoke --limit 6 --ragas faithfulness
"""
import argparse
import json
import sys

sys.path.insert(0, "src")
from evallab.config import settings  # noqa: E402
from evallab.pipeline import RunConfig, run_full_evaluation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="manual")
    ap.add_argument("--generator", default=settings.generator_model)
    ap.add_argument("--judge", default=settings.judge_model)
    ap.add_argument("--ragas", default="faithfulness,answer_relevancy,context_recall")
    ap.add_argument("--deepeval", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run_id, summary = run_full_evaluation(RunConfig(
        label=args.label, generator=args.generator, judge=args.judge,
        ragas_metrics=tuple(m for m in args.ragas.split(",") if m),
        use_deepeval=args.deepeval, limit=args.limit))
    print(f"run_id={run_id}")
    print(json.dumps(summary, indent=2))
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, **summary}, indent=2))


if __name__ == "__main__":
    main()
