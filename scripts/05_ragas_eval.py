"""Module 5: score frozen responses with RAGAS metrics, judged by a local model.

Run:
  python scripts/05_ragas_eval.py --responses results/responses_qwen2.5_3b.jsonl \
         --judge qwen2.5:3b --limit 10
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from evallab.custom_metrics import deterministic_scores  # noqa: E402
from evallab.ragas_eval import (default_metric_specs, evaluate_records,  # noqa: E402
                                make_ragas_embeddings, make_ragas_llm, summarize)
from evallab.records import load_records, save_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--judge", default="qwen2.5:3b")
    ap.add_argument("--metrics", default="all", help="comma list, e.g. faithfulness,context_recall")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    records = load_records(Path(args.responses))
    if args.limit:
        records = records[: args.limit]
    specs = default_metric_specs(make_ragas_llm(args.judge), make_ragas_embeddings())
    if args.metrics != "all":
        wanted = set(args.metrics.split(","))
        specs = [s for s in specs if s.name in wanted]

    t0 = time.perf_counter()
    asyncio.run(evaluate_records(records, specs, concurrency=args.concurrency))
    for r in records:  # cheap deterministic metrics ride along
        r.scores.update(deterministic_scores(r))
    elapsed = time.perf_counter() - t0

    out = Path(args.responses).with_name(Path(args.responses).stem + f"__ragas_{args.judge.replace(':', '_')}.jsonl")
    save_records(records, out)
    summary = summarize(records)
    print(f"\nRAGAS summary (judge={args.judge}, n={len(records)}, {elapsed:.0f}s)")
    for name, s in summary.items():
        print(f"  {name:<22} mean={s['mean']:<7} scored={s['n_scored']:<3} errors={s['n_errors']}")
    print(f"\nper-item scores -> {out}")
    Path("results").mkdir(exist_ok=True)
    Path("results/ragas_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
