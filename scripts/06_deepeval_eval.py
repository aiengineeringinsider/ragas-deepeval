"""Module 6: score frozen responses with DeepEval metrics (local Ollama judge).

Run:
  python scripts/06_deepeval_eval.py --responses results/responses_qwen2.5_3b.jsonl \
         --judge qwen2.5:3b --limit 10
For the pytest flavour, see tests/eval/test_rag_quality.py and Module 6.
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from evallab.custom_metrics import make_numeric_grounding_metric  # noqa: E402
from evallab.deepeval_eval import default_metrics, make_judge, score_record  # noqa: E402
from evallab.ragas_eval import summarize  # noqa: E402
from evallab.records import load_records, save_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--judge", default="qwen2.5:3b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-geval", action="store_true")
    args = ap.parse_args()

    records = load_records(Path(args.responses))[: args.limit or None]
    judge = make_judge(args.judge)
    metrics = default_metrics(judge, include_geval=not args.no_geval)
    metrics.append(make_numeric_grounding_metric())

    t0 = time.perf_counter()
    for r in records:
        score_record(r, metrics)
        shown = {k: (round(v, 2) if not math.isnan(v) else "nan") for k, v in r.scores.items()}
        print(f"{r.id}: {shown}")
    print(f"\nDeepEval summary (judge={args.judge}, n={len(records)}, "
          f"{time.perf_counter() - t0:.0f}s)")
    thresholds = {getattr(m, "name", None) or m.__name__: m.threshold for m in metrics}
    for name, s in summarize(records).items():
        thr = thresholds.get(name)
        print(f"  {name:<32} mean={s['mean']:<7} thr={thr} scored={s['n_scored']} errors={s['n_errors']}")
    out = Path(args.responses).with_name(Path(args.responses).stem + f"__deepeval_{args.judge.replace(':', '_')}.jsonl")
    save_records(records, out)
    print(f"\nper-item scores -> {out}")


if __name__ == "__main__":
    main()
