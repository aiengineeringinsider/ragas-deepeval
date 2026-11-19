"""Module 10: statistics for human calibration.

Part 1 (no model): how many golden items do you need, and how wide is the
uncertainty on a judge's accuracy measured on 40 labels?
Part 2 (--llm): score the 40 calibration cases with RAGAS faithfulness
(continuous, 0..1), then sweep the pass threshold against the human labels.

Run:  python examples/calibration_analysis.py
      python examples/calibration_analysis.py --llm
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from evallab.config import settings  # noqa: E402
from evallab.custom_metrics import numeric_grounding  # noqa: E402
from evallab.judges import load_calibration  # noqa: E402
from evallab.stats import (bootstrap_ci, required_n_two_proportions,  # noqa: E402
                           threshold_sweep)

rows = load_calibration(settings.calibration_path, settings.corpus_dir)

print("Items needed to detect a change in pass rate (alpha=0.05, power=0.8):")
for p, d in [(0.80, 0.10), (0.80, 0.05), (0.90, 0.03)]:
    print(f"  baseline {p:.0%}, change {d:.0%}: {required_n_two_proportions(p, d):>5} per run")

correct = [float((numeric_grounding(r["answer"], [r["context"]], r["question"]) == 1.0)
                 == bool(r["human_faithful"])) for r in rows]
mean, lo, hi = bootstrap_ci(correct)
print(f"\nnumeric_grounding agreement with humans: {mean:.2f} (95% bootstrap CI {lo:.2f} to {hi:.2f}, n={len(rows)})")

if "--llm" in sys.argv:
    from ragas.metrics.collections import Faithfulness

    from evallab.ragas_eval import make_ragas_llm

    metric = Faithfulness(llm=make_ragas_llm(settings.judge_model))

    async def score_all():
        out = []
        for r in rows:
            try:
                res = await metric.ascore(user_input=r["question"], response=r["answer"],
                                          retrieved_contexts=[r["context"]])
                out.append(float(res.value))
            except Exception:
                out.append(float("nan"))
        return out

    scores = asyncio.run(score_all())
    Path("results").mkdir(exist_ok=True)
    Path("results/calibration_ragas_faithfulness.json").write_text(
        json.dumps([{"id": r["id"], "score": s, "human": r["human_faithful"]}
                    for r, s in zip(rows, scores)], indent=2))
    print(f"\nRAGAS faithfulness ({settings.judge_model}) threshold sweep vs human labels:")
    print(f"{'thr':>6}{'acc':>7}{'kappa':>7}{'H-rec':>7}{'H-prec':>8}{'J':>7}")
    for t in threshold_sweep(scores, [r["human_faithful"] for r in rows],
                             thresholds=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        print(f"{t['threshold']:>6.2f}{t['accuracy']:>7.2f}{t['kappa']:>7.2f}"
              f"{t['hallucination_recall']:>7.2f}{t['hallucination_precision']:>8.2f}{t['youden_j']:>7.2f}")
