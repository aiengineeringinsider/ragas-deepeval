"""Module 8: deterministic custom metrics vs human labels, then an LLM-judged
RAGAS DiscreteMetric on the cases the deterministic check cannot see.

Run:  python examples/custom_metrics_demo.py            (part 1 needs no model)
      python examples/custom_metrics_demo.py --llm      (part 2 uses qwen2.5:3b)
"""
import asyncio
import sys

sys.path.insert(0, "src")
from evallab.config import settings  # noqa: E402
from evallab.custom_metrics import numeric_grounding, ragas_policy_metric  # noqa: E402
from evallab.judges import load_calibration  # noqa: E402
from evallab.stats import binary_agreement, hallucination_view  # noqa: E402

rows = load_calibration(settings.calibration_path, settings.corpus_dir)

# Part 1: numeric grounding as a hallucination detector (no model at all)
pred = []
for r in rows:
    g = numeric_grounding(r["answer"], [r["context"]], r["question"])
    pred.append(1 if g == 1.0 else 0)          # any ungrounded number -> "unfaithful"
human = [r["human_faithful"] for r in rows]
a, h = binary_agreement(human, pred), hallucination_view(human, pred)
print(f"numeric_grounding as a judge on {len(rows)} cases: accuracy={a.accuracy:.2f} "
      f"kappa={a.cohen_kappa:.2f} hallucination_recall={h.recall:.2f} "
      f"hallucination_precision={h.precision:.2f}")
by_type = {}
for r, p in zip(rows, pred):
    if r["human_faithful"] == 0:
        by_type.setdefault(r["error_type"], []).append(p == 0)
for t, caught in sorted(by_type.items()):
    print(f"  {t:<18} caught {sum(caught)}/{len(caught)}")
false_flags = [r["id"] for r, p in zip(rows, pred) if r["human_faithful"] == 1 and p == 0]
print(f"  faithful answers wrongly flagged: {false_flags}")

# Part 2: LLM-judged discrete metric for promises the context does not make
if "--llm" in sys.argv:
    from evallab.ragas_eval import make_ragas_llm

    llm = make_ragas_llm("qwen2.5:3b")
    metric = ragas_policy_metric()

    async def run():
        for cid in ["h010", "h014", "h028", "h034", "h027"]:
            r = next(x for x in rows if x["id"] == cid)
            res = await metric.ascore(llm=llm, context=r["context"], question=r["question"],
                                      answer=r["answer"])
            print(f"{cid} human_faithful={r['human_faithful']} commitment_safety={res.value}"
                  f"  {getattr(res, 'reason', '')[:80]}")

    asyncio.run(run())
