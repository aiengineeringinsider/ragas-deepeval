"""Module 16: component-level evaluation with DeepEval tracing.

Run:  python scripts/13_traced_eval.py            # deterministic span metrics
      python scripts/13_traced_eval.py --judge    # + validated faithfulness judge on the generator span
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepeval.dataset import EvaluationDataset, Golden  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig  # noqa: E402

from evallab.config import settings  # noqa: E402
from evallab.custom_metrics import make_numeric_grounding_metric, make_validated_faithfulness_metric  # noqa: E402
from evallab.golden import load_golden  # noqa: E402
from evallab.llm import OllamaChat  # noqa: E402
from evallab.rag import build_default_pipeline  # noqa: E402
from evallab.tracing import SourceRecallMetric, TracedPipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--judge", action="store_true")
ap.add_argument("--limit", type=int, default=10)
args = ap.parse_args()

gen_metrics = [make_numeric_grounding_metric()]
if args.judge:
    gen_metrics.append(make_validated_faithfulness_metric(OllamaChat(settings.judge_model)))
traced = TracedPipeline(build_default_pipeline(), retrieval_metrics=[SourceRecallMetric()],
                        generation_metrics=gen_metrics)
items = [i for i in load_golden(settings.golden_path) if i.reference_sources][: args.limit]
dataset = EvaluationDataset(goldens=[Golden(input=i.question, additional_metadata={"gold": i.reference_sources})
                                     for i in items])
for golden in dataset.evals_iterator(async_config=AsyncConfig(run_async=False)):
    traced.answer(golden.input, golden.additional_metadata["gold"])
