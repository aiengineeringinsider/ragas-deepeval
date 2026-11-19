"""DeepEval 4 quickstart with a local Ollama judge.

Run:  python examples/deepeval_quickstart.py
Needs: ollama serve and qwen2.5:3b. No Confident AI account or OpenAI key.
"""
import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval import evaluate  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig  # noqa: E402
from deepeval.metrics import (AnswerRelevancyMetric, ContextualPrecisionMetric,  # noqa: E402
                              ContextualRecallMetric, ContextualRelevancyMetric,
                              FaithfulnessMetric, GEval, HallucinationMetric)
from deepeval.models import OllamaModel  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402

judge = OllamaModel(model="qwen2.5:3b", base_url="http://localhost:11434", temperature=0,
                    generation_kwargs={"num_ctx": 8192})

contexts = [
    "[SLA > Credit schedule (Business plan, 99.9% SLA)] Monthly uptime below 99.9% but at least "
    "99.0%: 10% credit. Below 99.0% but at least 95.0%: 25% credit. Below 95.0%: 50% credit.",
    "[SLA > Claiming credits] Credits are applied to the next invoice and are never paid out in cash.",
    "[Pricing > Business] The Business plan costs $0.018 per GB-month.",
]
test_case = LLMTestCase(
    input="A Business customer had 98.5% uptime in a month. What credit do they get?",
    actual_output="They get a 25% credit of that month's storage charges, paid out in cash.",
    expected_output="A 25% credit of that month's storage charges, applied to the next invoice.",
    retrieval_context=contexts,   # what the retriever returned (Faithfulness, Contextual*)
    context=contexts[:2],         # ground-truth passages (HallucinationMetric)
)

correctness = GEval(
    name="Correctness",
    criteria="Is the actual output factually consistent with the expected output? "
             "Penalize any contradiction or missing fact that changes the answer.",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                       SingleTurnParams.EXPECTED_OUTPUT],
    model=judge, threshold=0.6,
)

metrics = [
    AnswerRelevancyMetric(threshold=0.7, model=judge),
    FaithfulnessMetric(threshold=0.7, model=judge),
    HallucinationMetric(threshold=0.5, model=judge),     # lower is better
    ContextualPrecisionMetric(threshold=0.5, model=judge),
    ContextualRecallMetric(threshold=0.6, model=judge),
    ContextualRelevancyMetric(threshold=0.5, model=judge),
    correctness,
]

if __name__ == "__main__":
    result = evaluate(
        test_cases=[test_case], metrics=metrics,
        async_config=AsyncConfig(run_async=True, max_concurrent=2),  # gentle on a laptop
        display_config=DisplayConfig(print_results=True, verbose_mode=False),
    )
    for tr in result.test_results:
        for md in tr.metrics_data:
            score = md.score if md.score is not None else float("nan")   # None on judge error
            print(f"{md.name:<24} score={score:.2f} threshold={md.threshold} success={md.success}")
            print(f"   reason: {(md.reason or md.error or '')[:110]}")
