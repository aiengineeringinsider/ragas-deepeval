"""DeepEval 4.x evaluation with a local Ollama judge.

Key DeepEval concepts used here:
  * LLMTestCase    : one interaction (input, actual_output, expected_output,
                     retrieval_context, context).
  * Golden         : a test case template without actual_output; the system
                     under test fills it in at run time.
  * metric.threshold: pass/fail line. A test case passes only if every metric
                     scores at or above its threshold (HallucinationMetric is
                     inverted: lower is better, it passes at or BELOW threshold).
  * OllamaModel    : DeepEval's built-in Ollama judge. It uses Ollama's
                     structured outputs (format=<JSON schema>), which suits
                     small models.

Note on naming: DeepEval 4 renamed LLMTestCaseParams to SingleTurnParams. The
old name still imports with a DeprecationWarning.
"""
from __future__ import annotations

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from .config import settings
from .records import EvalRecord


def make_judge(model: str | None = None):
    from deepeval.models import OllamaModel

    return OllamaModel(model=model or settings.judge_model, base_url=settings.ollama_host,
                       temperature=0.0, generation_kwargs={"num_ctx": settings.num_ctx})


def to_test_case(rec: EvalRecord):
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=rec.question,
        actual_output=rec.answer,
        expected_output=rec.reference,
        retrieval_context=rec.contexts or None,
        # HallucinationMetric compares the output with `context`, the ground
        # truth passages. We use the retrieved contexts here; in Module 8 we
        # discuss when to pass gold passages instead.
        context=rec.contexts or None,
        name=rec.id,
        tags=[rec.category],
        completion_time=rec.latency_s,
        metadata={"model": rec.model},
    )


def correctness_geval(judge, threshold: float = 0.6):
    from deepeval.metrics import GEval
    from deepeval.test_case import SingleTurnParams

    return GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare the facts, numbers, and plan names in 'actual output' with 'expected output'.",
            "Heavily penalize any number, price, duration, or percentage that differs from 'expected output'.",
            "Penalize omitting a fact from 'expected output' that changes the final answer.",
            "Do not penalize extra wording, citations like [1], or shown arithmetic if the facts are right.",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT],
        model=judge, threshold=threshold, async_mode=False,
    )


def refusal_geval(judge, threshold: float = 0.7):
    """Did the assistant refuse exactly when it should have?"""
    from deepeval.metrics import GEval
    from deepeval.test_case import SingleTurnParams

    return GEval(
        name="Appropriate Refusal",
        criteria=(
            "If 'expected output' says the documentation does not contain the answer, the "
            "'actual output' must decline to answer and must not invent facts. If 'expected "
            "output' contains a real answer, the 'actual output' must NOT refuse."
        ),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=judge, threshold=threshold, async_mode=False,
    )


def default_metrics(judge, include_geval: bool = True) -> list:
    from deepeval.metrics import (AnswerRelevancyMetric, ContextualPrecisionMetric,
                                  ContextualRecallMetric, ContextualRelevancyMetric,
                                  FaithfulnessMetric, HallucinationMetric)

    common = dict(model=judge, include_reason=True, async_mode=False)
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, **common),
        FaithfulnessMetric(threshold=0.7, **common),
        HallucinationMetric(threshold=0.5, **common),
        ContextualPrecisionMetric(threshold=0.5, **common),
        ContextualRecallMetric(threshold=0.6, **common),
        ContextualRelevancyMetric(threshold=0.5, **common),
    ]
    if include_geval:
        metrics += [correctness_geval(judge), refusal_geval(judge)]
    return metrics


def score_record(rec: EvalRecord, metrics: list) -> EvalRecord:
    """Run metrics one by one so a single judge failure does not lose the row."""
    tc = to_test_case(rec)
    for m in metrics:
        name = getattr(m, "name", None) or m.__class__.__name__
        if rec.category == "unanswerable" and m.__class__.__name__.startswith("Contextual"):
            rec.scores[name] = float("nan")
            rec.reasons[name] = "skipped: reference-based metric on unanswerable item"
            continue
        try:
            m.measure(tc)
            rec.scores[name] = float(m.score)
            rec.reasons[name] = (m.reason or "")[:500]
        except Exception as e:
            rec.scores[name] = float("nan")
            rec.reasons[name] = f"error: {type(e).__name__}: {e}"[:300]
    return rec
