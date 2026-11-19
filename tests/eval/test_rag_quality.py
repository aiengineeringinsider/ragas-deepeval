"""DeepEval regression tests against the live RAG system (Module 6 and 12).

Run with either:
  pytest tests/eval -m ollama -v
  deepeval test run tests/eval/test_rag_quality.py

Each golden item becomes one parametrized test. The RAG system generates the
answer at test time (Golden -> LLMTestCase), and assert_test fails the test if
any metric is below its threshold.

What blocks a merge, and why: only instruments we validated.
  * deterministic: refusal correctness, numeric grounding, citation validity
  * LLM: the single-call faithfulness judge validated against 40 human labels
    (Module 9: qwen2.5:3b kappa 0.80), and a single-call relevance judge.
DeepEval's built-in AnswerRelevancy and Faithfulness split answers into
statements and judge each one. With a 3B judge that failed correct answers in
the lab (8 of 8 tests failed), so they run in the nightly report
(scripts/06_deepeval_eval.py), not in this gate. Swap them back in when you
have a judge you have validated for them.
"""
import os

import pytest

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval import assert_test  # noqa: E402
from deepeval.dataset import EvaluationDataset, Golden  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from evallab.config import settings  # noqa: E402
from evallab.custom_metrics import (citation_validity, make_numeric_grounding_metric,  # noqa: E402
                                    make_relevance_metric, make_validated_faithfulness_metric,
                                    refusal_correctness)
from evallab.golden import load_golden  # noqa: E402

SMOKE = os.getenv("EVALLAB_SMOKE", "1") == "1"   # CI default: a fast, representative subset
SMOKE_IDS = {"g001", "g003", "g009", "g012", "g015", "g019", "g025", "g027"}

items = load_golden(settings.golden_path)
if SMOKE:
    items = [i for i in items if i.id in SMOKE_IDS]

dataset = EvaluationDataset(goldens=[
    Golden(input=i.question, expected_output=i.reference,
           additional_metadata={"id": i.id, "category": i.category}) for i in items])


@pytest.fixture(scope="session")
def rag():
    from evallab.rag import build_default_pipeline

    p = build_default_pipeline()
    if p.store.count() == 0:
        pytest.skip("Chroma collection is empty: run scripts/02_ingest.py first")
    return p


@pytest.fixture(scope="session")
def judge():
    from evallab.llm import OllamaChat

    return OllamaChat(settings.judge_model)


@pytest.mark.ollama
@pytest.mark.parametrize("golden", dataset.goldens, ids=lambda g: g.additional_metadata["id"])
def test_rag_answer_quality(golden: Golden, rag, judge):
    result = rag.answer(golden.input)
    category = golden.additional_metadata["category"]
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result.answer,
        expected_output=golden.expected_output,
        retrieval_context=result.contexts,
        completion_time=result.total_latency_s,
    )
    # Deterministic gates first: cheap, exact, and they never drift.
    assert refusal_correctness(result.answer, category) == 1.0, (
        f"refusal behaviour wrong for {category}: {result.answer!r}")

    if category == "unanswerable":
        # The refusal itself is checked above; here: no invented facts.
        metrics = [make_validated_faithfulness_metric(judge)]
    else:
        assert citation_validity(result.answer, len(result.contexts)) == 1.0, (
            f"answer cites a passage that does not exist or cites nothing: {result.answer!r}")
        metrics = [
            make_numeric_grounding_metric(threshold=1.0),
            make_validated_faithfulness_metric(judge),
            make_relevance_metric(judge),
        ]
    assert_test(test_case, metrics, run_async=False)
