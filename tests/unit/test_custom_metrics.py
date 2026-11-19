import math

import pytest

from evallab.custom_metrics import (citation_validity, hit_rate_at_k, is_refusal,
                                    make_numeric_grounding_metric, mrr, numeric_grounding,
                                    refusal_correctness, source_recall_at_k)

CTX = ["Business costs $0.018 per GB-month with a minimum monthly commitment of $250."]


Q = "A Business customer stores 20,000 GB. What is the storage charge?"


@pytest.mark.parametrize("answer,expected", [
    ("Business costs $0.018 per GB-month [1].", 1.0),
    ("20,000 GB x $0.018 = $360 [1].", 1.0),          # shown, correct arithmetic
    ("20,000 GB x $0.018 = $380 [1].", 2 / 3),        # shown, WRONG arithmetic
    ("They pay $360 [1].", 0.0),                      # unshown arithmetic is not trusted
    ("Business costs $0.016 per GB-month [1].", 0.0),  # fabricated price
    ("There is no number here.", 1.0),
])
def test_numeric_grounding(answer, expected):
    assert numeric_grounding(answer, CTX, question=Q) == pytest.approx(expected)


def test_citation_validity():
    assert citation_validity("Yes [1][2].", n_contexts=2) == 1.0
    assert citation_validity("Yes [1][7].", n_contexts=2) == 0.5
    assert citation_validity("Yes, definitely.", n_contexts=2) == 0.0
    assert citation_validity("I don't have that information in the Helios documentation.", 2) == 1.0


def test_refusal_detection_and_correctness():
    assert is_refusal("I don't have that information in the Helios documentation.")
    assert is_refusal("The documentation does not mention a GPU service.")
    assert not is_refusal("Starter costs $0.021 per GB-month [1].")
    assert refusal_correctness("I don't have that information.", "unanswerable") == 1.0
    assert refusal_correctness("I don't have that information.", "single_hop") == 0.0


def test_retrieval_metrics():
    got = ["a.md", "b.md", "c.md"]
    assert hit_rate_at_k(got, ["c.md"], k=2) == 0.0
    assert hit_rate_at_k(got, ["c.md"], k=3) == 1.0
    assert source_recall_at_k(got, ["a.md", "z.md"], k=3) == 0.5
    assert mrr(got, ["b.md"]) == 0.5
    assert math.isnan(mrr(got, []))


def test_deepeval_custom_metric_wraps_deterministic_check():
    from deepeval.test_case import LLMTestCase

    m = make_numeric_grounding_metric(threshold=1.0)
    ok = LLMTestCase(input="q", actual_output="It costs $0.018 [1].", retrieval_context=CTX)
    bad = LLMTestCase(input="q", actual_output="It costs $0.011 [1].", retrieval_context=CTX)
    assert m.measure(ok) == 1.0 and m.is_successful()
    assert m.measure(bad) == 0.0 and not m.is_successful()
