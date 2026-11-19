"""Custom evaluators: deterministic checks first, LLM judges second.

Deterministic metrics are cheap, exact, and never drift. Use them for every
property that can be checked with code (citations, refusals, numbers, retrieval
hit rate), and save LLM judges for properties that genuinely need language
understanding.
"""
from __future__ import annotations

import re

from .rag import REFUSAL

NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
CITE = re.compile(r"\[(\d+)\]")


# --------------------------------------------------------------------------
# Retrieval metrics (reference-based, no LLM): need gold source documents
# --------------------------------------------------------------------------
def hit_rate_at_k(retrieved_sources: list[str], gold_sources: list[str], k: int) -> float:
    if not gold_sources:
        return float("nan")
    return 1.0 if set(retrieved_sources[:k]) & set(gold_sources) else 0.0


def source_recall_at_k(retrieved_sources: list[str], gold_sources: list[str], k: int) -> float:
    """Fraction of gold documents that appear in the top-k. Matters for multi-hop."""
    if not gold_sources:
        return float("nan")
    return len(set(retrieved_sources[:k]) & set(gold_sources)) / len(set(gold_sources))


def mrr(retrieved_sources: list[str], gold_sources: list[str]) -> float:
    if not gold_sources:
        return float("nan")
    for rank, s in enumerate(retrieved_sources, start=1):
        if s in gold_sources:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------
# Answer metrics (deterministic)
# --------------------------------------------------------------------------
def _norm_num(tok: str) -> str:
    return tok.replace("$", "").replace(",", "").rstrip("%").rstrip(".")


_N = r"(\$?\d[\d,]*(?:\.\d+)?)"
_UNIT = r"(?:\s*(?:GB|TB|MB|ms|days?|hours?|%))?"
ARITH = re.compile(_N + _UNIT + r"\s*(x|\u00d7|\*|times|\+|-|minus|plus)\s*" + _N + _UNIT
                   + r"\s*=\s*" + _N, re.I)
_OPS = {"x": lambda a, b: a * b, "\u00d7": lambda a, b: a * b, "*": lambda a, b: a * b,
        "times": lambda a, b: a * b, "+": lambda a, b: a + b, "plus": lambda a, b: a + b,
        "-": lambda a, b: a - b, "minus": lambda a, b: a - b}


def numeric_grounding(answer: str, contexts: list[str], question: str = "",
                      allow: set[str] | None = None) -> float:
    """Share of numbers in the answer that are grounded. Returns 1.0 if there are none.

    A number is grounded if it appears in the contexts, or if it is the result of
    an arithmetic expression SHOWN in the answer (e.g. "20,000 GB x $0.018 = $360")
    whose operands are grounded and whose arithmetic is correct (1% tolerance).
    Numbers stated in the question (e.g. "stores 10,000 GB") count as grounded.
    A wrong calculation therefore lowers the score, which is what we want.
    Citation markers like [1] are ignored.
    """
    allow = allow or set()
    body = CITE.sub(" ", answer)
    ans_nums = [_norm_num(t) for t in NUM.findall(body) if _norm_num(t)]
    if not ans_nums:
        return 1.0
    ctx_nums = {_norm_num(t) for c in [*contexts, question] for t in NUM.findall(c) if _norm_num(t)} | allow
    grounded = set(ctx_nums)
    for m in ARITH.finditer(body):
        x, op, y, z = _norm_num(m.group(1)), m.group(2).lower(), _norm_num(m.group(3)), _norm_num(m.group(4))
        if not (_isfloat(x) and _isfloat(y) and _isfloat(z)):
            continue
        ok_operands = all(v in grounded for v in (x, y))
        expected = _OPS[op](float(x), float(y))
        if ok_operands and abs(expected - float(z)) <= 0.01 * max(abs(expected), 1e-9):
            grounded.add(z)
    return sum(1 for n in ans_nums if n in grounded) / len(ans_nums)


def _isfloat(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def citation_validity(answer: str, n_contexts: int) -> float:
    """1.0 if every [n] citation points at a real passage, and at least one exists
    (refusals are exempt). Invalid or missing citations lower the score."""
    if is_refusal(answer):
        return 1.0
    cites = [int(c) for c in CITE.findall(answer)]
    if not cites:
        return 0.0
    return sum(1 for c in cites if 1 <= c <= n_contexts) / len(cites)


REFUSAL_PATTERNS = re.compile(
    r"(don't|do not|doesn't|does not) (have|contain|mention)|not (mentioned|covered|available) in|"
    r"no information|cannot find|can't find|not in the (helios )?documentation", re.I)


def is_refusal(answer: str) -> bool:
    return answer.strip().startswith(REFUSAL[:30]) or bool(REFUSAL_PATTERNS.search(answer))


def refusal_correctness(answer: str, category: str) -> float:
    """1.0 when the assistant refuses exactly on unanswerable items."""
    should_refuse = category == "unanswerable"
    return 1.0 if is_refusal(answer) == should_refuse else 0.0


def deterministic_scores(rec, k: int = 4) -> dict[str, float]:
    return {
        "hit_rate@k": hit_rate_at_k(rec.sources, rec.reference_sources, k),
        "source_recall@k": source_recall_at_k(rec.sources, rec.reference_sources, k),
        "mrr": mrr(rec.sources, rec.reference_sources),
        "numeric_grounding": numeric_grounding(rec.answer, rec.contexts, rec.question),
        "citation_validity": citation_validity(rec.answer, len(rec.contexts)),
        "refusal_correctness": refusal_correctness(rec.answer, rec.category),
    }


# --------------------------------------------------------------------------
# RAGAS custom metric: an LLM-judged, discrete policy check
# --------------------------------------------------------------------------
def ragas_policy_metric():
    from ragas.metrics import DiscreteMetric

    return DiscreteMetric(
        name="commitment_safety",
        allowed_values=["pass", "fail"],
        prompt=(
            "You audit answers from a cloud-storage support assistant.\n"
            "FAIL the answer if it promises anything the context does not state: a refund, "
            "a cash payout, a guaranteed response time, a discount, or a feature on a plan "
            "that does not include it. Otherwise PASS.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
            "Return value 'pass' or 'fail' and a one-sentence reason."
        ),
    )


# --------------------------------------------------------------------------
# DeepEval custom metric: wrap a deterministic check in BaseMetric
# --------------------------------------------------------------------------
def make_numeric_grounding_metric(threshold: float = 1.0):
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase

    class NumericGroundingMetric(BaseMetric):
        """Fails when the answer states a number not supported by the context."""

        def __init__(self, threshold: float = threshold):
            self.threshold = threshold
            self.include_reason = True
            self.async_mode = False
            self.strict_mode = False
            self.verbose_mode = False
            self.evaluation_model = "deterministic"

        def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
            ctx = test_case.retrieval_context or []
            self.score = numeric_grounding(test_case.actual_output or "", [str(c) for c in ctx],
                                           test_case.input or "")
            self.success = self.score >= self.threshold
            self.reason = ("all numbers grounded" if self.success
                           else f"{(1 - self.score):.0%} of numbers not found in or derivable from context")
            return self.score

        async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
            return self.measure(test_case)

        def is_successful(self) -> bool:
            return bool(self.success)

        @property
        def __name__(self):
            return "Numeric Grounding"

    return NumericGroundingMetric()
