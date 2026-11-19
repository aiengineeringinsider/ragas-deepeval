"""Component-level evaluation with DeepEval tracing (Module 16).

End-to-end metrics tell you that an answer is bad; component metrics tell you
which component made it bad. We wrap the two stages of the RAG pipeline in
@observe spans, attach a retrieval metric to the retriever span and a
generation metric to the generator span, and let DeepEval score each span.
"""
from __future__ import annotations

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span, update_current_trace

import re

from .custom_metrics import source_recall_at_k

_HEADER = re.compile(r"^\[(?P<title>[^>\]]+?)(?: > [^\]]*)?\]")


def chunk_title(text: str) -> str:
    """Document title from a chunk header like '[Helios Cloud Storage: Pricing Plans > Business]'."""
    m = _HEADER.match(text.strip())
    return m.group("title").strip() if m else ""


class SourceRecallMetric(BaseMetric):
    """Deterministic retrieval metric for a retriever span: share of gold
    documents among the retrieved chunks. Span metadata does not reach the span's
    test case in DeepEval 4.3, so the gold document titles travel in `context`
    and retrieved titles are read from the chunk headers."""

    def __init__(self, threshold: float = 1.0):
        self.threshold, self.include_reason, self.async_mode = threshold, True, False
        self.strict_mode, self.verbose_mode, self.evaluation_model = False, False, "deterministic"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        gold = list(test_case.context or [])
        got = [chunk_title(t) for t in (test_case.retrieval_context or [])]
        self.score = 1.0 if not gold else source_recall_at_k(got, gold, len(got))
        self.success = self.score >= self.threshold
        missing = sorted(set(gold) - set(got))
        self.reason = "all gold documents retrieved" if not missing else f"missing: {missing}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Source Recall"


class TracedPipeline:
    """The RAG pipeline with one span per component. Metrics are passed in, so
    tests can attach deterministic metrics and nightly runs can attach judges."""

    def __init__(self, pipeline, retrieval_metrics: list | None = None,
                 generation_metrics: list | None = None):
        from .chunking import load_corpus
        from .config import settings

        self.pipeline = pipeline
        # file name -> document title, to express gold sources the way chunks name them
        self.titles = {c.source: chunk_title(c.text) for c in load_corpus(settings.corpus_dir)}
        self.retrieval_metrics = retrieval_metrics or []
        self.generation_metrics = generation_metrics or []

    @observe(type="retriever")
    def retrieve(self, question: str, gold_sources: list[str] | None = None):
        hits, _ = self.pipeline.retrieve(question)
        # Span fields mirror LLMTestCase fields (actual_output is called output).
        update_current_span(input=question, output=f"{len(hits)} chunks",
                            retrieval_context=[h.text for h in hits],
                            context=[self.titles.get(s, s) for s in (gold_sources or [])],
                            metrics=self.retrieval_metrics)
        return hits

    @observe(type="llm")
    def generate(self, question: str, hits) -> str:
        from .rag import SYSTEM_PROMPT, format_context

        resp = self.pipeline.llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{format_context(hits)}\n\nQuestion: {question}"}],
            temperature=0.0)
        answer = resp.text.strip()
        update_current_span(input=question, output=answer,
                            retrieval_context=[h.text for h in hits],
                            metrics=self.generation_metrics)
        return answer

    @observe(type="agent")
    def answer(self, question: str, gold_sources: list[str] | None = None) -> str:
        hits = self.retrieve(question, gold_sources)
        answer = self.generate(question, hits)
        update_current_trace(input=question, output=answer)
        return answer
