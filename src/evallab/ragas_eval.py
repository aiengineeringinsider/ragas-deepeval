"""RAGAS 0.4 evaluation against local Ollama models.

RAGAS 0.4 metrics live in ragas.metrics.collections. Each metric is built with
an LLM (and, for answer relevancy, an embedding model) and scored with the
async ascore(...) method, which takes the exact fields that metric needs.

Ollama serves an OpenAI-compatible API at http://localhost:11434/v1, so we
give RAGAS an AsyncOpenAI client pointed at it. No OpenAI account is involved;
the api_key value is a required placeholder that Ollama ignores.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from .config import settings
from .records import EvalRecord


def make_ragas_llm(model: str | None = None, temperature: float = 0.0, max_tokens: int = 2048):
    import instructor
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(base_url=settings.openai_compat_url, api_key="ollama",
                         timeout=settings.request_timeout_s)
    # Mode.JSON sends response_format={"type":"json_object"}, which Ollama
    # supports. Small models follow it far more reliably than tool calling.
    return llm_factory(model or settings.judge_model, provider="openai", client=client,
                       mode=instructor.Mode.JSON, temperature=temperature,
                       max_tokens=max_tokens)


def make_ragas_embeddings(model: str | None = None):
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings

    client = AsyncOpenAI(base_url=settings.openai_compat_url, api_key="ollama")
    return OpenAIEmbeddings(client=client, model=model or settings.embed_model)


@dataclass
class MetricSpec:
    name: str
    metric: Any
    build_kwargs: Callable[[EvalRecord], dict]
    needs_reference: bool = False   # skip for unanswerable items
    needs_context: bool = True


def default_metric_specs(llm, embeddings) -> list[MetricSpec]:
    from ragas.metrics.collections import (AnswerRelevancy, ContextPrecision, ContextRecall,
                                           ContextRelevance, FactualCorrectness, Faithfulness,
                                           NoiseSensitivity)

    return [
        MetricSpec("faithfulness", Faithfulness(llm=llm),
                   lambda r: dict(user_input=r.question, response=r.answer,
                                  retrieved_contexts=r.contexts)),
        MetricSpec("answer_relevancy", AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=3),
                   lambda r: dict(user_input=r.question, response=r.answer), needs_context=False),
        MetricSpec("context_precision", ContextPrecision(llm=llm),
                   lambda r: dict(user_input=r.question, reference=r.reference,
                                  retrieved_contexts=r.contexts), needs_reference=True),
        MetricSpec("context_recall", ContextRecall(llm=llm),
                   lambda r: dict(user_input=r.question, retrieved_contexts=r.contexts,
                                  reference=r.reference), needs_reference=True),
        MetricSpec("context_relevance", ContextRelevance(llm=llm),
                   lambda r: dict(user_input=r.question, retrieved_contexts=r.contexts)),
        MetricSpec("factual_correctness", FactualCorrectness(llm=llm, mode="f1"),
                   lambda r: dict(response=r.answer, reference=r.reference),
                   needs_reference=True, needs_context=False),
        MetricSpec("noise_sensitivity", NoiseSensitivity(llm=llm, mode="relevant"),
                   lambda r: dict(user_input=r.question, response=r.answer,
                                  reference=r.reference, retrieved_contexts=r.contexts),
                   needs_reference=True),
    ]


async def _score_one(spec: MetricSpec, rec: EvalRecord, sem: asyncio.Semaphore,
                     retries: int = 2) -> tuple[float, str | None, float]:
    if spec.needs_reference and rec.category == "unanswerable":
        return math.nan, "skipped: reference-based metric on unanswerable item", 0.0
    if spec.needs_context and not rec.contexts:
        return math.nan, "skipped: no retrieved contexts", 0.0
    async with sem:
        last_err = None
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                res = await spec.metric.ascore(**spec.build_kwargs(rec))
                reason = getattr(res, "reason", None)
                return float(res.value), reason, time.perf_counter() - t0
            except Exception as e:  # small judges sometimes emit invalid JSON
                last_err = f"{type(e).__name__}: {e}"[:300]
        return math.nan, f"error: {last_err}", 0.0


async def evaluate_records(records: list[EvalRecord], specs: list[MetricSpec],
                           concurrency: int = 2) -> list[EvalRecord]:
    """Score every record with every metric. NaN marks skipped or failed cells.

    concurrency=2 is deliberate: a laptop running a 3B model gains little from
    more parallel requests, and too many causes timeouts that look like low
    scores if you are not careful.
    """
    sem = asyncio.Semaphore(concurrency)
    for rec in records:
        results = await asyncio.gather(*[_score_one(s, rec, sem) for s in specs])
        for spec, (value, reason, latency) in zip(specs, results):
            rec.scores[spec.name] = value
            if reason:
                rec.reasons[spec.name] = reason
            rec.scores.setdefault("_judge_latency_s", 0.0)
            rec.scores["_judge_latency_s"] += latency
        printable = {k: (round(v, 2) if v == v else "nan")
                     for k, v in rec.scores.items() if not k.startswith("_")}
        print(f"{rec.id}: {printable}")
    return records


def summarize(records: list[EvalRecord]) -> dict[str, dict[str, float]]:
    """Mean, count, and failure count per metric (NaNs excluded from the mean)."""
    names = sorted({k for r in records for k in r.scores if not k.startswith("_")})
    out = {}
    for n in names:
        vals = [r.scores.get(n, math.nan) for r in records]
        ok = [v for v in vals if v == v]
        errors = sum(1 for r in records if str(r.reasons.get(n, "")).startswith("error"))
        out[n] = {"mean": round(sum(ok) / len(ok), 4) if ok else math.nan,
                  "n_scored": len(ok), "n_errors": errors}
    return out
