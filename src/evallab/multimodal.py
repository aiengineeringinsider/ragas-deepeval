"""Multimodal RAG evaluation with qwen2.5vl:3b (Module 11).

The "retrieval" step returns images (a pricing table, a latency chart, an SLA
credit chart) instead of text chunks. We then evaluate three things:
  1. answer correctness against a reference (deterministic key-fact match);
  2. multimodal faithfulness: is the answer supported by the image? (RAGAS
     MultiModalFaithfulness, with the vision model as judge);
  3. multimodal relevance: does the answer address the question given the image?
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings

VISION_SYSTEM = ("You answer questions about Helios Cloud Storage using ONLY the provided image. "
                 "If the image does not contain the answer, say you cannot tell from the image. "
                 "Answer in one or two sentences and quote exact numbers from the image.")


@dataclass
class MMItem:
    id: str
    question: str
    images: list[str]
    reference: str
    key_facts: list[str] = field(default_factory=list)  # strings that must appear in a correct answer


def load_mm_golden(path: Path, image_dir: Path) -> list[MMItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            d["images"] = [str(image_dir / p) for p in d["images"]]
            items.append(MMItem(**d))
    return items


def answer_with_image(question: str, images: list[str], model: str | None = None) -> tuple[str, float]:
    import ollama

    client = ollama.Client(host=settings.ollama_host, timeout=settings.request_timeout_s)
    t0 = time.perf_counter()
    resp = client.chat(model=model or settings.vision_model,
                       messages=[{"role": "system", "content": VISION_SYSTEM},
                                 {"role": "user", "content": question, "images": images}],
                       options={"temperature": 0.0, "num_ctx": settings.num_ctx, "seed": 42})
    return resp.message.content.strip(), time.perf_counter() - t0


def _norm(s: str) -> str:
    return re.sub(r"[\s,]", "", s.lower())


def key_fact_recall(answer: str, key_facts: list[str]) -> float:
    """Deterministic reference-based check: share of required facts present."""
    if not key_facts:
        return float("nan")
    a = _norm(answer)
    return sum(1 for f in key_facts if _norm(f) in a) / len(key_facts)


def make_vision_ragas_llm(model: str | None = None):
    """RAGAS multimodal metrics need an InstructorLLM backed by a vision model."""
    from .ragas_eval import make_ragas_llm

    return make_ragas_llm(model or settings.vision_model, max_tokens=1024)


async def ragas_multimodal_scores(llm, question: str, answer: str, images: list[str]) -> dict:
    from ragas.metrics.collections import MultiModalFaithfulness, MultiModalRelevance

    out = {}
    for name, metric, kwargs in [
        ("mm_faithfulness", MultiModalFaithfulness(llm=llm),
         dict(response=answer, retrieved_contexts=images)),
        ("mm_relevance", MultiModalRelevance(llm=llm),
         dict(user_input=question, response=answer, retrieved_contexts=images)),
    ]:
        try:
            res = await metric.ascore(**kwargs)
            out[name] = float(res.value)
            out[name + "_reason"] = getattr(res, "reason", "")
        except Exception as e:
            out[name] = float("nan")
            out[name + "_reason"] = f"error: {type(e).__name__}: {e}"[:200]
    return out
