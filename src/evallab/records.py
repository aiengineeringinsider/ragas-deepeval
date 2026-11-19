"""Evaluation records: one row per (golden item, system response).

Generation and evaluation are separated on purpose. We run the RAG system
once, save every response with its contexts to JSONL, and then evaluate that
frozen file with as many metrics and judges as we like. Scores become
reproducible, and swapping a judge never re-rolls the system's answers.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .golden import GoldenItem


@dataclass
class EvalRecord:
    id: str
    question: str
    reference: str
    category: str
    answer: str
    contexts: list[str]
    sources: list[str]
    reference_sources: list[str]
    latency_s: float
    model: str
    scores: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def run_system(pipeline, items: list[GoldenItem], progress: bool = True) -> list[EvalRecord]:
    records = []
    for n, it in enumerate(items, start=1):
        r = pipeline.answer(it.question)
        records.append(EvalRecord(
            id=it.id, question=it.question, reference=it.reference, category=it.category,
            answer=r.answer, contexts=r.contexts, sources=r.sources,
            reference_sources=it.reference_sources, latency_s=round(r.total_latency_s, 3),
            model=r.model))
        if progress:
            print(f"[{n:>2}/{len(items)}] {it.id} {r.total_latency_s:5.1f}s  {r.answer[:70]!r}")
    return records


def save_records(records: list[EvalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(r.to_json() for r in records) + "\n", encoding="utf-8")


def load_records(path: Path) -> list[EvalRecord]:
    return [EvalRecord(**json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
