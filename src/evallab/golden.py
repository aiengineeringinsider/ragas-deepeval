"""Golden dataset: schema, validation, synthetic candidate generation.

A golden item is the ground truth the system is judged against. The schema is
strict on purpose: a typo in a category or an empty reference is a bug in the
test suite, and it should fail loudly at load time, not as a strange score.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["single_hop", "multi_hop", "numeric_reasoning", "unanswerable"]
Difficulty = Literal["easy", "medium", "hard"]


class GoldenItem(BaseModel):
    id: str = Field(pattern=r"^g\d{3,}$")
    question: str = Field(min_length=8)
    reference: str = Field(min_length=3)
    reference_sources: list[str] = Field(default_factory=list)
    category: Category
    difficulty: Difficulty = "medium"
    tags: list[str] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def must_be_question_like(cls, v: str) -> str:
        if not v.rstrip().endswith("?"):
            raise ValueError("questions should end with '?' so intent is explicit")
        return v

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = GoldenItem.model_validate_json(line)
        if item.id in seen:
            raise ValueError(f"{path}:{n} duplicate id {item.id}")
        seen.add(item.id)
        items.append(item)
    return items


def validate_sources(items: list[GoldenItem], corpus_dir: Path) -> list[str]:
    """Return a list of problems: references to documents that do not exist."""
    names = {p.name for p in corpus_dir.glob("*.md")}
    problems = []
    for it in items:
        for s in it.reference_sources:
            if s not in names:
                problems.append(f"{it.id}: unknown source {s}")
        if it.answerable and not it.reference_sources:
            problems.append(f"{it.id}: answerable item has no reference_sources")
    return problems


def dataset_card(items: list[GoldenItem]) -> dict:
    """Composition summary. Print it in every report: a score means nothing
    without knowing what mix of questions produced it."""
    return {
        "n_items": len(items),
        "by_category": dict(Counter(i.category for i in items)),
        "by_difficulty": dict(Counter(i.difficulty for i in items)),
        "multi_source": sum(len(i.reference_sources) > 1 for i in items),
    }


# --------------------------------------------------------------------------
# Synthetic candidate generation (human review is still mandatory)
# --------------------------------------------------------------------------
SYNTH_PROMPT = """You write evaluation questions for a customer-support assistant.
From the passage below, write ONE question a real customer might ask and its
correct answer, using only facts stated in the passage.
Return JSON: {{"question": "...", "answer": "..."}}

Passage:
{passage}"""

SYNTH_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["question", "answer"],
}


def synthesize_candidates(chunks, llm, n_per_chunk: int = 1) -> list[dict]:
    """Generate candidate Q/A pairs from chunks. Output is a *draft*: each
    candidate is written with status="needs_review" and must be checked by a
    person before it is promoted into the golden file."""
    from .llm import parse_json

    out = []
    for c in chunks:
        for _ in range(n_per_chunk):
            resp = llm.chat([{"role": "user", "content": SYNTH_PROMPT.format(passage=c.text)}],
                            json_schema=SYNTH_SCHEMA, temperature=0.7, seed=None)
            try:
                qa = parse_json(resp.text)
            except (ValueError, json.JSONDecodeError):
                continue
            out.append({"question": qa["question"].strip(), "reference": qa["answer"].strip(),
                        "reference_sources": [c.source], "source_chunk": c.chunk_id,
                        "status": "needs_review", "generator": llm.model})
    return out


def dedupe_candidates(cands: list[dict], embedder, threshold: float = 0.92) -> list[dict]:
    """Drop near-duplicate questions (cosine >= threshold)."""
    from .embeddings import cosine

    kept, vecs = [], []
    for c, v in zip(cands, embedder.embed_documents([c["question"] for c in cands])):
        if all(cosine(v, u) < threshold for u in vecs):
            kept.append(c)
            vecs.append(v)
    return kept
