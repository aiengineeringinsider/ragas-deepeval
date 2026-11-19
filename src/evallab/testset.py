"""Synthetic test sets with the RAGAS TestsetGenerator (Module 15).

RAGAS builds a knowledge graph from the documents (headlines, summaries,
entities, embeddings), then synthesizes single-hop and multi-hop questions
with references. Generated items are candidates, never goldens: review_queue
removes near-duplicates of existing goldens and flags items that cannot
discriminate, and a person promotes the survivors.
"""
from __future__ import annotations

import re

from .config import settings
from .lexical import tokenize


def build_generator(model: str | None = None, embed_model: str | None = None):
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator

    llm = LangchainLLMWrapper(ChatOllama(model=model or settings.judge_model,
                                         base_url=settings.ollama_host, temperature=0.0,
                                         num_ctx=settings.num_ctx))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=embed_model or settings.embed_model,
                                                      base_url=settings.ollama_host))
    return TestsetGenerator(llm=llm, embedding_model=emb), llm


def load_documents():
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    return DirectoryLoader(str(settings.corpus_dir), glob="*.md", loader_cls=TextLoader).load()


def generate_testset(testset_size: int = 12, model: str | None = None):
    """Returns a pandas DataFrame: user_input, reference_contexts, reference, synthesizer_name."""
    from ragas.run_config import RunConfig
    from ragas.testset.synthesizers import default_query_distribution

    generator, llm = build_generator(model)
    testset = generator.generate_with_langchain_docs(
        load_documents(), testset_size=testset_size,
        query_distribution=default_query_distribution(llm),
        run_config=RunConfig(timeout=600, max_workers=2, max_retries=3))
    return testset.to_pandas()


_TRAP = re.compile(r"\d|\bnot\b|\bno\b|\bonly\b|\bexcept\b|\bminimum\b|\bcannot\b", re.I)


def _jaccard(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / len(x | y) if x | y else 0.0


def review_queue(rows: list[dict], goldens: list, dup_threshold: float = 0.6) -> list[dict]:
    """Deterministic triage of generated candidates before a human sees them.

    Each row needs user_input, reference, and synthesizer_name. Adds flags:
      duplicate_of: id of an existing golden with Jaccard >= dup_threshold
      has_trap:     reference contains a number, negation, limit, or exception
      multi_source: more than one reference context (a multi-hop candidate)
    """
    out = []
    for r in rows:
        q = str(r.get("user_input", ""))
        best = max(goldens, key=lambda g: _jaccard(q, g.question), default=None)
        dup = best.id if best is not None and _jaccard(q, best.question) >= dup_threshold else None
        out.append({
            "question": q, "reference": str(r.get("reference", "")),
            "synthesizer": str(r.get("synthesizer_name", "")),
            "duplicate_of": dup,
            "has_trap": bool(_TRAP.search(str(r.get("reference", "")))),
            "multi_source": len(r.get("reference_contexts") or []) > 1,
            "status": "rejected_duplicate" if dup else "needs_review",
        })
    return out
