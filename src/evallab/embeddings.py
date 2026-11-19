"""Embedding back ends.

nomic-embed-text was trained with task prefixes. Documents must be embedded
as "search_document: ..." and queries as "search_query: ...". Skipping the
prefixes silently costs retrieval quality, which then shows up later as low
context recall. We apply them here so no caller can forget.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence

from .config import settings


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    def __init__(self, model: str | None = None, host: str | None = None, batch_size: int = 32):
        import ollama

        self.model = model or settings.embed_model
        self.batch_size = batch_size
        self._client = ollama.Client(host=host or settings.ollama_host)
        self._prefixed = self.model.startswith("nomic-embed-text")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self._client.embed(model=self.model, input=texts[i:i + self.batch_size])
            vectors.extend(resp.embeddings)
        return vectors

    def embed_documents(self, texts):
        texts = [f"search_document: {t}" if self._prefixed else t for t in texts]
        return self._embed(list(texts))

    def embed_query(self, text):
        return self._embed([f"search_query: {text}" if self._prefixed else text])[0]


class HashingEmbedder:
    """Bag-of-words hashing embedder: no model, no network. Used in tests and CI.

    It is a lexical baseline, not a semantic model, but it is deterministic,
    which is exactly what a unit test needs.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in text.lower().split():
            tok = "".join(ch for ch in tok if ch.isalnum() or ch in ".%$")
            if not tok:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
