"""BM25 keyword retrieval and reciprocal-rank fusion (hybrid search).

Why this exists: dense embeddings blur near-duplicate passages. In the lab's
smoke test, the question "A Business customer stores 10,000 GB ... how much
will they pay?" retrieved the Starter, Enterprise, discount and SLA chunks but
NOT the Business pricing chunk, so the generator priced the wrong plan. The
word "Business" is exactly what BM25 is good at. Fusing the two rankings with
reciprocal-rank fusion (RRF) keeps semantic matches and restores exact ones.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOP = set("a an the of for to in on is are be do does how what which who can i we you "
            "they it its and or with at by per much will".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 over a small in-memory corpus (fine for thousands of chunks)."""

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75):
        self.docs = [tokenize(t) for t in texts]
        self.tf = [Counter(d) for d in self.docs]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * len(d) / (self.avgdl or 1))
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    def rank(self, query: str, k: int) -> list[int]:
        s = self.scores(query)
        order = sorted(range(self.n), key=lambda i: s[i], reverse=True)
        return [i for i in order[:k] if s[i] > 0]


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Reciprocal-rank fusion: score(id) = sum over rankings of 1 / (k + rank)."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, item in enumerate(ranking, start=1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + r)
    return sorted(score, key=lambda i: score[i], reverse=True)
