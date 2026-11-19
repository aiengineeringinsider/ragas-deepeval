"""The system under test: a small but complete RAG pipeline.

answer() returns a RAGResult carrying everything an evaluator needs
(question, answer, retrieved contexts, sources, latency) so evaluation never
has to re-run retrieval and can never evaluate different contexts than the
ones the generator actually saw.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .config import settings
from .lexical import BM25, rrf
from .llm import ChatModel
from .vectorstore import ChromaStore, RetrievedChunk

SYSTEM_PROMPT = """You are the Helios Cloud Storage support assistant.
Answer ONLY from the numbered context passages. Rules:
1. If the context does not contain the answer, reply exactly:
   "I don't have that information in the Helios documentation."
   Do not add citations to that sentence.
2. Use the figures for the plan, region, or severity the question names. Never answer
   with another plan's numbers; if the named plan's figures are missing, use rule 1.
3. Cite the passages you used with bracketed numbers, e.g. [1] or [2][3].
4. Show the arithmetic for any price, credit, or date calculation, and apply every
   rule in the passages that changes the result (minimums, caps, exclusions).
5. Be concise: at most 4 sentences."""

REFUSAL = "I don't have that information in the Helios documentation."


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[str]
    sources: list[str]
    retrieval_scores: list[float]
    model: str
    retrieval_latency_s: float
    generation_latency_s: float
    metadata: dict = field(default_factory=dict)

    @property
    def total_latency_s(self) -> float:
        return self.retrieval_latency_s + self.generation_latency_s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_latency_s"] = self.total_latency_s
        return d


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))


class RAGPipeline:
    def __init__(self, store: ChromaStore, llm: ChatModel, top_k: int | None = None,
                 min_score: float = 0.0, mode: str | None = None):
        self.store = store
        self.llm = llm
        self.top_k = top_k or settings.top_k
        self.min_score = min_score  # drop very weak matches before generation
        self.mode = mode or settings.retrieval_mode
        self._bm25: BM25 | None = None
        self._chunks: list[RetrievedChunk] = []

    def _lexical_index(self) -> tuple[BM25, list[RetrievedChunk]]:
        # Built lazily and once: the corpus only changes when you re-ingest.
        if self._bm25 is None or len(self._chunks) != self.store.count():
            self._chunks = self.store.all_chunks()
            self._bm25 = BM25([c.text for c in self._chunks])
        return self._bm25, self._chunks

    def retrieve(self, question: str) -> tuple[list[RetrievedChunk], float]:
        t0 = time.perf_counter()
        if self.mode == "dense":
            hits = [h for h in self.store.search(question, self.top_k)
                    if h.score >= self.min_score]
            return hits, time.perf_counter() - t0
        # Hybrid: a wider dense candidate list, a BM25 list, fused by reciprocal rank.
        dense = [h for h in self.store.search(question, self.top_k * 3)
                 if h.score >= self.min_score]
        bm25, chunks = self._lexical_index()
        lexical = [chunks[i] for i in bm25.rank(question, self.top_k * 3)]
        by_id = {h.chunk_id: h for h in lexical}
        by_id.update({h.chunk_id: h for h in dense})  # keep dense cosine scores when present
        fused = rrf([[h.chunk_id for h in dense], [h.chunk_id for h in lexical]])
        return [by_id[i] for i in fused[: self.top_k]], time.perf_counter() - t0

    def answer(self, question: str) -> RAGResult:
        hits, r_lat = self.retrieve(question)
        if not hits:
            return RAGResult(question, REFUSAL, [], [], [], self.llm.model, r_lat, 0.0,
                             {"refused": True, "reason": "no_context"})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{format_context(hits)}\n\nQuestion: {question}"},
        ]
        resp = self.llm.chat(messages, temperature=0.0)
        answer = resp.text.strip()
        return RAGResult(
            question=question,
            answer=answer,
            contexts=[h.text for h in hits],
            sources=[h.source for h in hits],
            retrieval_scores=[round(h.score, 4) for h in hits],
            model=self.llm.model,
            retrieval_latency_s=r_lat,
            generation_latency_s=resp.latency_s,
            metadata={"refused": answer.startswith("I don't have that information")},
        )


def build_default_pipeline(in_memory: bool = False) -> RAGPipeline:
    """Wire the real components: Ollama embeddings + Chroma + Ollama generator."""
    from .embeddings import OllamaEmbedder
    from .llm import OllamaChat

    store = ChromaStore(OllamaEmbedder(), in_memory=in_memory)
    return RAGPipeline(store, OllamaChat(settings.generator_model))
