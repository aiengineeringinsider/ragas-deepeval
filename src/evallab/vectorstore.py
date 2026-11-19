"""ChromaDB persistence and retrieval.

We compute embeddings ourselves and hand Chroma the vectors. That keeps the
embedding model explicit and identical at index time and query time, which is
the single most common source of silent retrieval bugs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk
from .config import settings
from .embeddings import Embedder


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    section: str
    score: float  # cosine similarity in [-1, 1]; higher is better


class ChromaStore:
    def __init__(self, embedder: Embedder, path: Path | None = None,
                 collection: str | None = None, in_memory: bool = False):
        import chromadb

        self.embedder = embedder
        if in_memory:
            self._client = chromadb.EphemeralClient()
        else:
            p = path or settings.chroma_dir
            p.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(p))
        self.name = collection or settings.collection
        self._col = self._client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        self._client.delete_collection(self.name)
        self._col = self._client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self._col.count()

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    def all_chunks(self) -> list[RetrievedChunk]:
        """Every stored chunk (score 0.0). Used to build the BM25 index for hybrid search."""
        res = self._col.get(include=["documents", "metadatas"])
        return [RetrievedChunk(cid, doc, (meta or {}).get("source", ""),
                               (meta or {}).get("section", ""), 0.0)
                for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or settings.top_k
        res = self._col.query(query_embeddings=[self.embedder.embed_query(query)],
                              n_results=min(k, max(self.count(), 1)),
                              include=["documents", "metadatas", "distances"])
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            out.append(RetrievedChunk(cid, doc, meta.get("source", ""),
                                      meta.get("section", ""), 1.0 - float(dist)))
        return out
