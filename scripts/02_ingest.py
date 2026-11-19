"""Module 3: chunk the Helios corpus, embed with nomic-embed-text, store in ChromaDB.

Run:  python scripts/02_ingest.py [--reset]
"""
import argparse
import sys
import time
from collections import Counter

sys.path.insert(0, "src")
from evallab.chunking import load_corpus  # noqa: E402
from evallab.config import settings  # noqa: E402
from evallab.embeddings import OllamaEmbedder  # noqa: E402
from evallab.vectorstore import ChromaStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop and rebuild the collection")
    args = ap.parse_args()

    chunks = load_corpus(settings.corpus_dir, settings.chunk_size, settings.chunk_overlap)
    per_doc = Counter(c.source for c in chunks)
    lengths = [len(c.text) for c in chunks]
    print(f"Loaded {len(per_doc)} documents -> {len(chunks)} chunks "
          f"(chars min/mean/max = {min(lengths)}/{sum(lengths) // len(lengths)}/{max(lengths)})")

    store = ChromaStore(OllamaEmbedder())
    if args.reset:
        store.reset()
    t0 = time.perf_counter()
    n = store.add(chunks)
    print(f"Embedded and upserted {n} chunks in {time.perf_counter() - t0:.1f}s; "
          f"collection '{store.name}' now has {store.count()} vectors at {settings.chroma_dir}")

    for q in ["How long is soft delete on Business?", "What is the maximum object size?"]:
        print(f"\nQ: {q}")
        for h in store.search(q, k=3):
            print(f"  {h.score:.3f}  {h.source:<36} {h.section}")


if __name__ == "__main__":
    main()
