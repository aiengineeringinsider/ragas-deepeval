"""Module 3: ask the RAG assistant a question from the terminal.

Run:  python scripts/03_ask.py "How much does Starter cost per GB-month?"
"""
import sys

sys.path.insert(0, "src")
from evallab.rag import build_default_pipeline  # noqa: E402


def main():
    question = " ".join(sys.argv[1:]) or "What happens 15 days after an unpaid invoice is due?"
    rag = build_default_pipeline()
    if rag.store.count() == 0:
        raise SystemExit("Collection is empty. Run: python scripts/02_ingest.py")
    r = rag.answer(question)
    print(f"Q: {r.question}\n\nA: {r.answer}\n")
    for i, (src, s) in enumerate(zip(r.sources, r.retrieval_scores), start=1):
        print(f"  [{i}] {s:.3f} {src}")
    print(f"\nretrieval {r.retrieval_latency_s * 1000:.0f} ms | "
          f"generation {r.generation_latency_s:.1f} s | model {r.model}")


if __name__ == "__main__":
    main()
