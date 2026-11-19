import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _ollama_up() -> bool:
    try:
        import ollama

        from evallab.config import settings

        ollama.Client(host=settings.ollama_host, timeout=3).list()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _ollama_up():
        return
    skip = pytest.mark.skip(reason="Ollama is not reachable; eval tests need a live model")
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test.sqlite"


@pytest.fixture
def offline_pipeline():
    """A full RAG pipeline with a hashing embedder, in-memory Chroma, and a fake LLM."""
    from evallab.chunking import load_corpus
    from evallab.config import settings
    from evallab.embeddings import HashingEmbedder
    from evallab.llm import FakeLLM
    from evallab.rag import RAGPipeline
    from evallab.vectorstore import ChromaStore

    store = ChromaStore(HashingEmbedder(), in_memory=True, collection="test_helios")
    store.reset()
    store.add(load_corpus(settings.corpus_dir))
    return RAGPipeline(store, FakeLLM(default="Starter costs $0.021 per GB-month [1]."))
