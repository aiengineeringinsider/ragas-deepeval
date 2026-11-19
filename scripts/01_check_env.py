"""Module 2: verify Python packages, the Ollama server, and every required model.

Run:  python scripts/01_check_env.py
Exit code 0 means the lab is ready; 1 means something is missing (and says what).
"""
import importlib
import sys
import time

sys.path.insert(0, "src")
from evallab.config import settings  # noqa: E402

PACKAGES = ["ollama", "chromadb", "ragas", "deepeval", "fastapi", "streamlit", "pytest",
            "sklearn", "scipy", "pandas"]
REQUIRED_MODELS = ["qwen2.5:3b", "qwen2.5vl:3b", "qwen3:1.7b", "llama3.2:1b",
                   "nomic-embed-text:latest"]


def main() -> int:
    ok = True
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  !! Python 3.10+ is required (3.11 or 3.12 recommended)")
        ok = False
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  [ok] {name:<10} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"  [missing] {name}  ->  pip install -r requirements.txt")
            ok = False

    import ollama

    client = ollama.Client(host=settings.ollama_host)
    try:
        installed = {m.model for m in client.list().models}
    except Exception as e:
        print(f"\nCannot reach Ollama at {settings.ollama_host}: {e}")
        print("Start it with `ollama serve` (or open the Ollama app) and retry.")
        return 1
    print(f"\nOllama at {settings.ollama_host}: {len(installed)} models installed")
    for m in REQUIRED_MODELS:
        present = m in installed or f"{m}:latest" in installed
        print(f"  [{'ok' if present else 'missing'}] {m}" + ("" if present else f"  ->  ollama pull {m}"))
        ok &= present

    if ok:
        t0 = time.perf_counter()
        r = client.chat(model=settings.generator_model,
                        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
                        options={"temperature": 0})
        print(f"\nSmoke test {settings.generator_model}: {r.message.content.strip()!r} "
              f"in {time.perf_counter() - t0:.1f}s")
        e = client.embed(model=settings.embed_model, input=["search_query: hello"])
        print(f"Embedding dim ({settings.embed_model}): {len(e.embeddings[0])}")
    print("\nREADY" if ok else "\nNOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
