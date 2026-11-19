"""Central configuration for the lab.

Every setting can be overridden with an environment variable (or a .env file),
so the same code runs on a laptop, in CI, and inside Docker without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # Ollama
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    generator_model: str = _env("EVALLAB_GENERATOR", "qwen2.5:3b")
    judge_model: str = _env("EVALLAB_JUDGE", "qwen2.5:3b")
    judge_models: tuple[str, ...] = tuple(
        _env("EVALLAB_JUDGES", "qwen2.5:3b,qwen3:1.7b,llama3.2:1b").split(",")
    )
    vision_model: str = _env("EVALLAB_VISION", "qwen2.5vl:3b")
    embed_model: str = _env("EVALLAB_EMBED", "nomic-embed-text:latest")
    num_ctx: int = int(_env("EVALLAB_NUM_CTX", "8192"))
    request_timeout_s: float = float(_env("EVALLAB_TIMEOUT", "300"))

    # Retrieval
    chroma_dir: Path = Path(_env("EVALLAB_CHROMA_DIR", str(PROJECT_ROOT / ".chroma")))
    collection: str = _env("EVALLAB_COLLECTION", "helios_docs")
    chunk_size: int = int(_env("EVALLAB_CHUNK_SIZE", "700"))
    chunk_overlap: int = int(_env("EVALLAB_CHUNK_OVERLAP", "100"))
    top_k: int = int(_env("EVALLAB_TOP_K", "4"))

    # Data and results
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    golden_path: Path = PROJECT_ROOT / "data" / "golden" / "golden_v1.jsonl"
    calibration_path: Path = PROJECT_ROOT / "data" / "human_labels" / "calibration_v1.jsonl"
    image_dir: Path = PROJECT_ROOT / "data" / "images"
    results_dir: Path = Path(_env("EVALLAB_RESULTS_DIR", str(PROJECT_ROOT / "results")))
    db_path: Path = Path(_env("EVALLAB_DB", str(PROJECT_ROOT / "results" / "evallab.sqlite")))
    baseline_path: Path = PROJECT_ROOT / "baselines" / "baseline.json"

    @property
    def openai_compat_url(self) -> str:
        """Ollama exposes an OpenAI-compatible API under /v1 (used by RAGAS)."""
        return self.ollama_host.rstrip("/") + "/v1"


settings = Settings()
