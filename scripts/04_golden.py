"""Module 4: validate the golden set, print its card, run the system over it,
and (optionally) draft new synthetic candidates for human review.

Run:
  python scripts/04_golden.py validate
  python scripts/04_golden.py run     --out results/responses_qwen25.jsonl
  python scripts/04_golden.py synth   --n 8 --out data/golden/candidates.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
from evallab.chunking import load_corpus  # noqa: E402
from evallab.config import settings  # noqa: E402
from evallab.golden import (dataset_card, dedupe_candidates, load_golden,  # noqa: E402
                            synthesize_candidates, validate_sources)
from evallab.records import run_system, save_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "run", "synth"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    items = load_golden(settings.golden_path)
    if args.cmd == "validate":
        problems = validate_sources(items, settings.corpus_dir)
        print(json.dumps(dataset_card(items), indent=2))
        print("problems:", problems or "none")
        raise SystemExit(1 if problems else 0)

    if args.cmd == "run":
        from evallab.rag import build_default_pipeline

        out = args.out or f"results/responses_{settings.generator_model.replace(':', '_')}.jsonl"
        recs = run_system(build_default_pipeline(), items)
        save_records(recs, Path(out))
        print(f"saved {len(recs)} responses -> {out}")

    if args.cmd == "synth":
        from evallab.embeddings import OllamaEmbedder
        from evallab.llm import OllamaChat

        chunks = load_corpus(settings.corpus_dir)
        random.Random(0).shuffle(chunks)
        cands = synthesize_candidates(chunks[: args.n], OllamaChat(settings.generator_model))
        cands = dedupe_candidates(cands, OllamaEmbedder())
        out = args.out or "data/golden/candidates.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for c in cands:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"wrote {len(cands)} candidates (status=needs_review) -> {out}")


if __name__ == "__main__":
    main()
