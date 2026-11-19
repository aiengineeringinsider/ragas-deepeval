"""End-to-end evaluation run used by the API, the CLI, and CI.

generate (RAG over golden) -> deterministic metrics -> RAGAS subset ->
optional DeepEval subset -> summary -> SQLite.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

from .config import settings
from .custom_metrics import deterministic_scores
from .golden import dataset_card, load_golden
from .records import EvalRecord, run_system, save_records
from .store import save_run


@dataclass
class RunConfig:
    label: str = "manual"
    generator: str = settings.generator_model
    judge: str = settings.judge_model
    ragas_metrics: tuple[str, ...] = ("faithfulness", "answer_relevancy", "context_recall")
    use_deepeval: bool = False
    limit: int = 0
    categories: tuple[str, ...] = ()


def summarize_scores(records: list[EvalRecord]) -> dict:
    names = sorted({k for r in records for k in r.scores if not k.startswith("_")})
    out = {}
    for n in names:
        vals = [r.scores[n] for r in records if n in r.scores and not math.isnan(r.scores[n])]
        out[n] = round(sum(vals) / len(vals), 4) if vals else None
    out["p50_latency_s"] = sorted(r.latency_s for r in records)[len(records) // 2] if records else None
    return out


def run_full_evaluation(cfg: RunConfig, pipeline=None, ragas_llm=None, ragas_emb=None) -> tuple[str, dict]:
    items = load_golden(settings.golden_path)
    if cfg.categories:
        items = [i for i in items if i.category in cfg.categories]
    if cfg.limit:
        items = items[: cfg.limit]

    if pipeline is None:
        from .embeddings import OllamaEmbedder
        from .llm import OllamaChat
        from .rag import RAGPipeline
        from .vectorstore import ChromaStore

        pipeline = RAGPipeline(ChromaStore(OllamaEmbedder()), OllamaChat(cfg.generator))

    records = run_system(pipeline, items, progress=False)
    for r in records:
        r.scores.update(deterministic_scores(r))

    if cfg.ragas_metrics:
        from .ragas_eval import (default_metric_specs, evaluate_records, make_ragas_embeddings,
                                 make_ragas_llm)

        specs = default_metric_specs(ragas_llm or make_ragas_llm(cfg.judge),
                                     ragas_emb or make_ragas_embeddings())
        specs = [s for s in specs if s.name in cfg.ragas_metrics]
        asyncio.run(evaluate_records(records, specs))

    if cfg.use_deepeval:
        from .deepeval_eval import default_metrics, make_judge, score_record

        metrics = default_metrics(make_judge(cfg.judge), include_geval=True)
        for r in records:
            score_record(r, metrics)

    summary = {"dataset": dataset_card(items), "scores": summarize_scores(records)}
    run_id = save_run(records, cfg.label, cfg.generator, cfg.judge, summary,
                      config=cfg.__dict__ | {"top_k": settings.top_k,
                                             "chunk_size": settings.chunk_size})
    out = settings.results_dir / f"run_{run_id}.jsonl"
    save_records(records, out)  # frozen, scored records for the regression gate
    summary["records_path"] = str(out)
    return run_id, summary
