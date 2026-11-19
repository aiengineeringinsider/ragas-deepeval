"""FastAPI backend for the Local LLM Evaluation Platform (Module 14).

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from evallab import store  # noqa: E402
from evallab.config import settings  # noqa: E402
from evallab.custom_metrics import deterministic_scores  # noqa: E402
from evallab.records import EvalRecord  # noqa: E402

app = FastAPI(title="Local LLM Evaluation Platform", version="1.0.0")


# ---------------------------------------------------------------- dependencies
@lru_cache(maxsize=1)
def get_pipeline():
    from evallab.rag import build_default_pipeline

    return build_default_pipeline()


# --------------------------------------------------------------------- schemas
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieval_scores: list[float]
    latency_s: float
    model: str
    checks: dict[str, float]


class EvaluateRequest(BaseModel):
    label: str = "api"
    generator: str = settings.generator_model
    judge: str = settings.judge_model
    ragas_metrics: list[str] = ["faithfulness", "answer_relevancy", "context_recall"]
    use_deepeval: bool = False
    limit: int = Field(default=0, ge=0)


class HumanLabel(BaseModel):
    run_id: str
    item_id: str
    annotator: str = Field(min_length=1)
    faithful: int = Field(ge=0, le=1)
    score: int = Field(ge=1, le=5)
    note: str = ""


# ------------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    import ollama

    try:
        models = sorted(m.model for m in ollama.Client(host=settings.ollama_host).list().models)
        return {"status": "ok", "ollama": settings.ollama_host, "models": models}
    except Exception as e:  # report, do not crash: the dashboard still works offline
        return {"status": "degraded", "ollama": settings.ollama_host, "error": str(e)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline=Depends(get_pipeline)):
    r = pipeline.answer(req.question)
    rec = EvalRecord(id="adhoc", question=req.question, reference="", category="adhoc",
                     answer=r.answer, contexts=r.contexts, sources=r.sources,
                     reference_sources=[], latency_s=r.total_latency_s, model=r.model)
    checks = {k: v for k, v in deterministic_scores(rec).items()
              if k in {"numeric_grounding", "citation_validity"}}
    return AskResponse(answer=r.answer, sources=r.sources, retrieval_scores=r.retrieval_scores,
                       latency_s=round(r.total_latency_s, 3), model=r.model, checks=checks)


JOBS: dict[str, dict] = {}


def _run_job(job_id: str, req: EvaluateRequest):
    from evallab.pipeline import RunConfig, run_full_evaluation

    JOBS[job_id]["status"] = "running"
    try:
        run_id, summary = run_full_evaluation(RunConfig(
            label=req.label, generator=req.generator, judge=req.judge,
            ragas_metrics=tuple(req.ragas_metrics), use_deepeval=req.use_deepeval, limit=req.limit))
        JOBS[job_id].update(status="done", run_id=run_id, summary=summary)
    except Exception as e:
        JOBS[job_id].update(status="failed", error=f"{type(e).__name__}: {e}")


@app.post("/evaluate", status_code=202)
def evaluate(req: EvaluateRequest, background: BackgroundTasks):
    """Evaluation takes minutes on a laptop, so it runs in the background.
    Poll GET /jobs/{job_id}."""
    import uuid

    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"status": "queued", "request": req.model_dump()}
    background.add_task(_run_job, job_id, req)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]


@app.get("/runs")
def runs():
    return store.list_runs()


@app.get("/runs/{run_id}")
def run_detail(run_id: str):
    rows = store.run_results(run_id)
    if not rows:
        raise HTTPException(404, "unknown run")
    return rows


@app.post("/human-labels", status_code=201)
def add_label(label: HumanLabel):
    return {"label_id": store.add_human_label(**label.model_dump())}


@app.get("/human-labels")
def get_labels(run_id: str | None = None):
    return store.human_labels(run_id)
