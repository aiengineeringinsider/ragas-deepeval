"""SQLite results store shared by the CLI scripts, the API, and the dashboard.

SQLite is the right tool here: zero setup, one file you can commit or attach
to a CI artifact, and plenty fast for thousands of evaluation rows.
"""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .records import EvalRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT, label TEXT, generator TEXT,
    judge TEXT, git_sha TEXT, config_json TEXT, summary_json TEXT);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT, item_id TEXT, category TEXT, question TEXT, answer TEXT,
    reference TEXT, contexts_json TEXT, sources_json TEXT, latency_s REAL,
    metric TEXT, score REAL, reason TEXT,
    PRIMARY KEY (run_id, item_id, metric));
CREATE TABLE IF NOT EXISTS human_labels (
    label_id TEXT PRIMARY KEY, run_id TEXT, item_id TEXT, annotator TEXT,
    faithful INTEGER, score INTEGER, note TEXT, created_at TEXT);
"""


@contextmanager
def connect(db_path: Path | None = None):
    p = db_path or settings.db_path
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def save_run(records: list[EvalRecord], label: str, generator: str, judge: str,
             summary: dict, config: dict | None = None, git_sha: str = "",
             db_path: Path | None = None) -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    with connect(db_path) as con:
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                    (run_id, datetime.now(timezone.utc).isoformat(), label, generator, judge,
                     git_sha, json.dumps(config or {}), json.dumps(summary, default=str)))
        for r in records:
            for metric, score in r.scores.items():
                if metric.startswith("_"):
                    continue
                con.execute("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (run_id, r.id, r.category, r.question, r.answer, r.reference,
                             json.dumps(r.contexts), json.dumps(r.sources), r.latency_s,
                             metric, _clean(score), r.reasons.get(metric)))
    return run_id


def list_runs(db_path: Path | None = None) -> list[dict]:
    with connect(db_path) as con:
        return [dict(r) for r in con.execute("SELECT * FROM runs ORDER BY created_at DESC")]


def run_results(run_id: str, db_path: Path | None = None) -> list[dict]:
    with connect(db_path) as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM results WHERE run_id=? ORDER BY item_id, metric", (run_id,))]


def add_human_label(run_id: str, item_id: str, annotator: str, faithful: int, score: int,
                    note: str = "", db_path: Path | None = None) -> str:
    label_id = uuid.uuid4().hex
    with connect(db_path) as con:
        con.execute("INSERT INTO human_labels VALUES (?,?,?,?,?,?,?,?)",
                    (label_id, run_id, item_id, annotator, int(faithful), int(score), note,
                     datetime.now(timezone.utc).isoformat()))
    return label_id


def human_labels(run_id: str | None = None, db_path: Path | None = None) -> list[dict]:
    with connect(db_path) as con:
        q, args = "SELECT * FROM human_labels", ()
        if run_id:
            q, args = q + " WHERE run_id=?", (run_id,)
        return [dict(r) for r in con.execute(q, args)]
