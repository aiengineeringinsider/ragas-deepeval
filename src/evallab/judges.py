"""Hand-built LLM judges for the judge-comparison and bias experiments.

RAGAS and DeepEval chain several LLM calls per metric (extract claims, then
verify each). That is good for quality, but it makes it hard to isolate the
*judge model* as the variable. For Modules 7, 9, and 10 we therefore use
single-call judges with a fixed prompt and a JSON schema, so the only thing
changing between runs is the model (or one controlled prompt factor).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .llm import ChatModel, parse_json

FAITHFULNESS_PROMPT = """You are a strict fact-checker.
Decide whether EVERY factual claim in the ANSWER is supported by the CONTEXT.
- If any claim (a number, plan name, duration, permission, or promise) is missing
  from or contradicted by the CONTEXT, the answer is NOT faithful.
- Saying the information is unavailable is faithful if the CONTEXT indeed lacks it.
- Ignore style. Judge only support.

CONTEXT:
{context}

QUESTION: {question}
ANSWER: {answer}

Return JSON: {{"reason": "<one sentence>", "faithful": true|false}}"""

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "faithful": {"type": "boolean"}},
    "required": ["reason", "faithful"],
}

SCORE_PROMPT = """Rate the ANSWER to the QUESTION on a 1-5 scale using only the CONTEXT.
5 = fully correct, complete, and supported; 4 = correct but missing a minor detail;
3 = partly correct; 2 = mostly wrong or a key fact missing; 1 = wrong or fabricated.

CONTEXT:
{context}

QUESTION: {question}
ANSWER: {answer}

Return JSON: {{"reason": "<one sentence>", "score": <integer 1-5>}}"""

SCORE_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"},
                   "score": {"type": "integer", "minimum": 1, "maximum": 5}},
    "required": ["reason", "score"],
}

PAIRWISE_PROMPT = """Two assistants answered the same QUESTION using the CONTEXT.
Pick the answer that is more correct and better supported by the CONTEXT.

CONTEXT:
{context}

QUESTION: {question}

ANSWER A: {a}

ANSWER B: {b}

Return JSON: {{"reason": "<one sentence>", "winner": "A" | "B"}}"""

PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "winner": {"type": "string", "enum": ["A", "B"]}},
    "required": ["reason", "winner"],
}


RELEVANCE_PROMPT = """You check whether an ANSWER responds to the QUESTION that was asked.
- Judge only whether it addresses what was asked. Do NOT judge whether it is correct;
  a separate faithfulness check does that.
- Citation markers like [1], shown arithmetic, and a short restatement of the
  question are fine.
- An answer about a different plan, product, number, or topic than the one asked
  does NOT address the question.
- "I don't have that information" does NOT address an answerable question.

QUESTION: {question}
ANSWER: {answer}

Return JSON: {{"reason": "<one sentence>", "addresses_question": true|false}}"""

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "addresses_question": {"type": "boolean"}},
    "required": ["reason", "addresses_question"],
}


@dataclass
class Verdict:
    label: int | None          # 1 faithful, 0 not, None = unparseable
    score: int | None
    reason: str
    latency_s: float
    raw: str


def judge_faithfulness(llm: ChatModel, question: str, context: str, answer: str,
                       prompt: str = FAITHFULNESS_PROMPT, seed: int | None = 42,
                       temperature: float = 0.0) -> Verdict:
    msg = prompt.format(context=context, question=question, answer=answer)
    r = llm.chat([{"role": "user", "content": msg}], json_schema=FAITHFULNESS_SCHEMA,
                 temperature=temperature, seed=seed)
    try:
        d = parse_json(r.text)
        return Verdict(int(bool(d["faithful"])), None, str(d.get("reason", "")), r.latency_s, r.text)
    except (ValueError, KeyError, json.JSONDecodeError):
        return Verdict(None, None, "unparseable", r.latency_s, r.text)


def judge_relevance(llm: ChatModel, question: str, answer: str,
                    seed: int | None = 42) -> Verdict:
    """Single-call, binary relevance judge (one decision, not a statement split)."""
    msg = RELEVANCE_PROMPT.format(question=question, answer=answer)
    r = llm.chat([{"role": "user", "content": msg}], json_schema=RELEVANCE_SCHEMA,
                 temperature=0.0, seed=seed)
    try:
        d = parse_json(r.text)
        return Verdict(int(bool(d["addresses_question"])), None, str(d.get("reason", "")),
                       r.latency_s, r.text)
    except (ValueError, KeyError, json.JSONDecodeError):
        return Verdict(None, None, "unparseable", r.latency_s, r.text)


def judge_score(llm: ChatModel, question: str, context: str, answer: str,
                seed: int | None = 42) -> Verdict:
    msg = SCORE_PROMPT.format(context=context, question=question, answer=answer)
    r = llm.chat([{"role": "user", "content": msg}], json_schema=SCORE_SCHEMA, seed=seed)
    try:
        d = parse_json(r.text)
        s = max(1, min(5, int(d["score"])))
        return Verdict(None, s, str(d.get("reason", "")), r.latency_s, r.text)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return Verdict(None, None, "unparseable", r.latency_s, r.text)


def judge_pairwise(llm: ChatModel, question: str, context: str, a: str, b: str,
                   seed: int | None = 42) -> tuple[str | None, float]:
    msg = PAIRWISE_PROMPT.format(context=context, question=question, a=a, b=b)
    r = llm.chat([{"role": "user", "content": msg}], json_schema=PAIRWISE_SCHEMA, seed=seed)
    try:
        w = parse_json(r.text)["winner"].strip().upper()
        return (w if w in {"A", "B"} else None), r.latency_s
    except (ValueError, KeyError, json.JSONDecodeError):
        return None, r.latency_s


def load_calibration(path: Path, corpus_dir: Path) -> list[dict]:
    """Load human-labeled cases and attach the full text of their context documents."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["context"] = "\n\n".join((corpus_dir / s).read_text(encoding="utf-8")
                                     for s in row["context_sources"])
        rows.append(row)
    return rows
