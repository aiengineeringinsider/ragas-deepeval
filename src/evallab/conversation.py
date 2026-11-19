"""Multi-turn evaluation (Module 17).

A single-turn RAG system answers each message in isolation, so "we are on
Business" in turn 1 is forgotten by turn 3. ConversationalRAG rewrites each new
user message into a standalone question using the conversation so far, then
calls the ordinary pipeline. Evaluation checks what single-turn tests cannot:
whether facts the user stated earlier are still applied later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings

REWRITE_PROMPT = """Rewrite the LAST user message as one standalone question that keeps every
fact the user stated earlier in the conversation (plan, amounts, dates, region).
Return only the question.

CONVERSATION:
{history}

LAST USER MESSAGE: {message}"""


@dataclass
class Conversation:
    id: str
    turns: list[dict] = field(default_factory=list)   # [{"role": "user"|"assistant", "content"}]
    facts: dict = field(default_factory=dict)          # what the user stated, e.g. {"plan": "business"}
    expectations: list[dict] = field(default_factory=list)  # [{"turn": i, "must_contain": "..."}]


class ConversationalRAG:
    def __init__(self, pipeline, llm):
        self.pipeline, self.llm = pipeline, llm

    def standalone(self, history: list[dict], message: str) -> str:
        if not history:
            return message
        h = "\n".join(f"{t['role']}: {t['content']}" for t in history)
        r = self.llm.chat([{"role": "user", "content": REWRITE_PROMPT.format(history=h, message=message)}],
                          temperature=0.0)
        return r.text.strip().strip('"') or message

    def reply(self, history: list[dict], message: str) -> tuple[str, list[str]]:
        q = self.standalone(history, message)
        res = self.pipeline.answer(q)
        return res.answer, res.contexts

    def run(self, user_messages: list[str], conv_id: str = "c") -> Conversation:
        conv = Conversation(conv_id)
        for msg in user_messages:
            answer, contexts = self.reply(conv.turns, msg)
            conv.turns.append({"role": "user", "content": msg})
            conv.turns.append({"role": "assistant", "content": answer, "retrieval_context": contexts})
        return conv


def load_conversations(path: Path | None = None) -> list[dict]:
    p = path or (settings.golden_path.parent / "conversations_v1.jsonl")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def helios_conversations(spec: dict, conv: Conversation):
    """DeepEval ConversationalTestCase for one scripted conversation."""
    from deepeval.test_case import ConversationalTestCase, Turn

    return ConversationalTestCase(
        turns=[Turn(role=t["role"], content=t["content"],
                    retrieval_context=t.get("retrieval_context") or None) for t in conv.turns],
        scenario=spec.get("scenario"), expected_outcome=spec.get("expected_outcome"),
        chatbot_role="Helios Cloud Storage support assistant that answers only from documentation",
        name=spec["id"],
    )


def retention_check(conv: Conversation, expectations: list[dict]) -> dict:
    """Deterministic: does each expected assistant turn contain the required fact?

    expectations: [{"assistant_turn": 2, "must_contain": "$250", "must_not_contain": "$0.021"}]
    assistant_turn counts assistant messages from 1.
    """
    answers = [t["content"] for t in conv.turns if t["role"] == "assistant"]
    results = []
    for e in expectations:
        i = e["assistant_turn"] - 1
        text = answers[i] if 0 <= i < len(answers) else ""
        ok = (e.get("must_contain", "") in text) and not (
            e.get("must_not_contain") and e["must_not_contain"] in text)
        results.append({"assistant_turn": e["assistant_turn"], "passed": ok, "answer": text[:120]})
    passed = sum(r["passed"] for r in results)
    return {"id": conv.id, "passed": passed, "total": len(results), "details": results}


def conversation_metrics(judge):
    """LLM-judged multi-turn metrics, each with an explicit definition."""
    from deepeval.metrics import ConversationalGEval, KnowledgeRetentionMetric, TurnRelevancyMetric
    from deepeval.test_case import MultiTurnParams

    return [
        KnowledgeRetentionMetric(model=judge, threshold=0.7, async_mode=False),
        TurnRelevancyMetric(model=judge, threshold=0.7, async_mode=False),
        ConversationalGEval(
            name="Applies Stated Plan",
            evaluation_steps=[
                "Find the plan the user states (Starter, Business, or Enterprise).",
                "For every later assistant turn that gives a price, credit, or limit, check it uses "
                "that plan's figures.",
                "Fail if any such turn uses another plan's figures or asks the user to repeat the plan.",
            ],
            evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
            model=judge, threshold=0.7, async_mode=False),
    ]


