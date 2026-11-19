"""A small tool-using Helios agent and the data structures to evaluate it (Module 18).

The agent answers billing questions by calling tools instead of reading prices
from retrieved text: a pricing calculator, an SLA-credit calculator, and the
document search from the RAG pipeline. Tools are plain functions with exact
answers, so a wrong final answer is either a wrong tool, wrong arguments, or a
wrong reading of the tool output, and the evaluation can tell which.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import settings

PRICING = {  # from data/corpus/01_pricing_plans.md
    "starter": {"per_gb": 0.021, "minimum": 0.0},
    "business": {"per_gb": 0.018, "minimum": 250.0},
}
CREDIT_BANDS = {  # from data/corpus/02_sla_and_credits.md: (lower bound, credit %)
    "business": [(99.9, 0), (99.0, 10), (95.0, 25), (0.0, 50)],
    "enterprise": [(99.99, 0), (99.9, 10), (99.0, 25), (0.0, 100)],
}


def price_quote(plan: str, gb: float) -> dict:
    """Monthly storage charge for a plan, including the minimum commitment."""
    p = PRICING.get(plan.lower())
    if p is None:
        return {"error": f"no list price for plan {plan!r}"}
    raw = round(gb * p["per_gb"], 2)
    return {"plan": plan.lower(), "gb": gb, "usage_charge": raw,
            "minimum": p["minimum"], "amount_due": max(raw, p["minimum"])}


def sla_credit(plan: str, uptime_pct: float) -> dict:
    """Service-credit percentage for a month's uptime on a plan with an SLA."""
    bands = CREDIT_BANDS.get(plan.lower())
    if bands is None:
        return {"error": f"plan {plan!r} has no uptime SLA"}
    for lower, credit in bands:
        if uptime_pct >= lower:
            return {"plan": plan.lower(), "uptime_pct": uptime_pct, "credit_pct": credit}
    return {"error": "invalid uptime"}


TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "price_quote", "description": "Monthly storage charge for a plan and stored GB.",
        "parameters": {"type": "object", "required": ["plan", "gb"], "properties": {
            "plan": {"type": "string", "enum": ["starter", "business"]},
            "gb": {"type": "number"}}}}},
    {"type": "function", "function": {
        "name": "sla_credit", "description": "Service credit percentage for a month's uptime.",
        "parameters": {"type": "object", "required": ["plan", "uptime_pct"], "properties": {
            "plan": {"type": "string", "enum": ["business", "enterprise"]},
            "uptime_pct": {"type": "number"}}}}},
    {"type": "function", "function": {
        "name": "search_docs", "description": "Search the Helios documentation for policy text.",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string"}}}}},
]

AGENT_PROMPT = """You are the Helios billing assistant. Use the tools for every price or credit
calculation; never compute them yourself. Use search_docs for policy questions.
Answer in at most three sentences and state the number the tool returned."""


@dataclass
class AgentRun:
    question: str
    answer: str
    tool_calls: list[dict] = field(default_factory=list)  # [{"name", "args", "output"}]
    steps: int = 0


class HeliosAgent:
    def __init__(self, model: str | None = None, search: Callable[[str], str] | None = None,
                 max_steps: int = 4):
        self.model = model or settings.generator_model
        self.search = search or (lambda q: "search unavailable")
        self.max_steps = max_steps
        self.impl = {"price_quote": price_quote, "sla_credit": sla_credit,
                     "search_docs": lambda query: {"passages": self.search(query)}}

    def run(self, question: str) -> AgentRun:
        import ollama

        client = ollama.Client(host=settings.ollama_host, timeout=settings.request_timeout_s)
        messages: list[dict[str, Any]] = [{"role": "system", "content": AGENT_PROMPT},
                                          {"role": "user", "content": question}]
        run = AgentRun(question, "")
        for step in range(self.max_steps):
            resp = client.chat(model=self.model, messages=messages, tools=TOOLS,
                               options={"temperature": 0.0, "seed": 42, "num_ctx": settings.num_ctx})
            msg = resp.message
            messages.append(msg)
            run.steps = step + 1
            if not msg.tool_calls:
                run.answer = (msg.content or "").strip()
                return run
            for call in msg.tool_calls:
                name, args = call.function.name, dict(call.function.arguments)
                fn = self.impl.get(name)
                out = fn(**args) if fn else {"error": f"unknown tool {name}"}
                run.tool_calls.append({"name": name, "args": args, "output": out})
                messages.append({"role": "tool", "content": json.dumps(out), "tool_name": name})
        run.answer = "stopped: step limit reached"
        return run


def load_agent_golden(path: Path | None = None) -> list[dict]:
    p = path or (settings.golden_path.parent / "agent_v1.jsonl")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def agent_test_case(item: dict, run: AgentRun):
    """DeepEval test case: tools actually called vs the tools the golden expects."""
    from deepeval.test_case import LLMTestCase, ToolCall

    return LLMTestCase(
        input=item["question"], actual_output=run.answer, expected_output=item["reference"],
        tools_called=[ToolCall(name=c["name"], input_parameters=c["args"]) for c in run.tool_calls],
        expected_tools=[ToolCall(name=t["name"], input_parameters=t["args"])
                        for t in item["expected_tools"]],
    )


def ragas_messages(item: dict, run: AgentRun):
    """The same trajectory as RAGAS messages, for ToolCallAccuracy."""
    from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

    msgs = [HumanMessage(content=item["question"])]
    for c in run.tool_calls:
        msgs.append(AIMessage(content="", tool_calls=[ToolCall(name=c["name"], args=c["args"])]))
        msgs.append(ToolMessage(content=json.dumps(c.get("output", {}))))
    msgs.append(AIMessage(content=run.answer))
    reference = [ToolCall(name=t["name"], args=t["args"]) for t in item["expected_tools"]]
    return msgs, reference
