"""Thin, testable wrapper around the Ollama Python client.

Why a wrapper instead of calling ollama.chat everywhere?
  * one place to set temperature, context length, and keep_alive;
  * one place to disable Qwen3 "thinking" so judges return clean JSON;
  * latency is measured for every call, which Module 9 needs;
  * tests can swap in a FakeLLM without touching the network.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .config import settings


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatModel(Protocol):
    model: str

    def chat(self, messages: list[dict[str, Any]], *, json_schema: dict | None = None,
             temperature: float = 0.0, seed: int | None = 42) -> LLMResponse: ...


def _is_thinking_model(model: str) -> bool:
    return model.startswith("qwen3")


class OllamaChat:
    """Synchronous chat client for one Ollama model."""

    def __init__(self, model: str, host: str | None = None, num_ctx: int | None = None):
        import ollama  # imported lazily so unit tests never need the package

        self.model = model
        self.num_ctx = num_ctx or settings.num_ctx
        self._client = ollama.Client(host=host or settings.ollama_host,
                                     timeout=settings.request_timeout_s)

    def chat(self, messages, *, json_schema=None, temperature=0.0, seed=42) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx,
                        **({"seed": seed} if seed is not None else {})},
            "keep_alive": "10m",
        }
        if json_schema is not None:
            kwargs["format"] = json_schema  # Ollama structured outputs
        if _is_thinking_model(self.model):
            kwargs["think"] = False  # judges must answer, not deliberate in <think>
        t0 = time.perf_counter()
        resp = self._client.chat(**kwargs)
        latency = time.perf_counter() - t0
        return LLMResponse(
            text=resp.message.content or "",
            latency_s=latency,
            model=self.model,
            prompt_tokens=getattr(resp, "prompt_eval_count", None),
            completion_tokens=getattr(resp, "eval_count", None),
        )


class FakeLLM:
    """Deterministic stand-in used by unit tests and CI without Ollama."""

    def __init__(self, model: str = "fake", responses: list[str] | None = None,
                 default: str = "I don't know."):
        self.model = model
        self._responses = list(responses or [])
        self.default = default
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, *, json_schema=None, temperature=0.0, seed=42) -> LLMResponse:
        self.calls.append(messages)
        text = self._responses.pop(0) if self._responses else self.default
        return LLMResponse(text=text, latency_s=0.001, model=self.model)


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and chatter."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])
