"""RAGAS 0.4 quickstart: score ONE hand-written sample with every metric,
using local Ollama models only.

Run:  python examples/ragas_quickstart.py
Needs: ollama serve, qwen2.5:3b, nomic-embed-text.
"""
import asyncio

import instructor
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (AnswerRelevancy, ContextPrecision, ContextRecall,
                                       ContextRelevance, FactualCorrectness, Faithfulness,
                                       NoiseSensitivity)

# Ollama's OpenAI-compatible endpoint. The key is required by the client, ignored by Ollama.
client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
llm = llm_factory("qwen2.5:3b", provider="openai", client=client,
                  mode=instructor.Mode.JSON, temperature=0.0, max_tokens=2048)
emb = OpenAIEmbeddings(client=client, model="nomic-embed-text")

sample = dict(
    user_input="A Business customer had 98.5% uptime in a month. What credit do they get?",
    response="98.5% is below 99.0% but at least 95.0%, so they get a 25% credit of that "
             "month's storage charges, which Helios pays out in cash.",          # one false claim
    reference="98.5% is below 99.0% but at least 95.0%, so a Business customer is entitled "
              "to a 25% credit of that month's storage charges, applied to the next invoice.",
    retrieved_contexts=[
        "[SLA > Credit schedule (Business plan, 99.9% SLA)] Monthly uptime below 99.9% but at "
        "least 99.0%: 10% credit. Below 99.0% but at least 95.0%: 25% credit. Below 95.0%: 50% credit.",
        "[SLA > Claiming credits] Credits are applied to the next invoice and are never paid out in cash.",
        "[Pricing > Business] The Business plan costs $0.018 per GB-month.",   # noise
    ],
)


async def main():
    s = sample
    jobs = {
        "faithfulness": Faithfulness(llm=llm).ascore(
            user_input=s["user_input"], response=s["response"], retrieved_contexts=s["retrieved_contexts"]),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb).ascore(
            user_input=s["user_input"], response=s["response"]),
        "context_precision": ContextPrecision(llm=llm).ascore(
            user_input=s["user_input"], reference=s["reference"], retrieved_contexts=s["retrieved_contexts"]),
        "context_recall": ContextRecall(llm=llm).ascore(
            user_input=s["user_input"], retrieved_contexts=s["retrieved_contexts"], reference=s["reference"]),
        "context_relevance": ContextRelevance(llm=llm).ascore(
            user_input=s["user_input"], retrieved_contexts=s["retrieved_contexts"]),
        "factual_correctness": FactualCorrectness(llm=llm, mode="f1").ascore(
            response=s["response"], reference=s["reference"]),
        "noise_sensitivity": NoiseSensitivity(llm=llm).ascore(
            user_input=s["user_input"], response=s["response"], reference=s["reference"],
            retrieved_contexts=s["retrieved_contexts"]),
    }
    for name, coro in jobs.items():     # sequential: kind to a laptop running one 3B model
        try:
            result = await coro
            print(f"{name:<20} {result.value:.3f}")
        except Exception as e:
            print(f"{name:<20} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
