"""Module 17: multi-turn evaluation of scripted Helios conversations.

Run:  python scripts/14_conversations.py           # deterministic retention check
      python scripts/14_conversations.py --judge   # + DeepEval conversational metrics
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evallab.config import settings  # noqa: E402
from evallab.conversation import (ConversationalRAG, conversation_metrics,  # noqa: E402
                                  helios_conversations, load_conversations, retention_check)
from evallab.llm import OllamaChat  # noqa: E402
from evallab.rag import build_default_pipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--judge", action="store_true")
ap.add_argument("--no-rewrite", action="store_true", help="single-turn baseline: no question rewriting")
args = ap.parse_args()

pipeline = build_default_pipeline()
bot = ConversationalRAG(pipeline, OllamaChat(settings.generator_model))
if args.no_rewrite:
    bot.standalone = lambda history, message: message
specs = load_conversations()
convs = [bot.run(s["user_messages"], s["id"]) for s in specs]
for s, c in zip(specs, convs):
    print(json.dumps(retention_check(c, s["expectations"]), ensure_ascii=False))
if args.judge:
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig

    from evallab.deepeval_eval import make_judge

    evaluate([helios_conversations(s, c) for s, c in zip(specs, convs)],
             conversation_metrics(make_judge()), async_config=AsyncConfig(run_async=False))
