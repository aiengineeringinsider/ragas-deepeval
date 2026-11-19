"""Module 18: run the tool-using agent on the agent golden set and score its trajectories.

Run:  python scripts/15_agent_eval.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepeval.metrics import ToolCorrectnessMetric  # noqa: E402
from deepeval.test_case import ToolCallParams  # noqa: E402
from ragas.metrics.collections import ToolCallAccuracy  # noqa: E402

from evallab.agent import HeliosAgent, agent_test_case, load_agent_golden, ragas_messages  # noqa: E402
from evallab.deepeval_eval import make_judge  # noqa: E402
from evallab.rag import build_default_pipeline, format_context  # noqa: E402

pipeline = build_default_pipeline()
agent = HeliosAgent(search=lambda q: format_context(pipeline.retrieve(q)[0]))
judge = make_judge()   # ToolCorrectnessMetric builds a model even in deterministic mode
tca = ToolCallAccuracy()
print(f"{'id':5} {'steps':5} {'names':6} {'args':6} {'ragas':6} answer")
for item in load_agent_golden():
    run = agent.run(item["question"])
    tc = agent_test_case(item, run)
    names = ToolCorrectnessMetric(threshold=1.0, model=judge, async_mode=False)
    args = ToolCorrectnessMetric(threshold=1.0, model=judge, async_mode=False,
                                 evaluation_params=[ToolCallParams.INPUT_PARAMETERS])
    names.measure(tc)
    args.measure(tc)
    msgs, ref = ragas_messages(item, run)
    r = asyncio.run(tca.ascore(user_input=msgs, reference_tool_calls=ref))
    print(f"{item['id']:5} {run.steps:5} {names.score:6.2f} {args.score:6.2f} {r.value:6.2f} {run.answer[:60]!r}")
    for c in run.tool_calls:
        print(f"      {c['name']}({c['args']}) -> {c['output']}")
