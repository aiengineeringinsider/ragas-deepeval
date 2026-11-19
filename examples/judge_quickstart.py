"""Module 7: the three judge formats (binary, 1-5 score, pairwise) on two
human-labeled cases, with each local judge model.

Run:  python examples/judge_quickstart.py
"""
import sys

sys.path.insert(0, "src")
from evallab.config import settings  # noqa: E402
from evallab.judges import (judge_faithfulness, judge_pairwise, judge_score,  # noqa: E402
                            load_calibration)
from evallab.llm import OllamaChat  # noqa: E402

rows = {r["id"]: r for r in load_calibration(settings.calibration_path, settings.corpus_dir)}
good, bad = rows["h015"], rows["h016"]   # same question: $250 (correct) vs $180 (omits minimum)

for model in settings.judge_models:
    llm = OllamaChat(model)
    print(f"\n=== judge: {model} ===")
    for case in (good, bad):
        v = judge_faithfulness(llm, case["question"], case["context"], case["answer"])
        s = judge_score(llm, case["question"], case["context"], case["answer"])
        print(f"{case['id']} human=({case['human_faithful']}, {case['human_score']}) "
              f"judge=({v.label}, {s.score})  {v.latency_s:4.1f}s  reason: {v.reason[:70]}")
    w1, _ = judge_pairwise(llm, good["question"], good["context"], good["answer"], bad["answer"])
    w2, _ = judge_pairwise(llm, good["question"], good["context"], bad["answer"], good["answer"])
    print(f"pairwise: good-as-A -> {w1}, good-as-B -> {w2}  "
          f"({'consistent' if (w1, w2) == ('A', 'B') else 'ORDER-DEPENDENT or wrong'})")
