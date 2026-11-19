"""Module 11: multimodal RAG evaluation with qwen2.5vl:3b.

Run:  python scripts/make_images.py && python scripts/09_multimodal_eval.py
"""
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")
from evallab.config import settings  # noqa: E402
from evallab.custom_metrics import is_refusal  # noqa: E402
from evallab.multimodal import (answer_with_image, key_fact_recall, load_mm_golden,  # noqa: E402
                                make_vision_ragas_llm, ragas_multimodal_scores)


async def main():
    items = load_mm_golden(Path("data/golden/multimodal_v1.jsonl"), settings.image_dir)
    llm = make_vision_ragas_llm()
    rows = []
    for it in items:
        answer, lat = answer_with_image(it.question, it.images)
        kfr = key_fact_recall(answer, it.key_facts)
        if not it.key_facts:  # unanswerable from the image: correct == refusal
            kfr = 1.0 if is_refusal(answer) or "cannot tell" in answer.lower() else 0.0
        mm = await ragas_multimodal_scores(llm, it.question, answer, it.images)
        row = {"id": it.id, "question": it.question, "answer": answer, "latency_s": round(lat, 2),
               "key_fact_recall": kfr, **mm}
        rows.append(row)
        print(f"{it.id} kfr={kfr:.2f} mmF={mm['mm_faithfulness']} mmR={mm['mm_relevance']} "
              f"{lat:.1f}s | {answer[:80]!r}")

    def mean(k):
        v = [r[k] for r in rows if not math.isnan(r[k])]
        return sum(v) / len(v) if v else float("nan")

    print(f"\nkey_fact_recall={mean('key_fact_recall'):.2f}  "
          f"mm_faithfulness={mean('mm_faithfulness'):.2f}  mm_relevance={mean('mm_relevance'):.2f}")
    Path("results").mkdir(exist_ok=True)
    Path("results/multimodal_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
