"""Judge-comparison and bias experiments (Modules 9 and 10).

Every experiment writes raw per-item verdicts to JSONL before computing any
statistic, so a surprising number can always be traced back to the exact
prompts and model outputs that produced it.
"""
from __future__ import annotations

import json
from pathlib import Path

from .judges import (FAITHFULNESS_PROMPT, judge_faithfulness, judge_pairwise, judge_score)
from .stats import (binary_agreement, hallucination_view, latency_summary, self_consistency,
                    spearman)


def run_judge_on_calibration(llm, rows: list[dict], repeats: int = 1,
                             temperature: float = 0.0) -> list[dict]:
    out = []
    for row in rows:
        labels, lats, reasons = [], [], []
        for rep in range(repeats):
            # with temperature > 0, vary the seed to measure sampling variance
            seed = 42 if temperature == 0 else 1000 + rep
            v = judge_faithfulness(llm, row["question"], row["context"], row["answer"],
                                   seed=seed, temperature=temperature)
            labels.append(v.label)
            lats.append(v.latency_s)
            reasons.append(v.reason)
        s = judge_score(llm, row["question"], row["context"], row["answer"])
        out.append({"id": row["id"], "model": llm.model, "human_faithful": row["human_faithful"],
                    "human_score": row["human_score"], "error_type": row["error_type"],
                    "labels": labels, "label": labels[0], "score": s.score,
                    "latency_s": lats[0], "reason": reasons[0]})
    return out


def summarize_judge(results: list[dict]) -> dict:
    parsed = [r for r in results if r["label"] is not None]
    human = [r["human_faithful"] for r in parsed]
    judge = [r["label"] for r in parsed]
    acc = binary_agreement(human, judge).as_dict()
    hall = hallucination_view(human, judge).as_dict()
    scored = [r for r in results if r["score"] is not None]
    rho, p = spearman([float(r["human_score"]) for r in scored], [float(r["score"]) for r in scored])
    runs = list(zip(*[r["labels"] for r in results])) if results and len(results[0]["labels"]) > 1 else []
    return {
        "model": results[0]["model"] if results else None,
        "n": len(results),
        "parse_failures": len(results) - len(parsed),
        "accuracy": acc["accuracy"],
        "precision_faithful": acc["precision"],
        "recall_faithful": acc["recall"],
        "f1_faithful": acc["f1"],
        "hallucination_recall": hall["recall"],
        "hallucination_precision": hall["precision"],
        "cohen_kappa": acc["cohen_kappa"],
        "spearman_rho": round(rho, 4) if rho == rho else rho,
        "spearman_p": p,
        "self_consistency": self_consistency(runs) if runs else None,
        **{f"latency_{k}": round(v, 3) for k, v in
           latency_summary([r["latency_s"] for r in results]).items()},
    }


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


# --------------------------------------------------------------------------
# Bias probes
# --------------------------------------------------------------------------
def position_bias(llm, pairs: list[dict]) -> dict:
    """Each pair has a known better answer. Ask twice with A/B swapped.

    consistent  : picks the same underlying answer in both orders
    first_rate  : share of all verdicts that chose position A
    accuracy    : share of verdicts that chose the known-better answer
    """
    consistent = first = correct = total = 0
    for p in pairs:
        w1, _ = judge_pairwise(llm, p["question"], p["context"], p["good"], p["bad"])
        w2, _ = judge_pairwise(llm, p["question"], p["context"], p["bad"], p["good"])
        if w1 is None or w2 is None:
            continue
        total += 1
        picked_good_1, picked_good_2 = (w1 == "A"), (w2 == "B")
        consistent += int(picked_good_1 == picked_good_2)
        first += int(w1 == "A") + int(w2 == "A")
        correct += int(picked_good_1) + int(picked_good_2)
    return {"model": llm.model, "pairs": total,
            "order_consistency": consistent / total if total else float("nan"),
            "first_position_rate": first / (2 * total) if total else float("nan"),
            "accuracy": correct / (2 * total) if total else float("nan")}


PADDING = (" To give you some helpful background, Helios Cloud Storage is designed to be "
           "reliable, secure, and cost-effective for teams of every size, and our documentation "
           "covers pricing, limits, security, and support in detail so you can plan with confidence.")


def verbosity_bias(llm, rows: list[dict]) -> dict:
    """Same answer, with and without content-free padding. A judge without
    verbosity bias gives the same 1-5 score to both."""
    deltas = []
    for r in rows:
        s1 = judge_score(llm, r["question"], r["context"], r["answer"]).score
        s2 = judge_score(llm, r["question"], r["context"], r["answer"] + PADDING).score
        if s1 is not None and s2 is not None:
            deltas.append(s2 - s1)
    return {"model": llm.model, "n": len(deltas),
            "mean_score_delta_padded_minus_plain": sum(deltas) / len(deltas) if deltas else float("nan"),
            "share_padded_higher": sum(d > 0 for d in deltas) / len(deltas) if deltas else float("nan")}


PROMPT_VARIANTS = {
    "baseline": FAITHFULNESS_PROMPT,
    "lenient_wording": FAITHFULNESS_PROMPT.replace("You are a strict fact-checker.",
                                                   "You are a helpful reviewer."),
    "answer_first": FAITHFULNESS_PROMPT.replace(
        "CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER: {answer}",
        "ANSWER: {answer}\nQUESTION: {question}\n\nCONTEXT:\n{context}"),
}


def prompt_sensitivity(llm, rows: list[dict]) -> dict:
    """Agreement with humans under semantically equivalent prompt variants,
    plus the share of items whose verdict flips between variants."""
    labels = {}
    for name, prompt in PROMPT_VARIANTS.items():
        labels[name] = [judge_faithfulness(llm, r["question"], r["context"], r["answer"],
                                           prompt=prompt).label for r in rows]
    human = [r["human_faithful"] for r in rows]
    per_variant = {}
    for name, ls in labels.items():
        keep = [(h, l) for h, l in zip(human, ls) if l is not None]
        per_variant[name] = binary_agreement([h for h, _ in keep], [l for _, l in keep]).accuracy
    flips = sum(1 for i in range(len(rows)) if len({labels[n][i] for n in labels}) > 1)
    return {"model": llm.model, "accuracy_by_variant": per_variant,
            "flip_rate": flips / len(rows) if rows else float("nan")}


def self_preference(judge_llm, generators: dict[str, list[dict]]) -> dict:
    """Mean 1-5 score the judge gives to answers from each generator model on
    the same questions. If the judge scores its own family higher than humans
    do, that gap is self-preference. Needs human scores for a fair read."""
    out = {}
    for gen_model, rows in generators.items():
        scores = [judge_score(judge_llm, r["question"], r["context"], r["answer"]).score for r in rows]
        scores = [s for s in scores if s is not None]
        out[gen_model] = sum(scores) / len(scores) if scores else float("nan")
    return {"judge": judge_llm.model, "mean_score_by_generator": out}
