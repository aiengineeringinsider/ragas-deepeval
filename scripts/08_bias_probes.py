"""Module 9: probe each judge for position, verbosity, and prompt-sensitivity bias,
and run a self-preference check across generator families.

Run:  python scripts/08_bias_probes.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from evallab.config import settings  # noqa: E402
from evallab.experiments import (position_bias, prompt_sensitivity, self_preference,  # noqa: E402
                                 verbosity_bias)
from evallab.judges import load_calibration  # noqa: E402
from evallab.llm import OllamaChat  # noqa: E402


def build_pairs(rows):
    """Pair each faithful answer with the unfaithful answer to the same question."""
    by_q = {}
    for r in rows:
        by_q.setdefault(r["golden_id"], {})[r["human_faithful"]] = r
    return [{"question": d[1]["question"], "context": d[1]["context"],
             "good": d[1]["answer"], "bad": d[0]["answer"]}
            for d in by_q.values() if 0 in d and 1 in d]


def main():
    rows = load_calibration(settings.calibration_path, settings.corpus_dir)
    faithful = [r for r in rows if r["human_faithful"] == 1]
    pairs = build_pairs(rows)
    report = {}
    for model in settings.judge_models:
        llm = OllamaChat(model)
        print(f"\n== {model} ==")
        report[model] = {
            "position": position_bias(llm, pairs),
            "verbosity": verbosity_bias(llm, faithful),
            "prompt_sensitivity": prompt_sensitivity(llm, rows),
        }
        print(json.dumps(report[model], indent=2))

    # Self-preference: answers to the same golden questions produced by two
    # generator families (run 04_golden.py run with EVALLAB_GENERATOR set to
    # each model first). Judge qwen2.5:3b scores both sets.
    gens = {}
    for gen in ["qwen2.5:3b", "llama3.2:1b"]:
        p = Path(f"results/responses_{gen.replace(':', '_')}.jsonl")
        if p.exists():
            recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            gens[gen] = [{"question": r["question"], "context": "\n\n".join(r["contexts"]),
                          "answer": r["answer"]} for r in recs]
    if len(gens) == 2:
        report["self_preference"] = self_preference(OllamaChat("qwen2.5:3b"), gens)
        print(json.dumps(report["self_preference"], indent=2))
    else:
        print("\n(self-preference skipped: generate responses with both generators first)")
    Path("results/bias_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
