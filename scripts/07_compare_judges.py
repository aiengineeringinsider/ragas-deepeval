"""Modules 9 and 10: compare local judges against human labels.

For each judge model: binary faithfulness verdicts (with repeats for
self-consistency) and 1-5 scores on the 40-case calibration set, then
accuracy, precision, recall, F1, Cohen's kappa, Spearman rho, latency, and
pairwise McNemar tests between judges.

Run:  python scripts/07_compare_judges.py --repeats 3
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, "src")
from evallab.config import settings  # noqa: E402
from evallab.experiments import run_judge_on_calibration, save_jsonl, summarize_judge  # noqa: E402
from evallab.judges import load_calibration  # noqa: E402
from evallab.llm import OllamaChat  # noqa: E402
from evallab.stats import kappa_label, mcnemar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(settings.judge_models))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_calibration(settings.calibration_path, settings.corpus_dir)[: args.limit or None]
    print(f"calibration cases: {len(rows)} "
          f"(faithful={sum(r['human_faithful'] for r in rows)}, "
          f"unfaithful={sum(1 - r['human_faithful'] for r in rows)})")

    all_results, summaries = {}, []
    for model in args.models.split(","):
        print(f"\n== judge {model} ==")
        res = run_judge_on_calibration(OllamaChat(model), rows, repeats=args.repeats,
                                       temperature=args.temperature)
        save_jsonl(res, Path(f"results/judge_{model.replace(':', '_')}.jsonl"))
        s = summarize_judge(res)
        summaries.append(s)
        all_results[model] = {r["id"]: r for r in res}
        print(json.dumps(s, indent=2))

    print("\n" + f"{'judge':<14}{'acc':>6}{'F1':>6}{'H-rec':>7}{'kappa':>7}{'rho':>7}"
          f"{'cons':>6}{'p50 s':>7}{'fail':>5}")
    for s in summaries:
        cons = s["self_consistency"]
        print(f"{s['model']:<14}{s['accuracy']:>6.2f}{s['f1_faithful']:>6.2f}"
              f"{s['hallucination_recall']:>7.2f}{s['cohen_kappa']:>7.2f}{s['spearman_rho']:>7.2f}"
              f"{(cons if cons is not None else float('nan')):>6.2f}{s['latency_p50']:>7.2f}"
              f"{s['parse_failures']:>5}   kappa: {kappa_label(s['cohen_kappa'])}")

    print("\nMcNemar (same items, is one judge more accurate?)")
    for a, b in combinations(all_results, 2):
        ids = [i for i in all_results[a] if all_results[a][i]["label"] is not None
               and all_results[b].get(i, {}).get("label") is not None]
        ca = [all_results[a][i]["label"] == all_results[a][i]["human_faithful"] for i in ids]
        cb = [all_results[b][i]["label"] == all_results[b][i]["human_faithful"] for i in ids]
        print(f"  {a} vs {b}: {mcnemar(ca, cb)}")
    Path("results/judge_comparison.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
