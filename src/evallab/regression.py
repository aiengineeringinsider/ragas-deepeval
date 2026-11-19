"""Regression detection: compare a candidate run with a stored baseline.

Two gates, both must pass:
  1. Absolute floors: each metric mean must stay above a hard minimum.
  2. Relative drop: over the same items, the paired mean difference
     (candidate - baseline) fails the gate when it is worse than -max_drop
     AND its bootstrap confidence interval lies entirely below zero.

Why both: floors catch catastrophes; paired deltas catch slow erosion that a
floor would let through, while the CI keeps judge noise from failing builds.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .records import EvalRecord
from .stats import paired_bootstrap_delta

DEFAULT_FLOORS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.60,
    "context_recall": 0.60,
    "refusal_correctness": 0.90,
    "citation_validity": 0.80,
    "numeric_grounding": 0.85,
}


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def report(self) -> str:
        lines = [f"REGRESSION GATE: {'PASS' if self.passed else 'FAIL'}"]
        for m, d in self.details.items():
            lines.append(
                f"  {m:<22} base={d['baseline']:.3f} cand={d['candidate']:.3f} "
                f"delta={d['delta']:+.3f} CI=[{d['low']:+.3f},{d['high']:+.3f}] {d['status']}")
        lines += [f"  - {f}" for f in self.failures]
        return "\n".join(lines)


def _mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def per_item_scores(records: list[EvalRecord]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in records:
        for k, v in r.scores.items():
            if not k.startswith("_"):
                out.setdefault(k, {})[r.id] = v
    return out


def compare(baseline: dict[str, dict[str, float]], candidate: dict[str, dict[str, float]],
            floors: dict[str, float] | None = None, max_drop: float = 0.05) -> GateResult:
    floors = DEFAULT_FLOORS if floors is None else floors
    res = GateResult(passed=True)
    for metric in sorted(set(baseline) & set(candidate)):
        ids = sorted(set(baseline[metric]) & set(candidate[metric]))
        a = [baseline[metric][i] for i in ids]
        b = [candidate[metric][i] for i in ids]
        d = paired_bootstrap_delta(a, b)
        base_mean, cand_mean = _mean(a), _mean(b)
        status = "ok"
        if metric in floors and cand_mean < floors[metric]:
            status = "BELOW_FLOOR"
            res.failures.append(f"{metric}: mean {cand_mean:.3f} < floor {floors[metric]:.2f}")
        elif d["n"] and d["delta"] < -max_drop and d["high"] < 0:
            status = "REGRESSED"
            res.failures.append(f"{metric}: dropped {d['delta']:+.3f} (CI high {d['high']:+.3f} < 0)")
        res.details[metric] = {"baseline": base_mean, "candidate": cand_mean, "delta": d["delta"],
                               "low": d["low"], "high": d["high"], "n": d["n"], "status": status}
    res.passed = not res.failures
    return res


def save_baseline(records: list[EvalRecord], path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta or {}, "scores": per_item_scores(records)}, indent=2),
                    encoding="utf-8")


def load_baseline(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {m: {k: (float("nan") if v is None else float(v)) for k, v in d.items()}
            for m, d in data["scores"].items()}
