"""Statistics for judge validation and regression decisions.

Conventions: for binary faithfulness labels, 1 = faithful (the positive class
for 'accept'), 0 = unfaithful. When the question is "does the judge catch
hallucinations?", compute metrics with positive=0 (see hallucination_view).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as sps
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score, recall_score)


@dataclass
class BinaryAgreement:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    cohen_kappa: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def binary_agreement(human: Sequence[int], judge: Sequence[int], positive: int = 1) -> BinaryAgreement:
    h, j = np.asarray(human), np.asarray(judge)
    tn, fp, fn, tp = confusion_matrix(h == positive, j == positive, labels=[False, True]).ravel()
    return BinaryAgreement(
        n=len(h),
        accuracy=float(accuracy_score(h, j)),
        precision=float(precision_score(h, j, pos_label=positive, zero_division=0)),
        recall=float(recall_score(h, j, pos_label=positive, zero_division=0)),
        f1=float(f1_score(h, j, pos_label=positive, zero_division=0)),
        cohen_kappa=float(cohen_kappa_score(h, j)) if len(set(h) | set(j)) > 1 else float("nan"),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def hallucination_view(human, judge) -> BinaryAgreement:
    """Treat 'unfaithful' (0) as the positive class: recall = share of real
    hallucinations the judge catches; precision = share of flags that are real."""
    return binary_agreement(human, judge, positive=0)


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman rank correlation and p-value, ignoring pairs with NaN."""
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return float("nan"), float("nan")
    rho, p = sps.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return float(rho), float(p)


def kappa_label(k: float) -> str:
    """Landis and Koch (1977) descriptive bands. A convention, not a law."""
    if math.isnan(k):
        return "undefined"
    for lim, name in [(0.0, "poor"), (0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
                      (0.80, "substantial"), (1.01, "almost perfect")]:
        if k <= lim:
            return name
    return "almost perfect"


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 7) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns (mean, low, high)."""
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return sum(vals) / len(vals), lo, hi


def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float], n_boot: int = 2000,
                           seed: int = 7) -> dict:
    """Bootstrap the mean paired difference (b - a) over the same items.

    Used by regression detection: a drop is only 'real' if the CI of the
    difference excludes zero. Pairing matters: the items are the same, so
    item difficulty cancels out.
    """
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if not pairs:
        return {"delta": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    diffs = [y - x for x, y in pairs]
    mean, lo, hi = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    return {"delta": mean, "low": lo, "high": hi, "n": len(diffs)}


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact McNemar test: do two judges differ in accuracy on the same items?"""
    b01 = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    b10 = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b01 + b10
    p = 1.0 if n == 0 else float(sps.binomtest(min(b01, b10), n, 0.5).pvalue)
    return {"a_only_correct": b01, "b_only_correct": b10, "p_value": p}


def self_consistency(runs: Sequence[Sequence[int]]) -> float:
    """Share of items on which every repeated run gave the same label."""
    if not runs:
        return float("nan")
    per_item = list(zip(*runs))
    return sum(1 for labels in per_item if len(set(labels)) == 1) / len(per_item)


def latency_summary(latencies: Sequence[float]) -> dict:
    arr = np.asarray([x for x in latencies if x == x])
    if arr.size == 0:
        return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan")}
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
            "mean": float(arr.mean())}


def threshold_sweep(scores: Sequence[float], human: Sequence[int],
                    thresholds: Sequence[float] | None = None) -> list[dict]:
    """For a continuous judge score (e.g. RAGAS faithfulness in [0, 1]) and binary
    human labels, report agreement at each candidate pass threshold.

    'pass' means score >= threshold. Use it to pick a threshold from data
    instead of defaulting to 0.5 or 0.7.
    """
    pairs = [(s, h) for s, h in zip(scores, human) if not math.isnan(s)]
    thresholds = thresholds or sorted({round(s, 3) for s, _ in pairs} | {0.5, 0.7, 0.9})
    out = []
    for t in thresholds:
        pred = [1 if s >= t else 0 for s, _ in pairs]
        a = binary_agreement([h for _, h in pairs], pred)
        hv = hallucination_view([h for _, h in pairs], pred)
        out.append({"threshold": t, "accuracy": a.accuracy, "kappa": a.cohen_kappa,
                    "hallucination_recall": hv.recall, "hallucination_precision": hv.precision,
                    "youden_j": hv.recall + (a.recall if a.recall == a.recall else 0) - 1})
    return out


def required_n_two_proportions(p: float, delta: float, alpha: float = 0.05,
                               power: float = 0.8) -> int:
    """Items per group needed to detect a change of `delta` in a pass rate near `p`
    (two-sided normal approximation). A planning number, not a guarantee; paired
    designs on the same golden set need fewer items."""
    za = sps.norm.ppf(1 - alpha / 2)
    zb = sps.norm.ppf(power)
    return math.ceil((za + zb) ** 2 * 2 * p * (1 - p) / delta ** 2)
