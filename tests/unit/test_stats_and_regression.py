import math

import pytest

from evallab.regression import compare
from evallab.stats import (binary_agreement, bootstrap_ci, hallucination_view, kappa_label,
                           mcnemar, paired_bootstrap_delta, self_consistency, spearman)


def test_binary_agreement_known_values():
    human = [1, 1, 1, 1, 0, 0, 0, 0]
    judge = [1, 1, 1, 0, 0, 0, 1, 1]
    a = binary_agreement(human, judge)
    assert a.accuracy == pytest.approx(5 / 8)
    assert a.precision == pytest.approx(3 / 5)
    assert a.recall == pytest.approx(3 / 4)
    assert (a.tn, a.fp, a.fn, a.tp) == (2, 2, 1, 3)
    # kappa = (po - pe) / (1 - pe) = (0.625 - 0.5) / 0.5
    assert a.cohen_kappa == pytest.approx(0.25)
    h = hallucination_view(human, judge)
    assert h.recall == pytest.approx(2 / 4)  # caught 2 of 4 hallucinations


def test_lenient_judge_has_high_accuracy_but_zero_kappa():
    human = [1] * 9 + [0]
    judge = [1] * 10  # always says "faithful"
    a = binary_agreement(human, judge)
    assert a.accuracy == pytest.approx(0.9)
    assert a.cohen_kappa == pytest.approx(0.0)
    assert hallucination_view(human, judge).recall == 0.0


def test_spearman_and_labels():
    rho, _ = spearman([1, 2, 3, 4, 5], [2, 3, 4, 5, 9])
    assert rho == pytest.approx(1.0)
    assert kappa_label(0.65) == "substantial"
    assert kappa_label(0.1) == "slight"


def test_bootstrap_and_consistency():
    mean, lo, hi = bootstrap_ci([0.8] * 50 + [0.6] * 50)
    assert lo < mean < hi and mean == pytest.approx(0.7)
    assert self_consistency([[1, 0, 1], [1, 0, 0], [1, 0, 1]]) == pytest.approx(2 / 3)
    assert mcnemar([True, True, False], [True, False, False])["a_only_correct"] == 1


def test_regression_gate_ignores_noise_and_catches_real_drops():
    ids = [f"g{i:03d}" for i in range(40)]
    base = {"faithfulness": {i: 0.9 for i in ids}}
    noisy = {"faithfulness": {i: (0.88 if n % 2 else 0.92) for n, i in enumerate(ids)}}
    assert compare(base, noisy, floors={}).passed

    dropped = {"faithfulness": {i: 0.75 for i in ids}}
    res = compare(base, dropped, floors={})
    assert not res.passed and "dropped" in res.failures[0]

    floor = compare(base, dropped, floors={"faithfulness": 0.8})
    assert not floor.passed and "floor" in floor.failures[0]


def test_paired_delta_handles_nan():
    d = paired_bootstrap_delta([1.0, math.nan, 0.5], [1.0, 0.2, 0.4])
    assert d["n"] == 2


def test_threshold_sweep_finds_separating_threshold():
    from evallab.stats import threshold_sweep

    scores = [1.0, 0.9, 0.8, 0.4, 0.3, 0.0]
    human = [1, 1, 1, 0, 0, 0]
    best = max(threshold_sweep(scores, human, thresholds=[0.3, 0.5, 0.9]), key=lambda r: r["kappa"])
    assert best["threshold"] == 0.5 and best["accuracy"] == 1.0
