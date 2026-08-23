"""Ensemble, stack, and calibration on synthetic OOF vectors."""

from __future__ import annotations

import numpy as np

from s6e8.calibration import calibrate_with_oof_cv
from s6e8.ensemble import auc_weights, blend_predictions, logistic_stack, rank01


def _toy():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 40)
    noise = rng.normal(0, 0.05, size=len(y))
    a = np.clip(y.astype(float) * 0.8 + 0.1 + noise, 0.01, 0.99)
    b = np.clip(y.astype(float) * 0.7 + 0.15 + noise[::-1] * 0.5, 0.01, 0.99)
    test_a = a[:20]
    test_b = b[:20]
    return y, [a, b], [test_a, test_b]


def test_rank01_is_in_unit_interval():
    x = np.array([0.9, 0.1, 0.1])
    r = rank01(x)
    assert r.min() > 0
    assert r.max() < 1
    assert r[0] > r[1]


def test_auc_weights_prefer_stronger_model():
    w = auc_weights([0.96, 0.51])
    assert w[0] > w[1]
    assert np.isclose(w.sum(), 1.0)


def test_blend_methods_run():
    y, oofs, tests = _toy()
    for method in ("mean", "rank", "logit", "grid", "auc_weighted"):
        result = blend_predictions(oofs, tests, y, method=method, component_aucs=[0.9, 0.8])
        assert result["oof"].shape == y.shape
        assert result["test"].shape[0] == tests[0].shape[0]
        assert 0.5 <= result["oof_auc"] <= 1.0
        assert np.isclose(result["weights"].sum(), 1.0)


def test_logistic_stack_oof_shape():
    y, oofs, tests = _toy()
    result = logistic_stack(oofs, tests, y, n_splits=4, seed=0)
    assert result["oof"].shape == y.shape
    assert result["mean_coef"].shape == (2,)
    assert 0.5 <= result["oof_auc"] <= 1.0


def test_isotonic_and_platt_preserve_ranking_quality():
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 50)
    p = np.clip(y.astype(float) * 0.6 + 0.2 + rng.normal(0, 0.05, size=len(y)), 0.02, 0.98)
    test = p[:30]
    iso = calibrate_with_oof_cv(p, y, test, method="isotonic", n_splits=4, seed=1)
    platt = calibrate_with_oof_cv(p, y, test, method="platt", n_splits=4, seed=1)
    assert iso["oof"].shape == p.shape
    assert platt["oof"].shape == p.shape
    assert iso["calibrated"]["auc"] >= 0.7
    assert platt["calibrated"]["auc"] >= 0.7
