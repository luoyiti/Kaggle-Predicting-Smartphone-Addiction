"""OOF blend / stack helpers. Synthetic arrays only."""

import numpy as np
import pandas as pd
import pytest

from s6e8.blending import (
    blend_geom,
    blend_geom_grid,
    blend_grid,
    blend_mean,
    blend_power,
    blend_power_grid,
    blend_rank_grid,
    power_mean,
    stack_logistic_cv,
    stack_ridge_cv,
)
from scripts_loader import load_script


def test_blend_mean_is_average():
    oofs = [np.array([0.2, 0.8]), np.array([0.4, 0.6])]
    tests = [np.array([0.1, 0.9]), np.array([0.3, 0.7])]
    oof, test, w = blend_mean(oofs, tests)
    np.testing.assert_allclose(oof, [0.3, 0.7])
    np.testing.assert_allclose(test, [0.2, 0.8])
    np.testing.assert_allclose(w, [0.5, 0.5])


def test_grid_prefers_the_better_model():
    y = np.array([0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    bad = 1.0 - good
    oof, _, w = blend_grid([good, bad], [good, bad], y, step=25)
    assert w[0] > w[1]
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, oof) >= roc_auc_score(y, good) - 1e-12


def test_stack_logistic_recovers_a_strong_column():
    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 2, size=n)
    good = np.clip(y.astype(float) + rng.normal(0, 0.05, size=n), 0, 1)
    bad = rng.random(n)
    oof, test, coef, extra = stack_logistic_cv(
        [good, bad],
        [good, bad],
        y,
        seed=0,
        n_splits=4,
        C=1.0,
    )
    assert extra["n_splits"] == 4
    assert coef[0] > coef[1]
    assert oof.shape == (n,)
    assert test.shape == (n,)
    assert np.isfinite(oof).all()


def test_stack_ridge_clips_to_unit_interval():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    oofs = [np.array([0.1, 0.9, 0.2, 0.8, 0.0, 1.0, 0.3, 0.7])]
    oof, test, coef, extra = stack_ridge_cv(oofs, oofs, y, seed=0, n_splits=2, alpha=0.1)
    assert extra["alpha"] == 0.1
    assert np.all((oof >= 0) & (oof <= 1))
    assert np.all((test >= 0) & (test <= 1))
    assert coef.shape == (1,)


def test_geom_is_exp_mean_log():
    oofs = [np.array([0.25, 0.64]), np.array([0.25, 0.64])]
    tests = [np.array([0.16, 0.36]), np.array([0.16, 0.36])]
    oof, test, w = blend_geom(oofs, tests)
    np.testing.assert_allclose(oof, [0.25, 0.64])
    np.testing.assert_allclose(test, [0.16, 0.36])
    np.testing.assert_allclose(w, [0.5, 0.5])
    mixed = power_mean([np.array([0.25]), np.array([0.64])], 0.0)
    np.testing.assert_allclose(mixed, [0.4])


def test_power_mean_harmonic_and_quadratic():
    pair = [np.array([0.25]), np.array([0.5])]
    np.testing.assert_allclose(power_mean(pair, -1.0), [1.0 / 3.0])
    np.testing.assert_allclose(power_mean(pair, 2.0), [np.sqrt((0.25**2 + 0.5**2) / 2.0)])
    oof, _, w = blend_power(pair, pair, p=-1.0)
    np.testing.assert_allclose(oof, [1.0 / 3.0])
    np.testing.assert_allclose(w, [0.5, 0.5])


def test_power_zero_matches_geom():
    arrays = [np.array([0.2, 0.8]), np.array([0.4, 0.6])]
    np.testing.assert_allclose(power_mean(arrays, 0.0), power_mean(arrays, 1e-16))


def test_rank_grid_prefers_the_better_model():
    y = np.array([0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    bad = 1.0 - good
    oof, _, w = blend_rank_grid([good, bad], [good, bad], y, step=25)
    assert w[0] > w[1]
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, oof) >= roc_auc_score(y, good) - 1e-12


def test_geom_grid_and_power_grid_are_finite():
    y = np.array([0, 0, 1, 1])
    a = np.array([0.2, 0.3, 0.7, 0.8])
    b = np.array([0.25, 0.35, 0.65, 0.75])
    oof_g, test_g, w_g = blend_geom_grid([a, b], [a, b], y, step=50)
    oof_p, test_p, w_p = blend_power_grid([a, b], [a, b], y, p=2.0, step=50)
    assert np.isfinite(oof_g).all() and np.isfinite(test_g).all()
    assert np.isfinite(oof_p).all() and np.isfinite(test_p).all()
    assert abs(float(np.sum(w_g)) - 1.0) < 1e-12
    assert abs(float(np.sum(w_p)) - 1.0) < 1e-12


def test_blend_oof_missing_seed_run_explains_mean_blend(tmp_path):
    blend = load_script("blend_oof.py")
    with pytest.raises(FileNotFoundError, match="catboost_exactcat_budget_seedavg"):
        blend._load("catboost_exactcat_budget_seed7", tmp_path)


def test_blend_oof_fails_loud_on_duplicate_ids(tmp_path):
    blend = load_script("blend_oof.py")
    folder = tmp_path / "dup"
    folder.mkdir()
    oof = pd.DataFrame(
        {"id": [1, 1], "addicted_label": [0, 1], "pred": [0.2, 0.8]}
    )
    test = pd.DataFrame({"id": [10, 11], "pred": [0.3, 0.7]})
    oof.to_parquet(folder / "oof.parquet", index=False)
    test.to_parquet(folder / "test.parquet", index=False)
    (folder / "metrics.json").write_text('{"oof_auc": 0.5}', encoding="utf-8")
    with pytest.raises(ValueError, match="ids are not unique"):
        blend._load("dup", tmp_path)
