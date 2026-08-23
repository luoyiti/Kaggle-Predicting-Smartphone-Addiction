"""OOF blend / stack helpers. Synthetic arrays only."""

import numpy as np

from s6e8.blending import blend_grid, blend_mean, stack_logistic_cv, stack_ridge_cv


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
