"""OOF blending / stacking helpers. CLI lives in ``scripts/blend_oof.py``."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy() / (len(x) + 1.0)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def grid_weights(n: int, step: int = 5) -> list[tuple[float, ...]]:
    ticks = list(range(0, 101, step))
    out = []
    for combo in product(ticks, repeat=n):
        if sum(combo) != 100:
            continue
        if all(v == 0 for v in combo):
            continue
        out.append(tuple(v / 100.0 for v in combo))
    return out


def blend_mean(oofs: list[np.ndarray], tests: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.ones(len(oofs), dtype=float) / len(oofs)
    return np.mean(oofs, axis=0), np.mean(tests, axis=0), weights


def blend_rank(oofs: list[np.ndarray], tests: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.ones(len(oofs), dtype=float) / len(oofs)
    return (
        np.mean([rank01(x) for x in oofs], axis=0),
        np.mean([rank01(x) for x in tests], axis=0),
        weights,
    )


def blend_logit(oofs: list[np.ndarray], tests: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.ones(len(oofs), dtype=float) / len(oofs)
    return (
        sigmoid(np.mean([logit(x) for x in oofs], axis=0)),
        sigmoid(np.mean([logit(x) for x in tests], axis=0)),
        weights,
    )


def blend_grid(
    oofs: list[np.ndarray],
    tests: list[np.ndarray],
    y: np.ndarray,
    step: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.vstack(oofs)
    best_auc = -1.0
    weights = np.ones(len(oofs), dtype=float) / len(oofs)
    blend_oof = stacked.mean(axis=0)
    for w in grid_weights(len(oofs), step=step):
        pred = np.tensordot(w, stacked, axes=(0, 0))
        auc = float(roc_auc_score(y, pred))
        if auc > best_auc:
            best_auc = auc
            weights = np.array(w, dtype=float)
            blend_oof = pred
    blend_test = np.tensordot(weights, np.vstack(tests), axes=(0, 0))
    return blend_oof, blend_test, weights


def _stack_matrix(oofs: list[np.ndarray]) -> np.ndarray:
    return np.column_stack(oofs)


def stack_logistic_cv(
    oofs: list[np.ndarray],
    tests: list[np.ndarray],
    y: np.ndarray,
    *,
    seed: int = 42,
    n_splits: int = 5,
    C: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Honest stacked OOF: logistic coefficients fit inside a second CV."""
    X = _stack_matrix(oofs)
    X_test = _stack_matrix(tests)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    coefs: list[np.ndarray] = []
    intercepts: list[float] = []
    for tr_idx, va_idx in splitter.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs")
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
        test_pred += clf.predict_proba(X_test)[:, 1] / splitter.n_splits
        coefs.append(clf.coef_.ravel())
        intercepts.append(float(clf.intercept_[0]))
    mean_coef = np.mean(coefs, axis=0)
    extra = {
        "mean_coef": mean_coef.tolist(),
        "mean_intercept": float(np.mean(intercepts)),
        "C": C,
        "n_splits": n_splits,
    }
    return oof, test_pred, mean_coef, extra


def stack_ridge_cv(
    oofs: list[np.ndarray],
    tests: list[np.ndarray],
    y: np.ndarray,
    *,
    seed: int = 42,
    n_splits: int = 5,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Ridge on probabilities. Output is clipped to [0, 1] for submission."""
    X = _stack_matrix(oofs)
    X_test = _stack_matrix(tests)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    coefs: list[np.ndarray] = []
    intercepts: list[float] = []
    for tr_idx, va_idx in splitter.split(X, y):
        clf = Ridge(alpha=alpha)
        clf.fit(X[tr_idx], y[tr_idx].astype(float))
        oof[va_idx] = clf.predict(X[va_idx])
        test_pred += clf.predict(X_test) / splitter.n_splits
        coefs.append(clf.coef_.ravel())
        intercepts.append(float(clf.intercept_))
    oof = np.clip(oof, 0.0, 1.0)
    test_pred = np.clip(test_pred, 0.0, 1.0)
    mean_coef = np.mean(coefs, axis=0)
    extra = {
        "mean_coef": mean_coef.tolist(),
        "mean_intercept": float(np.mean(intercepts)),
        "alpha": alpha,
        "n_splits": n_splits,
    }
    return oof, test_pred, mean_coef, extra


BLEND_METHODS = ("mean", "rank", "logit", "grid", "stack_logistic", "stack_ridge")
