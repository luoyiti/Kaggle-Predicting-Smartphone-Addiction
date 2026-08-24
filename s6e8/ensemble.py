"""Blend and stack saved OOF / test probability vectors."""

from __future__ import annotations

from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BlendMethod = Literal["mean", "rank", "logit", "grid", "auc_weighted"]


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy() / (len(x) + 1.0)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def grid_weights(n: int, step: int = 5) -> list[tuple[float, ...]]:
    ticks = list(range(0, 101, step))
    out: list[tuple[float, ...]] = []
    for combo in product(ticks, repeat=n):
        if sum(combo) != 100:
            continue
        if all(v == 0 for v in combo):
            continue
        out.append(tuple(v / 100.0 for v in combo))
    return out


def auc_weights(aucs: list[float]) -> np.ndarray:
    raw = np.clip(np.asarray(aucs, dtype=float) - 0.5, 1e-6, None)
    return raw / raw.sum()


def blend_predictions(
    oofs: list[np.ndarray],
    tests: list[np.ndarray],
    y: np.ndarray,
    *,
    method: BlendMethod = "mean",
    component_aucs: list[float] | None = None,
) -> dict[str, Any]:
    stacked = np.vstack(oofs)
    test_stacked = np.vstack(tests)
    n = stacked.shape[0]
    if method == "mean":
        weights = np.ones(n) / n
        blend_oof = stacked.mean(axis=0)
        blend_test = test_stacked.mean(axis=0)
    elif method == "rank":
        weights = np.ones(n) / n
        blend_oof = np.mean([rank01(x) for x in oofs], axis=0)
        blend_test = np.mean([rank01(x) for x in tests], axis=0)
    elif method == "logit":
        weights = np.ones(n) / n
        blend_oof = sigmoid(np.mean([logit(x) for x in oofs], axis=0))
        blend_test = sigmoid(np.mean([logit(x) for x in tests], axis=0))
    elif method == "auc_weighted":
        if not component_aucs:
            component_aucs = [float(roc_auc_score(y, x)) for x in oofs]
        weights = auc_weights(component_aucs)
        blend_oof = np.tensordot(weights, stacked, axes=(0, 0))
        blend_test = np.tensordot(weights, test_stacked, axes=(0, 0))
    elif method == "grid":
        best_auc = -1.0
        weights = np.ones(n) / n
        blend_oof = stacked.mean(axis=0)
        for w in grid_weights(n):
            pred = np.tensordot(w, stacked, axes=(0, 0))
            auc = float(roc_auc_score(y, pred))
            if auc > best_auc:
                best_auc = auc
                weights = np.array(w, dtype=float)
                blend_oof = pred
        blend_test = np.tensordot(weights, test_stacked, axes=(0, 0))
    else:
        raise ValueError(f"Unknown blend method {method!r}")

    auc = float(roc_auc_score(y, blend_oof))
    corr = np.corrcoef(stacked)
    return {
        "method": method,
        "weights": weights.astype(float),
        "oof": np.asarray(blend_oof, dtype=float),
        "test": np.asarray(blend_test, dtype=float),
        "oof_auc": auc,
        "oof_corr": corr,
    }


def logistic_stack(
    oofs: list[np.ndarray],
    tests: list[np.ndarray],
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    C: float = 1.0,
) -> dict[str, Any]:
    """Fold-safe logistic stacker on OOF columns; test = average of fold models."""
    X = np.column_stack(oofs)
    X_test = np.column_stack(tests)
    y_np = np.asarray(y)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    stacked_oof = np.zeros(len(y_np), dtype=float)
    stacked_test = np.zeros(len(X_test), dtype=float)
    coefs: list[np.ndarray] = []
    intercepts: list[float] = []
    for tr_idx, va_idx in splitter.split(X, y_np):
        clf = LogisticRegression(
            C=C, solver="lbfgs", max_iter=400, random_state=seed
        )
        clf.fit(X[tr_idx], y_np[tr_idx])
        stacked_oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
        stacked_test += clf.predict_proba(X_test)[:, 1] / n_splits
        coefs.append(clf.coef_.ravel())
        intercepts.append(float(clf.intercept_[0]))
    auc = float(roc_auc_score(y_np, stacked_oof))
    mean_coef = np.mean(np.vstack(coefs), axis=0)
    return {
        "method": "logistic_stack",
        "oof": stacked_oof,
        "test": stacked_test,
        "oof_auc": auc,
        "mean_coef": mean_coef,
        "mean_intercept": float(np.mean(intercepts)),
        "C": C,
        "n_splits": n_splits,
        "seed": seed,
    }
