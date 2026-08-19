"""Leakage-safe leave-one-fold-out blend selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BlendResult:
    """Predictions and fold-specific choices from a LOFO grid blend."""

    oof: np.ndarray
    test: np.ndarray
    auc: float
    fold_auc: tuple[float, ...]
    fold_weights: tuple[tuple[float, ...], ...]
    weight_mean: tuple[float, ...]
    fold_values: tuple[int, ...]


def _grid_weights(n_models: int, step: int = 5) -> tuple[tuple[float, ...], ...]:
    """Return lexicographically ordered convex weights on an integer grid."""
    if n_models < 1:
        raise ValueError("at least one model is required")
    if not isinstance(step, (int, np.integer)) or isinstance(step, bool) or step <= 0:
        raise ValueError("step must be a positive integer")

    ticks = range(0, 101, int(step))
    weights = tuple(
        tuple(value / 100.0 for value in candidate)
        for candidate in product(ticks, repeat=n_models)
        if sum(candidate) == 100
    )
    if not weights:
        raise ValueError("weight grid has no valid convex weights for the requested step")
    return weights


def _as_2d_finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_inputs(
    y: np.ndarray,
    fold_ids: np.ndarray,
    oof_matrix: np.ndarray,
    test_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_array = np.asarray(y)
    folds = np.asarray(fold_ids)
    oof = _as_2d_finite("oof_matrix", oof_matrix)
    test = _as_2d_finite("test_matrix", test_matrix)

    if y_array.ndim != 1 or folds.ndim != 1:
        raise ValueError("y and fold_ids must be one-dimensional")
    if len(y_array) != len(folds) or oof.shape[1] != len(y_array):
        raise ValueError("OOF matrix, y, and fold_ids must have matching length")
    if oof.shape[0] != test.shape[0]:
        raise ValueError("test matrix must have the same number of models as OOF matrix")
    if test.shape[1] < 1:
        raise ValueError("test matrix must include at least one prediction")
    if len(np.unique(y_array)) < 2:
        raise ValueError("y must contain both classes")

    fold_values = np.unique(folds)
    if len(fold_values) < 2:
        raise ValueError("at least two folds are required for leave-one-fold-out blending")
    for fold in fold_values:
        held_out = folds == fold
        if not held_out.any():
            raise ValueError(f"fold {fold!r} has no rows")
        if len(np.unique(y_array[held_out])) < 2:
            raise ValueError(f"fold {fold!r} must contain both classes")
        if len(np.unique(y_array[~held_out])) < 2:
            raise ValueError(f"training rows for fold {fold!r} must contain both classes")
    return y_array, folds, oof, test


def lofo_grid_blend(
    *,
    y: np.ndarray,
    fold_ids: np.ndarray,
    oof_matrix: np.ndarray,
    test_matrix: np.ndarray,
    step: int = 5,
) -> BlendResult:
    """Select each blend on the other folds, then score its held-out fold.

    The selected weights for a fold never inspect either the labels or the
    predictions of that fold.  This makes the returned OOF vector appropriate
    for deciding whether a blend is genuinely complementary.
    """
    y_array, folds, oof, test = _validate_inputs(y, fold_ids, oof_matrix, test_matrix)
    candidates = _grid_weights(oof.shape[0], step)
    fold_values = tuple(np.unique(folds).tolist())
    blend_oof = np.empty(len(y_array), dtype=float)
    fold_weights: list[tuple[float, ...]] = []
    fold_auc: list[float] = []
    fold_test_predictions: list[np.ndarray] = []

    for fold in fold_values:
        held_out = folds == fold
        selection = ~held_out
        best_weight: tuple[float, ...] | None = None
        best_auc = -np.inf
        for weights in candidates:
            selection_prediction = np.asarray(weights) @ oof[:, selection]
            score = float(roc_auc_score(y_array[selection], selection_prediction))
            if (
                score > best_auc
                or (score == best_auc and (best_weight is None or weights < best_weight))
            ):
                best_auc = score
                best_weight = weights

        if best_weight is None:  # Defensive: _grid_weights always returns at least one item.
            raise ValueError("weight grid has no valid weights")
        weights_array = np.asarray(best_weight)
        held_prediction = weights_array @ oof[:, held_out]
        blend_oof[held_out] = held_prediction
        fold_weights.append(best_weight)
        fold_auc.append(float(roc_auc_score(y_array[held_out], held_prediction)))
        fold_test_predictions.append(weights_array @ test)

    weight_matrix = np.asarray(fold_weights, dtype=float)
    return BlendResult(
        oof=blend_oof,
        test=np.mean(fold_test_predictions, axis=0),
        auc=float(roc_auc_score(y_array, blend_oof)),
        fold_auc=tuple(fold_auc),
        fold_weights=tuple(fold_weights),
        weight_mean=tuple(np.mean(weight_matrix, axis=0).tolist()),
        fold_values=fold_values,
    )
