"""Post-hoc probability calibration fit on OOF, applied to test predictions.

Isotonic/Platt can improve Brier/ECE. ROC-AUC is usually unchanged (isotonic
can create ties). Promotion must reject a material AUC drop.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression

from s6e8.metrics import calibration_summary, safe_roc_auc

CalibMethod = Literal["isotonic", "platt"]


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return model


def _fit_platt(x: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=400, random_state=seed)
    clf.fit(np.asarray(x, dtype=float).reshape(-1, 1), np.asarray(y))
    return clf


def apply_calibrator(model: Any, x: np.ndarray, method: CalibMethod) -> np.ndarray:
    x_np = np.asarray(x, dtype=float)
    if method == "isotonic":
        return np.asarray(model.predict(x_np), dtype=float)
    proba = model.predict_proba(x_np.reshape(-1, 1))[:, 1]
    return np.asarray(proba, dtype=float)


def calibrate_with_oof_cv(
    oof: np.ndarray,
    y: np.ndarray,
    test: np.ndarray,
    *,
    method: CalibMethod = "isotonic",
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Honest calibrated OOF via inner StratifiedKFold; test = mean of fold calibrators."""
    if method not in {"isotonic", "platt"}:
        raise ValueError(f"Unsupported calibration method {method!r}")
    oof = np.asarray(oof, dtype=float)
    y = np.asarray(y)
    test = np.asarray(test, dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cal_oof = np.zeros_like(oof, dtype=float)
    cal_test = np.zeros_like(test, dtype=float)
    for tr_idx, va_idx in splitter.split(oof.reshape(-1, 1), y):
        if method == "isotonic":
            model = _fit_isotonic(oof[tr_idx], y[tr_idx])
        else:
            model = _fit_platt(oof[tr_idx], y[tr_idx], seed)
        cal_oof[va_idx] = apply_calibrator(model, oof[va_idx], method)
        cal_test += apply_calibrator(model, test, method) / n_splits

    raw = calibration_summary(y, oof)
    fitted = calibration_summary(y, cal_oof)
    return {
        "method": method,
        "n_splits": n_splits,
        "seed": seed,
        "oof": cal_oof,
        "test": cal_test,
        "raw": raw,
        "calibrated": fitted,
        "auc_delta": float((fitted["auc"] or 0.0) - (raw["auc"] or 0.0)),
        "ece_delta": float(fitted["ece"] - raw["ece"]),
        "oof_auc": safe_roc_auc(y, cal_oof),
    }
