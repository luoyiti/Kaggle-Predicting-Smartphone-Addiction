"""Offline ranking, slice, and calibration metrics. No serving latency."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from s6e8.contracts import MISSINGNESS_COHORT_COLUMNS, SLICE_COLUMNS


def safe_roc_auc(y: np.ndarray, pred: np.ndarray, min_n: int = 20) -> float | None:
    y_np = np.asarray(y)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() < min_n:
        return None
    y_m = y_np[mask]
    if np.unique(y_m).size < 2:
        return None
    return float(roc_auc_score(y_m, p[mask]))


def expected_calibration_error(
    y: np.ndarray,
    pred: np.ndarray,
    n_bins: int = 15,
) -> float:
    y_np = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if np.isclose(hi, 1.0):
            member = (p >= lo) & (p <= hi)
        else:
            member = (p >= lo) & (p < hi)
        count = int(member.sum())
        if count == 0:
            continue
        acc = float(y_np[member].mean())
        conf = float(p[member].mean())
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def calibration_summary(y: np.ndarray, pred: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    p = np.clip(np.asarray(pred, dtype=float), 1e-6, 1.0 - 1e-6)
    y_np = np.asarray(y)
    return {
        "brier": float(brier_score_loss(y_np, p)),
        "ece": expected_calibration_error(y_np, p, n_bins=n_bins),
        "mean_pred": float(np.mean(p)),
        "positive_rate": float(np.mean(y_np)),
        "auc": float(roc_auc_score(y_np, p)),
    }


def _group_key(value: Any) -> str:
    if value is None:
        return "__NA__"
    try:
        if pd.isna(value):
            return "__NA__"
    except (TypeError, ValueError):
        pass
    return str(value)


def slice_auc_table(
    y: np.ndarray,
    pred: np.ndarray,
    groups: pd.Series,
    *,
    min_n: int = 50,
) -> dict[str, dict[str, Any]]:
    y_np = np.asarray(y)
    p = np.asarray(pred, dtype=float)
    out: dict[str, dict[str, Any]] = {}
    keys = groups.map(_group_key)
    for key, idx in keys.groupby(keys).groups.items():
        n = int(len(idx))
        row: dict[str, Any] = {"n": n, "positive_rate": float(np.mean(y_np[list(idx)]))}
        auc = safe_roc_auc(y_np[list(idx)], p[list(idx)], min_n=min_n)
        row["auc"] = auc
        out[str(key)] = row
    return out


def missingness_cohorts(df: pd.DataFrame) -> pd.Series:
    parts = [c for c in MISSINGNESS_COHORT_COLUMNS if c in df.columns]
    if not parts:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        any_missing = df[numeric].isna().any(axis=1) if numeric else pd.Series(False, index=df.index)
        return pd.Series(np.where(any_missing, "any_missing", "complete"), index=df.index)
    daily = df[parts[0]].isna() if parts[0] in df.columns else pd.Series(False, index=df.index)
    any_missing = df[parts].isna().any(axis=1)
    labels = np.where(daily, "daily_screen_missing", np.where(any_missing, "other_usage_missing", "complete"))
    return pd.Series(labels, index=df.index)


def hard_band_mask(pred: np.ndarray, lo: float = 0.3, hi: float = 0.7) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    return (p > lo) & (p < hi)


def slice_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    frame: pd.DataFrame,
    *,
    min_n: int = 50,
    extra_columns: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Cohort AUC for categoricals, missingness, and the uncertain-probability band."""
    payload: dict[str, Any] = {
        "overall_auc": safe_roc_auc(y, pred, min_n=min_n),
        "n": int(len(y)),
    }
    columns = list(SLICE_COLUMNS) + list(extra_columns or ())
    for col in columns:
        if col not in frame.columns:
            continue
        payload[col] = slice_auc_table(y, pred, frame[col], min_n=min_n)
    payload["missingness"] = slice_auc_table(y, pred, missingness_cohorts(frame), min_n=min_n)
    band = hard_band_mask(pred)
    payload["hard_band"] = {
        "lo": 0.3,
        "hi": 0.7,
        "n": int(band.sum()),
        "positive_rate": float(np.mean(np.asarray(y)[band])) if band.any() else None,
        "auc": safe_roc_auc(np.asarray(y)[band], np.asarray(pred)[band], min_n=min_n)
        if band.any()
        else None,
    }
    return payload
