"""OOF error-band analysis. Does not train; reads saved predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_NUMERIC = (
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
)


def _auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    if mask.sum() < 2:
        return None
    y = y[mask]
    scores = scores[mask]
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def band_mask(pred: np.ndarray, lo: float, hi: float) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    return (p > float(lo)) & (p < float(hi))


def summarize_slice(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y)
    pred = np.asarray(pred, dtype=float)
    n = int(mask.sum())
    out: dict[str, Any] = {
        "n": n,
        "frac": float(n / len(mask)) if len(mask) else 0.0,
        "pos_rate": None,
        "auc": None,
    }
    if n == 0:
        return out
    y_s = y[mask]
    p_s = pred[mask]
    out["pos_rate"] = float(np.mean(y_s))
    out["auc"] = _auc(y_s, p_s)
    return out


def column_aucs(
    frame: pd.DataFrame,
    y: np.ndarray,
    columns: list[str],
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        scores[column] = _auc(y, frame[column].to_numpy(dtype=float))
    return scores


def analyze_error_band(
    oof: pd.DataFrame,
    *,
    pred_col: str = "pred",
    y_col: str = "addicted_label",
    id_col: str = "id",
    lo: float = 0.3,
    hi: float = 0.7,
    features: pd.DataFrame | None = None,
    feature_columns: list[str] | None = None,
    compare: pd.DataFrame | None = None,
    compare_name: str | None = None,
) -> dict[str, Any]:
    """Summarize the uncertain-probability band of one OOF table.

    ``compare``, if given, is aligned by ``id_col`` and scored on the *primary*
    model's hard-band rows (does the partner rank those rows better?).
    """
    required = {id_col, y_col, pred_col}
    missing = required - set(oof.columns)
    if missing:
        raise KeyError(f"OOF is missing columns: {sorted(missing)}")
    work = oof[[id_col, y_col, pred_col]].copy()
    y = work[y_col].to_numpy()
    pred = work[pred_col].to_numpy(dtype=float)
    mask = band_mask(pred, lo, hi)
    report: dict[str, Any] = {
        "n": int(len(work)),
        "lo": float(lo),
        "hi": float(hi),
        "auc_all": _auc(y, pred),
        "band": summarize_slice(y, pred, mask),
        "outside": summarize_slice(y, pred, ~mask),
    }
    if features is not None:
        cols = list(feature_columns or DEFAULT_NUMERIC)
        feat = features.copy()
        if id_col not in feat.columns:
            raise KeyError(f"feature frame is missing {id_col!r}")
        merged = work.merge(feat[[id_col, *[c for c in cols if c in feat.columns]]], on=id_col, how="left")
        band_frame = merged.loc[mask]
        report["feature_auc_in_band"] = column_aucs(
            band_frame, band_frame[y_col].to_numpy(), cols
        )
        residual = band_frame[y_col].to_numpy(dtype=float) - band_frame[pred_col].to_numpy(dtype=float)
        corr: dict[str, float | None] = {}
        for column in cols:
            if column not in band_frame.columns:
                continue
            series = pd.to_numeric(band_frame[column], errors="coerce")
            valid = series.notna()
            if int(valid.sum()) < 3:
                corr[column] = None
                continue
            corr[column] = float(np.corrcoef(series[valid], residual[valid.to_numpy()])[0, 1])
        report["residual_corr_in_band"] = corr
    if compare is not None:
        name = compare_name or "compare"
        if id_col not in compare.columns or pred_col not in compare.columns:
            raise KeyError("compare OOF needs id and pred columns")
        joined = work.merge(
            compare[[id_col, pred_col]].rename(columns={pred_col: "compare_pred"}),
            on=id_col,
            how="inner",
        )
        if len(joined) != len(work):
            report["compare_align_n"] = int(len(joined))
        cmp_pred = joined["compare_pred"].to_numpy(dtype=float)
        cmp_y = joined[y_col].to_numpy()
        primary_mask = band_mask(joined[pred_col].to_numpy(dtype=float), lo, hi)
        report["compare"] = {
            "name": name,
            "auc_all": _auc(cmp_y, cmp_pred),
            "auc_on_primary_band": _auc(cmp_y[primary_mask], cmp_pred[primary_mask])
            if primary_mask.any()
            else None,
            "own_band": summarize_slice(cmp_y, cmp_pred, band_mask(cmp_pred, lo, hi)),
        }
    return report
