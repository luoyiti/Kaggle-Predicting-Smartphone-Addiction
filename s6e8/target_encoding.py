"""Fold-safe exact-value target encoding.

Row-wise feature code in ``s6e8.features`` must never see the target.
These maps are fit **only** on the current training fold, then applied to
that fold's train (leave-one-out), validation, and test rows.

Validation and test labels are never used. A value unseen (or below
``min_count``) in the training fold falls back to the fold prior.
Missing inputs stay missing so LightGBM can keep native NaN handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def parse_exact_te_config(config: dict[str, Any]) -> dict[str, Any] | None:
    feat = config.get("features") or {}
    eng = feat.get("engineering") or {}
    raw = eng.get("exact_target_encoding")
    if not raw:
        return None
    if not bool(raw.get("enabled", True)):
        return None
    columns = [str(c) for c in (raw.get("columns") or [])]
    if not columns:
        return None
    round_decimals = raw.get("round_decimals", 2)
    if round_decimals is not None:
        round_decimals = int(round_decimals)
    return {
        "columns": columns,
        "smoothing": float(raw.get("smoothing", 20.0)),
        "min_count": int(raw.get("min_count", 5)),
        "round_decimals": round_decimals,
        "suffix": str(raw.get("suffix", "_exact_te")),
        "leave_one_out_train": bool(raw.get("leave_one_out_train", True)),
    }


def te_feature_name(column: str, suffix: str = "_exact_te") -> str:
    return f"{column}{suffix}"


def value_keys(series: pd.Series, decimals: int | None) -> pd.Series:
    """Map a numeric series to exact-value keys (optional decimal rounding)."""
    if decimals is None:
        return series
    return series.round(int(decimals))


@dataclass
class ExactValueTargetEncoder:
    columns: list[str]
    smoothing: float = 20.0
    min_count: int = 5
    round_decimals: int | None = 2
    suffix: str = "_exact_te"
    leave_one_out_train: bool = True
    prior: float = 0.0
    sums: dict[str, pd.Series] = field(default_factory=dict)
    counts: dict[str, pd.Series] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> ExactValueTargetEncoder:
        y_s = pd.Series(np.asarray(y, dtype=float), index=X.index)
        if not np.isfinite(y_s).any():
            raise ValueError("exact-value TE fit requires at least one finite label")
        self.prior = float(np.nanmean(y_s.to_numpy()))
        self.sums = {}
        self.counts = {}
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"exact TE column {col!r} is missing from the fold frame")
            keys = value_keys(X[col], self.round_decimals)
            grouped = y_s.groupby(keys, dropna=True)
            self.sums[col] = grouped.sum()
            self.counts[col] = grouped.count().astype("float64")
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        *,
        leave_one_out: bool = False,
    ) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            name = te_feature_name(col, self.suffix)
            out[name] = self._transform_column(X[col], y, leave_one_out=leave_one_out, col=col)
        return out

    def _transform_column(
        self,
        series: pd.Series,
        y: np.ndarray | pd.Series | None,
        *,
        leave_one_out: bool,
        col: str,
    ) -> pd.Series:
        keys = value_keys(series, self.round_decimals)
        mapped_sum = keys.map(self.sums[col])
        mapped_count = keys.map(self.counts[col])
        if leave_one_out:
            if y is None:
                raise ValueError("leave-one-out exact-value TE requires the training labels")
            y_s = pd.Series(np.asarray(y, dtype=float), index=series.index)
            mapped_sum = mapped_sum - y_s
            mapped_count = mapped_count - 1.0

        count_filled = mapped_count.fillna(0.0)
        sum_filled = mapped_sum.fillna(0.0)
        denom = count_filled + self.smoothing
        with np.errstate(divide="ignore", invalid="ignore"):
            encoded = (sum_filled + self.smoothing * self.prior) / denom
        encoded = encoded.replace([np.inf, -np.inf], np.nan)

        fallback = mapped_count.isna() | (mapped_count < self.min_count) | encoded.isna()
        encoded = encoded.mask(fallback, self.prior)
        encoded = encoded.mask(series.isna(), np.nan)
        return encoded.astype("float64")

    def fold_stats(self, X_val: pd.DataFrame) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prior": self.prior,
            "smoothing": self.smoothing,
            "min_count": self.min_count,
            "columns": {},
        }
        for col in self.columns:
            keys = value_keys(X_val[col], self.round_decimals)
            mapped_count = keys.map(self.counts[col])
            observed = X_val[col].notna()
            n_obs = int(observed.sum())
            train_counts = self.counts[col]
            payload["columns"][col] = {
                "n_train_keys": int(len(train_counts)),
                "train_median_count": float(train_counts.median()) if len(train_counts) else 0.0,
                "train_min_count": float(train_counts.min()) if len(train_counts) else 0.0,
                "n_val_observed": n_obs,
                "n_val_unseen": int((observed & mapped_count.isna()).sum()),
                "n_val_rare": int(
                    (observed & mapped_count.notna() & (mapped_count < self.min_count)).sum()
                ),
            }
        return payload


def apply_fold_target_encoding(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    te_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit on the training fold only; encode train/valid/test for one CV fold."""
    encoder = ExactValueTargetEncoder(
        columns=list(te_cfg["columns"]),
        smoothing=float(te_cfg["smoothing"]),
        min_count=int(te_cfg["min_count"]),
        round_decimals=te_cfg.get("round_decimals"),
        suffix=str(te_cfg.get("suffix", "_exact_te")),
        leave_one_out_train=bool(te_cfg.get("leave_one_out_train", True)),
    )
    encoder.fit(X_train, y_train)
    train_out = encoder.transform(
        X_train, y_train, leave_one_out=encoder.leave_one_out_train
    )
    valid_out = encoder.transform(X_valid, leave_one_out=False)
    test_out = encoder.transform(X_test, leave_one_out=False)
    return train_out, valid_out, test_out, encoder.fold_stats(X_valid)
