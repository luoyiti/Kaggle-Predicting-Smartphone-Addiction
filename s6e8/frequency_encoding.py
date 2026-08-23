"""Fold-safe exact-value frequency encoding.

Unlike target encoding this transform never uses the label: it maps each
rounded numeric value to how often it appeared in the *training fold*.
Unseen values map to 0. Missing inputs stay missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from s6e8.target_encoding import value_keys


def parse_frequency_config(config: dict[str, Any]) -> dict[str, Any] | None:
    feat = config.get("features") or {}
    raw = feat.get("frequency_encoding")
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
        "round_decimals": round_decimals,
        "suffix": str(raw.get("suffix", "_freq")),
        "log1p": bool(raw.get("log1p", True)),
        "unseen_value": float(raw.get("unseen_value", 0.0)),
    }


def freq_feature_name(column: str, suffix: str = "_freq") -> str:
    return f"{column}{suffix}"


@dataclass
class ExactValueFrequencyEncoder:
    columns: list[str]
    round_decimals: int | None = 2
    suffix: str = "_freq"
    log1p: bool = True
    unseen_value: float = 0.0
    counts: dict[str, pd.Series] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> ExactValueFrequencyEncoder:
        self.counts = {}
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"frequency column {col!r} is missing from the fold frame")
            keys = value_keys(X[col], self.round_decimals)
            self.counts[col] = keys.value_counts(dropna=True).astype("float64")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            name = freq_feature_name(col, self.suffix)
            keys = value_keys(X[col], self.round_decimals)
            mapped = keys.map(self.counts[col]).astype("float64")
            mapped = mapped.fillna(self.unseen_value)
            mapped = mapped.mask(X[col].isna(), np.nan)
            if self.log1p:
                mapped = np.log1p(mapped)
            out[name] = mapped.astype("float64")
        return out

    def fold_stats(self, X_val: pd.DataFrame) -> dict[str, Any]:
        payload: dict[str, Any] = {"columns": {}}
        for col in self.columns:
            keys = value_keys(X_val[col], self.round_decimals)
            mapped = keys.map(self.counts[col])
            observed = X_val[col].notna()
            train_counts = self.counts[col]
            payload["columns"][col] = {
                "n_train_keys": int(len(train_counts)),
                "train_median_count": float(train_counts.median()) if len(train_counts) else 0.0,
                "n_val_observed": int(observed.sum()),
                "n_val_unseen": int((observed & mapped.isna()).sum()),
            }
        return payload


def apply_fold_frequency_encoding(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    freq_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    encoder = ExactValueFrequencyEncoder(
        columns=list(freq_cfg["columns"]),
        round_decimals=freq_cfg.get("round_decimals"),
        suffix=str(freq_cfg.get("suffix", "_freq")),
        log1p=bool(freq_cfg.get("log1p", True)),
        unseen_value=float(freq_cfg.get("unseen_value", 0.0)),
    )
    encoder.fit(X_train)
    return (
        encoder.transform(X_train),
        encoder.transform(X_valid),
        encoder.transform(X_test),
        encoder.fold_stats(X_valid),
    )
