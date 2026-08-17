"""Feature construction. All feature lists and flags come from YAML."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _ratio(numer: pd.Series, denom: pd.Series, eps: float) -> pd.Series:
    return numer / (denom + eps)


def add_engineered_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Row-wise transforms only — no target statistics, no fold leakage."""
    out = df.copy()
    feat_cfg = config["features"]
    eng = feat_cfg.get("engineering", {})
    eps = float(eng.get("ratio_eps", feat_cfg.get("ratio_eps", 1e-6)))
    numeric = feat_cfg["numeric"]
    categorical = feat_cfg["categorical"]

    if eng.get("add_n_missing", False):
        miss_cols = [c for c in numeric + categorical if c in out.columns]
        out["n_missing"] = out[miss_cols].isna().sum(axis=1)

    if eng.get("add_missing_indicators", False):
        for col in numeric + categorical:
            if col in out.columns:
                out[f"{col}_is_missing"] = out[col].isna().astype("int8")

    if eng.get("add_leisure_hours", False):
        out["leisure_hours"] = out["social_media_hours"] + out["gaming_hours"]

    if eng.get("add_screen_sleep_ratio", False):
        out["screen_sleep_ratio"] = _ratio(
            out["daily_screen_time_hours"], out["sleep_hours"], eps
        )

    if eng.get("add_weekend_weekday_ratio", False):
        out["weekend_weekday_ratio"] = _ratio(
            out["weekend_screen_time"], out["daily_screen_time_hours"], eps
        )

    if eng.get("add_notif_per_open", False):
        out["notif_per_open"] = _ratio(
            out["notifications_per_day"], out["app_opens_per_day"], eps
        )

    return out


def cast_categoricals(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col in config["features"]["categorical"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def feature_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    id_col = config["competition"]["id_col"]
    target = config["competition"]["target"]
    drop = {id_col, target}
    cols = [c for c in df.columns if c not in drop]
    return cols


def transform(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = add_engineered_features(df, config)
    out = cast_categoricals(out, config)
    return out
