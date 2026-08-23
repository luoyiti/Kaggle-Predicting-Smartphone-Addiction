"""Feature construction. All feature lists and flags come from YAML."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_STRONG_USAGE_COLS = (
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
)
DEFAULT_OTHER_SCREEN_TOTAL = "daily_screen_time_hours"
DEFAULT_OTHER_SCREEN_PARTS = (
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
)
DEFAULT_INTERACTION_PAIRS = (
    ("daily_screen_time_hours", "sleep_hours"),
    ("social_media_hours", "gaming_hours"),
    ("daily_screen_time_hours", "weekend_screen_time"),
    ("social_media_hours", "work_study_hours"),
    ("notifications_per_day", "app_opens_per_day"),
)
DEFAULT_ORDINAL_MAPS = {
    "stress_level": {"Low": 0, "Medium": 1, "High": 2},
    "academic_work_impact": {"No": 0, "Yes": 1},
    "gender": {"Male": 0, "Female": 1, "Other": 2},
}
DEFAULT_QUANTILE_BIN_COLUMNS = (
    "daily_screen_time_hours",
    "sleep_hours",
    "weekend_screen_time",
)
MISSING_CATEGORY_LEVEL = "__NA__"


def _ratio(numer: pd.Series, denom: pd.Series, eps: float) -> pd.Series:
    return numer / (denom + eps)


def _finite_median(s: pd.Series, fallback: float = 1.0) -> float:
    med = float(s.median())
    if not np.isfinite(med) or med == 0.0:
        return fallback
    return med


def _row_scaled_reduce(df: pd.DataFrame, cols: list[str], how: str, eps: float) -> pd.Series:
    parts = []
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"strong_usage column {col!r} is missing")
        scale = _finite_median(df[col], fallback=1.0)
        parts.append(df[col] / max(scale, eps))
    stacked = pd.concat(parts, axis=1)
    if how == "mean":
        return stacked.mean(axis=1)
    if how == "max":
        return stacked.max(axis=1)
    raise ValueError(f"Unknown row reduce {how!r}")


def add_engineered_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Row-wise transforms only — no target statistics, no fold leakage.

    Exact-value target encoding is applied inside CV in ``s6e8.target_encoding``.
    """
    out = df.copy()
    feat_cfg = config["features"]
    eng = feat_cfg.get("engineering", {}) or {}
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

    if eng.get("add_other_screen_hours", False):
        spec = eng.get("other_screen") or {}
        total = spec.get("total", DEFAULT_OTHER_SCREEN_TOTAL)
        parts = list(spec.get("parts") or DEFAULT_OTHER_SCREEN_PARTS)
        residual = out[total]
        for part in parts:
            residual = residual - out[part]
        out["other_screen_hours"] = residual.clip(lower=0)

    if eng.get("add_component_sum", False):
        spec = eng.get("other_screen") or {}
        parts = list(spec.get("parts") or DEFAULT_OTHER_SCREEN_PARTS)
        summed = out[parts[0]]
        for part in parts[1:]:
            summed = summed + out[part]
        out["component_sum"] = summed

    if eng.get("add_screen_imputed_weekend", False):
        daily = out["daily_screen_time_hours"]
        weekend = out["weekend_screen_time"]
        ratio = float((weekend / daily.replace(0, np.nan)).median())
        if not np.isfinite(ratio) or ratio <= 0:
            ratio = 1.24
        out["screen_imputed_weekend"] = daily.fillna(weekend / ratio)

    strong_cols = list(eng.get("strong_usage_cols") or DEFAULT_STRONG_USAGE_COLS)
    if eng.get("add_strong3_row_mean", False):
        out["strong3_row_mean"] = _row_scaled_reduce(out, strong_cols, "mean", eps)
    if eng.get("add_strong3_row_max", False):
        out["strong3_row_max"] = _row_scaled_reduce(out, strong_cols, "max", eps)

    if eng.get("add_or_usage_score", False):
        thresholds = dict(eng.get("or_usage_thresholds") or {})
        if not thresholds:
            thresholds = {
                "daily_screen_time_hours": 8.0,
                "social_media_hours": 4.0,
                "weekend_screen_time": 9.92,
            }
        terms = []
        for col, threshold in thresholds.items():
            if col not in out.columns:
                raise KeyError(f"or_usage column {col!r} is missing")
            terms.append(out[col] / float(threshold))
        out["or_usage_score"] = pd.concat(terms, axis=1).max(axis=1)

    if eng.get("add_leisure_work_ratio", False):
        leisure = out["social_media_hours"] + out["gaming_hours"]
        out["leisure_work_ratio"] = _ratio(leisure, out["work_study_hours"], eps)

    if eng.get("add_interactions", False):
        pairs = eng.get("interaction_pairs") or DEFAULT_INTERACTION_PAIRS
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"interaction pair must have two columns, got {pair!r}")
            left, right = str(pair[0]), str(pair[1])
            if left not in out.columns or right not in out.columns:
                raise KeyError(f"interaction columns {(left, right)} missing")
            out[f"{left}_x_{right}"] = out[left] * out[right]

    if eng.get("add_ordinal_cats", False):
        maps = eng.get("ordinal_maps") or DEFAULT_ORDINAL_MAPS
        suffix = str(eng.get("ordinal_suffix", "_ord"))
        for col, mapping in maps.items():
            if col not in out.columns:
                continue
            out[f"{col}{suffix}"] = out[col].map(mapping).astype("float64")

    if eng.get("add_quantile_bins", False):
        cols = list(eng.get("quantile_bin_columns") or DEFAULT_QUANTILE_BIN_COLUMNS)
        n_bins = int(eng.get("quantile_bins", 5))
        suffix = str(eng.get("quantile_suffix", "_qbin"))
        for col in cols:
            if col not in out.columns:
                raise KeyError(f"quantile bin column {col!r} is missing")
            try:
                binned = pd.qcut(out[col], q=n_bins, labels=False, duplicates="drop")
            except ValueError:
                binned = pd.Series(np.nan, index=out.index)
            out[f"{col}{suffix}"] = pd.Categorical(binned)

    if eng.get("encode_missing_as_category", False):
        fill = str(eng.get("missing_category_level", MISSING_CATEGORY_LEVEL))
        for col in categorical:
            if col in out.columns:
                as_str = out[col].astype("string")
                out[col] = as_str.fillna(fill)

    return out


def cast_categoricals(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col in config["features"]["categorical"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    eng = (config.get("features") or {}).get("engineering") or {}
    if eng.get("add_quantile_bins", False):
        suffix = str(eng.get("quantile_suffix", "_qbin"))
        for col in out.columns:
            if str(col).endswith(suffix) and col not in config["features"]["categorical"]:
                out[col] = out[col].astype("category")
    return out


def feature_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    id_col = config["competition"]["id_col"]
    target = config["competition"]["target"]
    extra_drop = config["features"].get("drop") or []
    drop = {id_col, target, *extra_drop}
    cols = [c for c in df.columns if c not in drop]
    return cols


def categorical_feature_names(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Columns LightGBM/CatBoost should treat as categorical (dtype or YAML list)."""
    named = [c for c in config["features"]["categorical"] if c in df.columns]
    typed = [c for c in df.columns if str(df[c].dtype) == "category" and c not in named]
    return named + typed


def transform(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = add_engineered_features(df, config)
    out = cast_categoricals(out, config)
    return out
