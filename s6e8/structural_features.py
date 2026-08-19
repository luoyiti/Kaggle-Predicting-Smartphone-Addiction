"""Configuration-driven, row-local structural feature transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SCREEN_COMPONENT_COLUMNS = (
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
)


def _required_columns(df: pd.DataFrame, columns: list[str], block: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{block} columns are missing: {missing}")


def canonical_numeric_value(value: object, decimals: int, missing_token: str) -> str:
    """Return a deterministic string representation for a scalar numeric value."""
    if pd.isna(value):
        return missing_token
    number = float(value)
    if decimals == 0:
        return str(int(round(number)))
    return f"{number:.{decimals}f}"


def add_exact_categorical_features(
    df: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Copy configured numeric values into deterministic categorical string keys."""
    out = df.copy()
    block = config["features"].get("exact_categorical") or {}
    if not bool(block.get("enabled", False)):
        return out
    columns = block.get("columns", "auto_numeric")
    if columns == "auto_numeric":
        columns = list(config["features"]["numeric"])
    columns = list(columns)
    _required_columns(out, columns, "exact_categorical")
    suffix = str(block.get("suffix", "__exact"))
    missing_token = str(block.get("missing_token", "__MISSING__"))
    decimal_places = dict(block.get("decimal_places") or {})
    for column in columns:
        decimals = int(decimal_places.get(column, 8))
        out[f"{column}{suffix}"] = out[column].map(
            lambda value, c=column, d=decimals: (
                f"{c}={canonical_numeric_value(value, d, missing_token)}"
            )
        )
    return out


def add_screen_budget_features(
    df: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Add screen-use budget arithmetic while retaining complete/observed sums."""
    out = df.copy()
    block = config["features"].get("screen_budget") or {}
    if not bool(block.get("enabled", False)):
        return out
    required = [
        "daily_screen_time_hours",
        *SCREEN_COMPONENT_COLUMNS,
        "weekend_screen_time",
        "sleep_hours",
    ]
    _required_columns(out, required, "screen_budget")
    components = out[list(SCREEN_COMPONENT_COLUMNS)]
    complete = components.sum(axis=1, min_count=len(SCREEN_COMPONENT_COLUMNS))
    observed = components.sum(axis=1, min_count=1)
    count = components.notna().sum(axis=1).astype("int8")
    daily = out["daily_screen_time_hours"]

    out["screen_component_sum_complete"] = complete
    out["screen_component_sum_observed"] = observed
    out["screen_component_count"] = count
    out["screen_remainder_complete"] = daily - complete
    out["screen_remainder_observed"] = daily - observed
    daily_nonzero = daily.replace(0, np.nan)
    out["screen_component_share_complete"] = complete.div(daily_nonzero)
    out["screen_remainder_share_complete"] = (daily - complete).div(daily_nonzero)
    out["weekend_minus_component_sum"] = out["weekend_screen_time"] - complete
    out["weekend_minus_remainder"] = out["weekend_screen_time"] - (daily - complete)
    out["awake_non_screen_hours"] = 24.0 - out["sleep_hours"] - daily

    tolerance = float(block.get("tolerance", 1e-9))
    remainder = out["screen_remainder_complete"]
    out["screen_budget_boundary"] = remainder.abs().le(tolerance).where(
        remainder.notna()
    ).astype("Int8")
    out["screen_budget_violation"] = remainder.lt(-tolerance).where(
        remainder.notna()
    ).astype("Int8")
    return out


def add_decimal_lattice_features(
    df: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Expose only each configured value's fractional part and first decimal digit."""
    out = df.copy()
    block = config["features"].get("decimal_lattice") or {}
    if not bool(block.get("enabled", False)):
        return out
    columns = block.get("columns", "auto_numeric")
    if columns == "auto_numeric":
        columns = list(config["features"]["numeric"])
    columns = list(columns)
    _required_columns(out, columns, "decimal_lattice")
    for column in columns:
        values = out[column]
        out[f"{column}__fraction"] = values - np.floor(values)
        out[f"{column}__first_decimal"] = (
            np.floor(values * 10)
            .mod(10)
            .where(values.notna())
            .astype("Int8")
        )
    return out


def add_structural_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply all enabled row-local structural feature blocks in config order."""
    out = add_exact_categorical_features(df, config)
    out = add_screen_budget_features(out, config)
    out = add_decimal_lattice_features(out, config)
    return out
