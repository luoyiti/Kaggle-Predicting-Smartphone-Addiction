"""Configuration-driven, row-local structural feature transforms.

These transforms never see the label. Exact-value *target* statistics stay in
``s6e8.target_encoding`` / ``s6e8.frequency_encoding`` and run inside CV.
"""

from __future__ import annotations

import hashlib
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


def _resolve_column_list(config: dict[str, Any], block: dict[str, Any]) -> list[str]:
    columns = block.get("columns", "auto_numeric")
    if columns == "auto_numeric":
        return list(config["features"]["numeric"])
    return [str(c) for c in columns]


def canonical_numeric_value(value: object, decimals: int, missing_token: str) -> str:
    """Return a deterministic string representation for a scalar numeric value."""
    if pd.isna(value):
        return missing_token
    number = float(value)
    if decimals <= 0:
        return str(int(round(number)))
    return f"{number:.{decimals}f}"


def format_exact_keys(series: pd.Series, decimals: int, missing_token: str) -> pd.Series:
    """Vectorized exact-value keys. Missing stays ``missing_token``."""
    mask = series.isna()
    out = pd.Series(missing_token, index=series.index, dtype="object")
    if int(mask.all()):
        return out
    values = series.to_numpy(dtype=float, na_value=np.nan)
    finite = ~mask.to_numpy()
    if decimals <= 0:
        formatted = np.round(values[finite]).astype(np.int64).astype(str)
    else:
        formatted = np.char.mod(f"%.{int(decimals)}f", values[finite])
    out.iloc[np.flatnonzero(finite)] = formatted
    return out


def _stable_bucket(text: str, n_bins: int) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return str(int(digest, 16) % int(n_bins))


def _joint_pairs(block: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = block.get("joint_pairs") or []
    resolved: list[tuple[str, str]] = []
    for pair in pairs:
        columns = [str(column) for column in pair]
        if len(columns) != 2:
            raise ValueError(
                "exact_categorical.joint_pairs entries must be length-2 column lists, "
                f"got {columns!r}"
            )
        resolved.append((columns[0], columns[1]))
    return resolved


def joint_exact_categorical_column_names(config: dict[str, Any]) -> list[str]:
    """Column names for explicit pairwise exact-value tokens (notif|app identity)."""
    block = config["features"].get("exact_categorical") or {}
    if not bool(block.get("enabled", False)):
        return []
    suffix = str(block.get("joint_suffix", "__joint"))
    return [f"{left}__{right}{suffix}" for left, right in _joint_pairs(block)]


def exact_categorical_column_names(config: dict[str, Any]) -> list[str]:
    block = config["features"].get("exact_categorical") or {}
    if not bool(block.get("enabled", False)):
        return []
    suffix = str(block.get("suffix", "__exact"))
    names = [f"{column}{suffix}" for column in _resolve_column_list(config, block)]
    names.extend(joint_exact_categorical_column_names(config))
    return names


def lattice_categorical_column_names(config: dict[str, Any]) -> list[str]:
    block = config["features"].get("decimal_lattice") or {}
    if not bool(block.get("enabled", False)):
        return []
    if not bool(block.get("as_categorical", False)):
        return []
    return [
        f"{column}__first_decimal"
        for column in _resolve_column_list(config, block)
    ]


def add_exact_categorical_features(
    df: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Copy configured numeric values into deterministic categorical string keys.

    CatBoost ordered CTRs can split on exact playground identities that a
    monotone numeric split misses. Prefix the column name so vocabularies
    do not collide across features.
    """
    out = df.copy()
    block = config["features"].get("exact_categorical") or {}
    if not bool(block.get("enabled", False)):
        return out
    columns = _resolve_column_list(config, block)
    _required_columns(out, columns, "exact_categorical")
    suffix = str(block.get("suffix", "__exact"))
    missing_token = str(block.get("missing_token", "__MISSING__"))
    decimal_places = dict(block.get("decimal_places") or {})
    for column in columns:
        decimals = int(decimal_places.get(column, 8))
        keys = format_exact_keys(out[column], decimals, missing_token)
        labels = column + "=" + keys
        hash_bins = block.get("hash_bins")
        if hash_bins:
            n_bins = int(hash_bins)
            labels = labels.map(lambda text: f"{column}#h{_stable_bucket(str(text), n_bins)}")
        out[f"{column}{suffix}"] = labels
    joint_suffix = str(block.get("joint_suffix", "__joint"))
    for left, right in _joint_pairs(block):
        _required_columns(out, [left, right], "exact_categorical.joint_pairs")
        left_keys = format_exact_keys(
            out[left], int(decimal_places.get(left, 8)), missing_token
        )
        right_keys = format_exact_keys(
            out[right], int(decimal_places.get(right, 8)), missing_token
        )
        out[f"{left}__{right}{joint_suffix}"] = (
            left + "=" + left_keys + "|" + right + "=" + right_keys
        )
    return out


def add_screen_budget_features(
    df: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Add daily-screen budget arithmetic (complete vs observed component sums)."""
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
    count = components.notna().sum(axis=1).astype("float64")
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
    out["screen_budget_boundary"] = (
        remainder.abs().le(tolerance).where(remainder.notna()).astype("float64")
    )
    out["screen_budget_violation"] = (
        remainder.lt(-tolerance).where(remainder.notna()).astype("float64")
    )
    return out


def add_decimal_lattice_features(
    df: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Expose each configured value's fractional part and first decimal digit."""
    out = df.copy()
    block = config["features"].get("decimal_lattice") or {}
    if not bool(block.get("enabled", False)):
        return out
    columns = _resolve_column_list(config, block)
    _required_columns(out, columns, "decimal_lattice")
    as_categorical = bool(block.get("as_categorical", False))
    for column in columns:
        values = out[column]
        out[f"{column}__fraction"] = values - np.floor(values)
        digit = np.floor(values * 10).mod(10).where(values.notna())
        if as_categorical:
            out[f"{column}__first_decimal"] = (
                digit.round().astype("Int64").astype("string").fillna("__MISSING__")
            )
        else:
            out[f"{column}__first_decimal"] = digit.astype("float64")
    return out


def add_structural_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply enabled row-local structural blocks in a fixed order."""
    out = add_exact_categorical_features(df, config)
    out = add_screen_budget_features(out, config)
    out = add_decimal_lattice_features(out, config)
    return out
