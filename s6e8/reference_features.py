"""Leakage-safe features fitted from an external reference distribution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def _typed_dataset_reference_path(
    dataset_source: str, filename: str, kaggle_input_root: Path
) -> Path:
    """Build the only supported typed Kaggle dataset path for a configured slug."""
    parts = dataset_source.strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("external_reference.dataset_source must be owner/dataset")
    owner, dataset = parts
    return kaggle_input_root / "datasets" / owner / dataset / filename


def resolve_external_reference_path(
    block: dict[str, Any], *, kaggle_input_root: Path = KAGGLE_INPUT_ROOT
) -> Path:
    """Resolve a reference CSV without globbing across unrelated Kaggle inputs.

    Existing configured paths preserve legacy Kaggle mount compatibility. If that
    exact path is not mounted, a dataset source permits only the typed Batch
    mount using the same filename; no other directory or filename is searched.
    """
    configured_path = Path(str(block["path"]))
    if configured_path.is_file():
        return configured_path

    dataset_source = str(block.get("dataset_source", "")).strip()
    candidates = [configured_path]
    if dataset_source:
        typed_path = _typed_dataset_reference_path(
            dataset_source, configured_path.name, kaggle_input_root
        )
        candidates.append(typed_path)
        if typed_path.is_file():
            return typed_path

    rendered_candidates = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "External reference CSV was not found. Checked exact configured and "
        f"typed dataset paths: {rendered_candidates}"
    )


def canonical_row_hash(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Hash selected values after stable numeric and missing-value normalization."""
    canonical = pd.DataFrame(index=frame.index)
    for column in columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            # ``round`` preserves integer extension/native dtypes, while pandas
            # hashes integer 20 and floating 20.0 differently.  Force every
            # numeric predictor onto the same float64 representation first.
            # Reassign both signed zero and missing values to canonical bit
            # patterns so source/query storage details cannot affect the hash.
            numeric = (
                pd.to_numeric(values, errors="coerce")
                .astype("float64")
                .round(8)
            )
            numeric = numeric.mask(numeric.eq(0), 0.0)
            canonical[column] = numeric.mask(numeric.isna(), np.nan)
        else:
            canonical[column] = values.astype("string").fillna("__MISSING__")
    return pd.util.hash_pandas_object(canonical, index=False).astype("uint64")


def prepare_reference_rows(
    reference: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictor_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deduplicate reference predictors and remove every train/test match."""
    raw_rows = len(reference)
    reference_hash = canonical_row_hash(reference, predictor_columns)
    keep_unique = ~reference_hash.duplicated(keep="first")
    deduplicated = reference.loc[keep_unique].copy()
    deduplicated_hash = reference_hash.loc[keep_unique]

    query_hashes = set(canonical_row_hash(train, predictor_columns))
    query_hashes.update(canonical_row_hash(test, predictor_columns))
    keep_non_overlap = ~deduplicated_hash.isin(query_hashes)
    retained = deduplicated.loc[keep_non_overlap].reset_index(drop=True)

    provenance = {
        "raw_rows": int(raw_rows),
        "unique_rows": int(len(deduplicated)),
        "duplicate_rows_removed": int(raw_rows - len(deduplicated)),
        "query_overlap_rows_removed": int(len(deduplicated) - len(retained)),
        "retained_rows": int(len(retained)),
    }
    return retained, provenance


def _empirical_cdf(sorted_values: np.ndarray, query: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(query, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    valid = np.isfinite(numeric)
    result[valid] = (
        np.searchsorted(sorted_values, numeric[valid], side="right")
        / len(sorted_values)
    )
    return result


def _robust_location(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) == 0:
        raise ValueError(f"Reference column {values.name!r} has no numeric values")
    median = float(np.median(numeric))
    q25, q75 = np.quantile(numeric, [0.25, 0.75])
    scale = float(q75 - q25)
    return median, scale if np.isfinite(scale) and scale > 0 else 1.0


def _sorted_numeric(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) == 0:
        raise ValueError(f"Reference column {values.name!r} has no numeric values")
    return np.sort(numeric)


def _frequency_keys(values: pd.Series, *, numeric: bool) -> pd.Series:
    if numeric:
        normalized = pd.to_numeric(values, errors="coerce").round(8).astype("object")
        return normalized.where(normalized.notna(), "__MISSING__")
    return values.astype("string").fillna("__MISSING__")


def _add_distribution_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    reference: pd.DataFrame,
    block: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()

    for column in block.get("cdf_columns") or []:
        sorted_values = _sorted_numeric(reference[column])
        name = f"ref_{column}__cdf"
        train_out[name] = _empirical_cdf(sorted_values, train[column])
        test_out[name] = _empirical_cdf(sorted_values, test[column])

    for column in block.get("distance_columns") or []:
        median, scale = _robust_location(reference[column])
        name = f"ref_{column}__robust_z"
        train_z = (pd.to_numeric(train[column], errors="coerce") - median) / scale
        test_z = (pd.to_numeric(test[column], errors="coerce") - median) / scale
        train_out[name] = train_z
        test_out[name] = test_z
        train_out[f"ref_{column}__robust_abs_z"] = train_z.abs()
        test_out[f"ref_{column}__robust_abs_z"] = test_z.abs()

    for column in block.get("frequency_columns") or []:
        numeric = pd.api.types.is_numeric_dtype(reference[column])
        reference_keys = _frequency_keys(reference[column], numeric=numeric)
        frequencies = reference_keys.value_counts(dropna=False) / len(reference)
        name = f"ref_{column}__frequency"
        train_out[name] = _frequency_keys(train[column], numeric=numeric).map(frequencies)
        test_out[name] = _frequency_keys(test[column], numeric=numeric).map(frequencies)
        train_out[name] = train_out[name].fillna(0.0).astype(float)
        test_out[name] = test_out[name].fillna(0.0).astype(float)

    return train_out, test_out


def _validated_binary_labels(reference: pd.DataFrame, target_column: str) -> pd.Series:
    if target_column not in reference.columns:
        raise ValueError(f"Label-aware reference target {target_column!r} is missing")
    labels = pd.to_numeric(reference[target_column], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Label-aware reference target must be binary 0/1")
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Label-aware reference target must retain both classes")
    return labels.astype("int8")


def _source_target_mean(
    source_values: pd.Series,
    labels: pd.Series,
    query: pd.Series,
    *,
    n_bins: int,
    smoothing: float,
) -> np.ndarray:
    numeric = pd.to_numeric(source_values, errors="coerce")
    valid = numeric.notna()
    global_mean = float(labels.mean())
    finite = numeric.loc[valid].to_numpy(dtype=float)
    if len(finite) == 0:
        return np.full(len(query), global_mean, dtype=float)

    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    query_numeric = pd.to_numeric(query, errors="coerce")
    if len(edges) == 1:
        count = int(valid.sum())
        total = float(labels.loc[valid].sum())
        value = (total + smoothing * global_mean) / (count + smoothing)
        result = np.full(len(query), global_mean, dtype=float)
        exact = query_numeric.eq(edges[0]).fillna(False).to_numpy()
        result[exact] = value
        return result

    source_bins = pd.cut(
        numeric, bins=edges, labels=False, include_lowest=True, right=True
    )
    stats = pd.DataFrame({"bin": source_bins, "label": labels}).dropna(
        subset=["bin"]
    )
    grouped = stats.groupby("bin", observed=True)["label"].agg(["sum", "count"])
    smoothed = (grouped["sum"] + smoothing * global_mean) / (
        grouped["count"] + smoothing
    )
    query_bins = pd.cut(
        query_numeric, bins=edges, labels=False, include_lowest=True, right=True
    )
    return query_bins.map(smoothed).fillna(global_mean).to_numpy(dtype=float)


def _add_label_aware_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    reference: pd.DataFrame,
    block: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_column = str(block.get("target_column", ""))
    labels = _validated_binary_labels(reference, target_column)
    train_out = train.copy()
    test_out = test.copy()

    for column in block.get("class_cdf_columns") or []:
        class_zero = _sorted_numeric(reference.loc[labels.eq(0), column])
        class_one = _sorted_numeric(reference.loc[labels.eq(1), column])
        name = f"ref_{column}__class_cdf_gap"
        train_out[name] = _empirical_cdf(class_one, train[column]) - _empirical_cdf(
            class_zero, train[column]
        )
        test_out[name] = _empirical_cdf(class_one, test[column]) - _empirical_cdf(
            class_zero, test[column]
        )

    n_bins = int(block.get("n_bins", 50))
    smoothing = float(block.get("smoothing", 20.0))
    if n_bins < 1:
        raise ValueError("external_reference.n_bins must be at least 1")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("external_reference.smoothing must be non-negative")
    for column in block.get("target_mean_columns") or []:
        name = f"ref_{column}__source_target_mean"
        train_out[name] = _source_target_mean(
            reference[column],
            labels,
            train[column],
            n_bins=n_bins,
            smoothing=smoothing,
        )
        test_out[name] = _source_target_mean(
            reference[column],
            labels,
            test[column],
            n_bins=n_bins,
            smoothing=smoothing,
        )

    return train_out, test_out


def _configured_columns(block: dict[str, Any], mode: str) -> dict[str, list[str]]:
    selected = {
        "predictors": list(block.get("predictor_columns") or []),
        "cdf": list(block.get("cdf_columns") or []),
        "distance": list(block.get("distance_columns") or []),
        "frequency": list(block.get("frequency_columns") or []),
    }
    if mode == "label_aware":
        selected["class_cdf"] = list(block.get("class_cdf_columns") or [])
        selected["target_mean"] = list(block.get("target_mean_columns") or [])
    return selected


def apply_external_reference_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
    *,
    kaggle_input_root: Path = KAGGLE_INPUT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit configured reference transforms and apply them to both query frames."""
    block = config.get("external_reference") or {}
    if not bool(block.get("enabled", False)):
        return train.copy(), test.copy(), {}

    mode = str(block.get("mode", "distribution"))
    if mode not in {"distribution", "label_aware"}:
        raise ValueError(f"Unsupported external_reference.mode {mode!r}")
    if block.get("remove_query_overlaps", True) is not True:
        raise ValueError("external_reference.remove_query_overlaps must be true")

    selected = _configured_columns(block, mode)
    predictors = selected["predictors"]
    if not predictors:
        raise ValueError("external_reference.predictor_columns must not be empty")
    configured_features = [
        column
        for key, columns in selected.items()
        if key != "predictors"
        for column in columns
    ]
    unknown = sorted(set(configured_features) - set(predictors))
    if unknown:
        raise ValueError(
            "External reference feature columns must be predictors: " + ", ".join(unknown)
        )

    configured_path = Path(str(block["path"]))
    path = resolve_external_reference_path(block, kaggle_input_root=kaggle_input_root)
    source_bytes = path.read_bytes()
    target_column = str(block.get("target_column", ""))
    usecols = list(predictors)
    if mode == "label_aware":
        if not target_column:
            raise ValueError("Label-aware external reference requires target_column")
        usecols.append(target_column)
    try:
        reference = pd.read_csv(path, usecols=usecols)
    except ValueError as exc:
        raise ValueError(f"External reference is missing selected columns: {exc}") from exc

    retained, counts = prepare_reference_rows(reference, train, test, predictors)
    if retained.empty:
        raise ValueError("No external reference rows remain after leakage filtering")

    train_out, test_out = _add_distribution_features(
        train, test, retained, block
    )
    external_supervision = mode == "label_aware"
    if external_supervision:
        train_out, test_out = _add_label_aware_features(
            train_out, test_out, retained, block
        )
    if list(train_out.columns) != list(test_out.columns):
        raise ValueError("External reference transforms changed train/test schema differently")

    provenance: dict[str, Any] = {
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "path": str(path),
        "configured_path": str(configured_path),
        "dataset_source": block.get("dataset_source"),
        "source_url": block.get("source_url"),
        "mode": mode,
        **counts,
        "selected_columns": selected,
        "external_supervision": external_supervision,
    }
    if external_supervision:
        provenance["target_column"] = target_column
    return train_out, test_out, provenance
