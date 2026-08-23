"""Target-free features fitted on the original ~7,500-row source distribution.

The original labelled rows are **not** appended to train. Statistics are fit on
the source table only (after dropping label columns and any exact feature-row
overlap with playground train+test). Train and test receive the same transform.

Config lives under ``features.reference``. Leakage controls:

- ``remove_query_overlaps`` must stay true (exact feature-row matches vs train
  **and** test are dropped from the source before fitting).
- Label columns are never loaded into the feature transform.
- Paths are exact configured / typed Kaggle mounts — no glob of ``/kaggle/input``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from s6e8.data import PROJECT_ROOT

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
LABEL_COLUMNS = (
    "addicted_label",
    "addiction_level",
    "Addiction Level",
    "addiction",
    "target",
)
DOWNLOAD_HINT = (
    "Original reference CSV was not found. Do not append those labelled rows to "
    "train. Download the source *features* only:\n"
    "  python3 -m kaggle datasets download -p data/raw --unzip "
    "jayjoshi37/smartphone-usage-and-addiction-prediction\n"
    "On Kaggle Kernels, attach dataset "
    "`jayjoshi37/smartphone-usage-and-addiction-prediction`. "
    "`scripts/prepare_kaggle_kernel.py` copies `features.reference.dataset_source` "
    "into kernel-metadata.json dataset_sources."
)


def parse_reference_block(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the enabled ``features.reference`` mapping, or None."""
    feat = config.get("features") or {}
    block = feat.get("reference")
    if not isinstance(block, dict):
        return None
    if not bool(block.get("enabled", False)):
        return None
    return block


def _typed_dataset_reference_path(
    dataset_source: str, filename: str, kaggle_input_root: Path
) -> Path:
    parts = dataset_source.strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("features.reference.dataset_source must be owner/dataset")
    owner, dataset = parts
    return kaggle_input_root / "datasets" / owner / dataset / filename


def _classic_dataset_reference_path(
    dataset_source: str, filename: str, kaggle_input_root: Path
) -> Path:
    parts = dataset_source.strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("features.reference.dataset_source must be owner/dataset")
    _owner, dataset = parts
    return kaggle_input_root / dataset / filename


def validate_dataset_source(source: str) -> str:
    text = str(source).strip()
    parts = text.split("/")
    if len(parts) != 2 or not all(parts) or any(part.strip() != part for part in parts):
        raise ValueError("features.reference.dataset_source must be owner/dataset")
    return text


def dataset_sources_for_kernel(config: dict[str, Any]) -> list[str]:
    """Kaggle `dataset_sources` entries needed to mount the original CSV."""
    block = parse_reference_block(config)
    if block is None:
        return []
    source = str(block.get("dataset_source") or "").strip()
    if not source:
        return []
    return [validate_dataset_source(source)]


def resolve_reference_path(
    block: dict[str, Any], *, kaggle_input_root: Path = KAGGLE_INPUT_ROOT
) -> Path:
    """Resolve the original CSV without globbing unrelated Kaggle inputs."""
    if "path" not in block or not str(block["path"]).strip():
        raise ValueError("features.reference.path is required when reference is enabled")
    configured = Path(str(block["path"]))
    candidates: list[Path] = []

    if configured.is_absolute():
        candidates.append(configured)
        if configured.is_file():
            return configured
    else:
        local = PROJECT_ROOT / configured
        candidates.append(local)
        if local.is_file():
            return local
        if configured.is_file():
            candidates.append(configured)
            return configured

    dataset_source = str(block.get("dataset_source") or "").strip()
    filename = configured.name
    if dataset_source:
        validate_dataset_source(dataset_source)
        typed = _typed_dataset_reference_path(dataset_source, filename, kaggle_input_root)
        classic = _classic_dataset_reference_path(
            dataset_source, filename, kaggle_input_root
        )
        for path in (typed, classic):
            candidates.append(path)
            if path.is_file():
                return path

    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{DOWNLOAD_HINT}\nChecked exact paths: {rendered}")


def canonical_row_hash(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Hash selected values after stable numeric and missing-value normalization."""
    canonical = pd.DataFrame(index=frame.index)
    for column in columns:
        if column not in frame.columns:
            raise KeyError(f"reference predictor {column!r} is missing from a query frame")
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce").astype("float64").round(8)
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
    """Deduplicate source predictors and drop every train/test feature-row match."""
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
    if len(sorted_values) == 0:
        return result
    result[valid] = (
        np.searchsorted(sorted_values, numeric[valid], side="right") / len(sorted_values)
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
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_out = train.copy()
    test_out = test.copy()
    added: list[str] = []

    for column in block.get("cdf_columns") or []:
        sorted_values = _sorted_numeric(reference[column])
        name = f"ref_{column}__cdf"
        train_out[name] = _empirical_cdf(sorted_values, train[column])
        test_out[name] = _empirical_cdf(sorted_values, test[column])
        added.append(name)

    for column in block.get("distance_columns") or []:
        median, scale = _robust_location(reference[column])
        name = f"ref_{column}__robust_z"
        train_z = (pd.to_numeric(train[column], errors="coerce") - median) / scale
        test_z = (pd.to_numeric(test[column], errors="coerce") - median) / scale
        train_out[name] = train_z
        test_out[name] = test_z
        abs_name = f"ref_{column}__robust_abs_z"
        train_out[abs_name] = train_z.abs()
        test_out[abs_name] = test_z.abs()
        added.extend([name, abs_name])

    for column in block.get("frequency_columns") or []:
        numeric = pd.api.types.is_numeric_dtype(reference[column])
        reference_keys = _frequency_keys(reference[column], numeric=numeric)
        frequencies = reference_keys.value_counts(dropna=False) / len(reference)
        name = f"ref_{column}__frequency"
        train_out[name] = (
            _frequency_keys(train[column], numeric=numeric)
            .map(frequencies)
            .fillna(0.0)
            .astype(float)
        )
        test_out[name] = (
            _frequency_keys(test[column], numeric=numeric)
            .map(frequencies)
            .fillna(0.0)
            .astype(float)
        )
        added.append(name)

    return train_out, test_out, added


def _add_knn_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    reference: pd.DataFrame,
    block: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    knn_block = block.get("knn") or {}
    if not isinstance(knn_block, dict) or not bool(knn_block.get("enabled", False)):
        return train, test, []
    columns = [str(c) for c in (knn_block.get("columns") or [])]
    if not columns:
        raise ValueError("features.reference.knn.columns must not be empty when knn is enabled")
    missing = [c for c in columns if c not in reference.columns]
    if missing:
        raise KeyError(f"kNN reference columns missing: {missing}")

    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    n_neighbors = int(knn_block.get("n_neighbors", 5))
    if n_neighbors < 1:
        raise ValueError("features.reference.knn.n_neighbors must be >= 1")

    ref_raw = reference[columns].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    ref_mat = scaler.fit_transform(imputer.fit_transform(ref_raw))
    k = min(n_neighbors, len(ref_mat))
    if k < 1:
        raise ValueError("No external reference rows remain for kNN")
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(ref_mat)

    def _query(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = frame[columns].apply(pd.to_numeric, errors="coerce")
        mat = scaler.transform(imputer.transform(raw))
        dists, _idx = nn.kneighbors(mat)
        return dists.mean(axis=1), dists.min(axis=1)

    train_out = train.copy()
    test_out = test.copy()
    train_mean, train_min = _query(train)
    test_mean, test_min = _query(test)
    train_out["ref_knn_mean_dist"] = train_mean
    train_out["ref_knn_min_dist"] = train_min
    test_out["ref_knn_mean_dist"] = test_mean
    test_out["ref_knn_min_dist"] = test_min
    return train_out, test_out, ["ref_knn_mean_dist", "ref_knn_min_dist"]


def _configured_columns(block: dict[str, Any]) -> dict[str, list[str]]:
    knn_block = block.get("knn") or {}
    knn_cols = list(knn_block.get("columns") or []) if isinstance(knn_block, dict) else []
    return {
        "predictors": [str(c) for c in (block.get("predictor_columns") or [])],
        "cdf": [str(c) for c in (block.get("cdf_columns") or [])],
        "distance": [str(c) for c in (block.get("distance_columns") or [])],
        "frequency": [str(c) for c in (block.get("frequency_columns") or [])],
        "knn": [str(c) for c in knn_cols],
    }


def apply_reference_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
    *,
    kaggle_input_root: Path = KAGGLE_INPUT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit target-free source distribution transforms and apply them to both frames."""
    block = parse_reference_block(config)
    if block is None:
        return train.copy(), test.copy(), {}

    if block.get("remove_query_overlaps", True) is not True:
        raise ValueError("features.reference.remove_query_overlaps must be true")
    if bool(block.get("use_labels", False)) or str(block.get("mode", "distribution")) != "distribution":
        raise ValueError(
            "features.reference is target-free only. Do not set use_labels/mode=label_aware; "
            "appending or supervising with original labels is rejected."
        )

    selected = _configured_columns(block)
    predictors = selected["predictors"]
    if not predictors:
        raise ValueError("features.reference.predictor_columns must not be empty")
    feature_cols = [
        column
        for key, columns in selected.items()
        if key != "predictors"
        for column in columns
    ]
    unknown = sorted(set(feature_cols) - set(predictors))
    if unknown:
        raise ValueError(
            "Reference feature columns must be listed in predictor_columns: "
            + ", ".join(unknown)
        )

    configured_path = Path(str(block["path"]))
    path = resolve_reference_path(block, kaggle_input_root=kaggle_input_root)
    source_bytes = path.read_bytes()
    drop_labels = {
        str(c) for c in (block.get("drop_label_columns") or LABEL_COLUMNS)
    }
    usecols = list(dict.fromkeys(predictors))
    try:
        reference = pd.read_csv(path, usecols=lambda name: name in usecols)
    except ValueError as exc:
        raise ValueError(f"External reference is missing selected columns: {exc}") from exc
    leaked = [c for c in reference.columns if c in drop_labels]
    if leaked:
        reference = reference.drop(columns=leaked)
    missing_predictors = [c for c in predictors if c not in reference.columns]
    if missing_predictors:
        raise ValueError(
            "External reference is missing predictor columns: " + ", ".join(missing_predictors)
        )

    retained, counts = prepare_reference_rows(reference, train, test, predictors)
    if retained.empty:
        raise ValueError("No external reference rows remain after leakage filtering")

    train_out, test_out, added = _add_distribution_features(train, test, retained, block)
    train_out, test_out, knn_added = _add_knn_features(train_out, test_out, retained, block)
    added = added + knn_added
    if list(train_out.columns) != list(test_out.columns):
        raise ValueError("Reference transforms changed train/test schema differently")

    provenance: dict[str, Any] = {
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "path": str(path),
        "configured_path": str(configured_path),
        "dataset_source": block.get("dataset_source"),
        "source_url": block.get("source_url"),
        "mode": "distribution",
        "external_supervision": False,
        "added_columns": added,
        **counts,
        "selected_columns": selected,
    }
    print(
        "reference_features "
        f"retained={provenance['retained_rows']} "
        f"overlap_removed={provenance['query_overlap_rows_removed']} "
        f"added={len(added)} sha256={provenance['sha256'][:12]}",
        flush=True,
    )
    return train_out, test_out, provenance
