# Structural Tree and Reference Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-safe exact categorical copies, structural screen-budget features, external reference features, CatBoost experiments, honest blend evaluation, and Kaggle cloud provenance to the existing fixed-five-fold pipeline.

**Architecture:** Row-local structural features stay in focused pure functions called by `s6e8.features.transform`; external reference transforms are fitted once from a deduplicated, overlap-filtered public source and applied in `_prepare_xy`. The CV runner owns fold assignments and artifact provenance, while a separate blend module performs leave-one-fold-out weight selection.

**Tech Stack:** Python 3.11+, pandas, NumPy, scikit-learn, CatBoost, PyYAML, pytest, Kaggle CLI, GitHub Actions.

**Spec:** `docs/plans/2026-08-18-structural-feature-engineering-design.md`

## Global Constraints

- Formal validation is exactly `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- All feature families and model parameters are enabled only through YAML.
- CatBoost computes categorical target statistics internally from each fold-training partition; no validation labels enter preprocessing.
- The 7,500-row source dataset is never concatenated with competition training rows.
- Exact source rows matching either competition train or test predictors are excluded before reference fitting.
- Label-aware source features are marked `external_supervision: true` in metrics and experiment metadata.
- Every formal run saves OOF predictions, test predictions, fold ids, metrics, resolved config, and provenance.
- Public prediction CSV files are neither downloaded nor consumed.
- No script or workflow automatically submits to the Kaggle leaderboard.
- Implementation uses tests-first development and small commits.

## File structure

- Create `s6e8/structural_features.py`: deterministic exact-category, screen-budget, and decimal-lattice transforms.
- Create `s6e8/reference_features.py`: source loading, canonical row hashes, overlap filtering, fitted distribution and label-aware transforms.
- Create `s6e8/blending.py`: pure alignment, correlation, grid search, and leave-one-fold-out blend evaluation.
- Modify `s6e8/features.py`: call structural transforms and expose the complete categorical-column list.
- Modify `s6e8/models/train.py`: integrate reference features, fold ids, provenance, CatBoost importance, and canonical artifacts.
- Modify `s6e8/runtime.py`: validate the named fixed-five-fold formal protocol and serialize dependency metadata.
- Modify `kaggle/runner.py`: install backend-specific optional dependencies.
- Modify `scripts/prepare_kaggle_kernel.py`: attach configured external Kaggle datasets.
- Modify `.github/workflows/kaggle-train.yml`: remove the competition-submission input and command.
- Modify `scripts/blend_oof.py`: add honest leave-one-fold-out blending and fold reports.
- Create six `configs/catboost_*.yaml` files: controlled experiment sequence from numeric control through source-label reference.
- Create `tests/test_structural_features.py`, `tests/test_reference_features.py`, and `tests/test_blending.py`.
- Modify `tests/test_config.py`, `tests/test_smoke_train.py`, `tests/test_cloud_helpers.py`, and `tests/test_prepare_kernel.py`.
- Modify `README.md` and `experiments/LOG.md` only after real Kaggle metrics exist.

---

### Task 1: Lock the formal five-fold protocol and enrich artifacts

**Files:**
- Modify: `s6e8/runtime.py:20-72`
- Modify: `s6e8/models/train.py:180-266`
- Modify: `s6e8/models/train.py:509-627`
- Modify: `tests/test_config.py`
- Modify: `tests/test_smoke_train.py`

**Interfaces:**
- Consumes: existing YAML `experiment`, `cv`, and `output` blocks.
- Produces: `validate_config(config)`, `artifacts["fold_ids"]`, canonical parquet artifacts, and resolved experiment metadata used by every later task.

- [ ] **Step 1: Write failing formal-protocol tests**

Add these tests to `tests/test_config.py`:

```python
import copy
import pytest


def test_formal_protocol_rejects_fold_or_seed_drift(baseline_config_path):
    config = load_config(baseline_config_path)
    config["experiment"]["formal"] = True
    config["experiment"]["validation_protocol"] = "fixed5_seed42_v1"
    validate_config(config)

    bad_folds = copy.deepcopy(config)
    bad_folds["cv"]["n_splits"] = 4
    with pytest.raises(ValueError, match="fixed5_seed42_v1"):
        validate_config(bad_folds)

    bad_seed = copy.deepcopy(config)
    bad_seed["experiment"]["seed"] = 7
    with pytest.raises(ValueError, match="fixed5_seed42_v1"):
        validate_config(bad_seed)
```

- [ ] **Step 2: Run the protocol test and verify red**

Run: `pytest tests/test_config.py::test_formal_protocol_rejects_fold_or_seed_drift -v`

Expected: FAIL because `validate_config` does not enforce the named protocol.

- [ ] **Step 3: Implement the named formal-protocol guard**

Add to `s6e8/runtime.py`:

```python
FORMAL_PROTOCOL = "fixed5_seed42_v1"


def _validate_formal_protocol(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    if not bool(experiment.get("formal", False)):
        return
    protocol = experiment.get("validation_protocol")
    expected = {
        "protocol": FORMAL_PROTOCOL,
        "seed": 42,
        "type": "stratified",
        "n_splits": 5,
        "shuffle": True,
    }
    actual = {
        "protocol": protocol,
        "seed": int(experiment["seed"]),
        "type": str(config["cv"].get("type", "")),
        "n_splits": int(config["cv"]["n_splits"]),
        "shuffle": bool(config["cv"].get("shuffle", False)),
    }
    if actual != expected:
        raise ValueError(f"Formal protocol {FORMAL_PROTOCOL} required; got {actual}")
```

Call `_validate_formal_protocol(config)` at the end of `validate_config`.

- [ ] **Step 4: Run the protocol tests and verify green**

Run: `pytest tests/test_config.py -v`

Expected: all configuration tests PASS.

- [ ] **Step 5: Write failing fold-artifact tests**

Extend `test_smoke_train_on_synthetic_data` in `tests/test_smoke_train.py`:

```python
    oof_frame = pd.read_parquet(written["oof_predictions"])
    assert set(oof_frame["fold"].unique()) == {0, 1}
    assert oof_frame["fold"].notna().all()
    assert Path(written["test_predictions"]).exists()

    metrics = json.loads(Path(written["metrics"]).read_text(encoding="utf-8"))
    assert metrics["fold_auc_min"] == min(metrics["fold_scores"])
    assert metrics["fold_auc_max"] == max(metrics["fold_scores"])
    assert metrics["n_categorical_features"] >= 0

    experiment = json.loads(Path(written["experiment"]).read_text(encoding="utf-8"))
    assert experiment["resolved_config"]["experiment"]["name"] == "synthetic_smoke"
    assert experiment["feature_names"] == artifacts["feature_names"]
```

Add `import json` at the top of the test file.

- [ ] **Step 6: Run the artifact test and verify red**

Run: `pytest tests/test_smoke_train.py::test_smoke_train_on_synthetic_data -v`

Expected: FAIL because fold ids and canonical artifact keys are absent.

- [ ] **Step 7: Record fold ids in `_run_cv`**

In `s6e8/models/train.py`, allocate and assign fold ids:

```python
fold_ids = np.full(len(X), -1, dtype=np.int16)

for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y_np)):
    fold_ids[va_idx] = fold
    # existing training body uses display_fold = fold + 1 for logging

if np.any(fold_ids < 0):
    raise RuntimeError("Every training row must receive exactly one validation fold")
```

Return `fold_ids` and `cat_cols` in the artifact dictionary. Extend `_prepare_xy`
to return an empty `data_provenance` mapping as its sixth value, and pass that
mapping through `_run_cv`; Task 4 will populate it. Preserve one-based fold
numbers only in console messages; stored fold ids are zero-based.

- [ ] **Step 8: Write canonical artifacts and resolved metadata**

In `save_artifacts`, build each frame once:

```python
oof_frame = pd.DataFrame({
    id_col: artifacts["train_ids"],
    target: artifacts["y"],
    "pred": artifacts["oof"],
    "fold": artifacts["fold_ids"],
})
oof_frame.to_parquet(oof_dir / "oof.parquet", index=False)
oof_frame.to_parquet(oof_dir / "oof_predictions.parquet", index=False)

test_frame = pd.DataFrame({id_col: artifacts["test_ids"], "pred": artifacts["test_pred"]})
test_frame.to_parquet(oof_dir / "test.parquet", index=False)
test_frame.to_parquet(oof_dir / "test_predictions.parquet", index=False)
```

Add written keys `oof_predictions` and `test_predictions`. Add metrics fields:

```python
"fold_auc_min": float(min(artifacts["fold_scores"])),
"fold_auc_max": float(max(artifacts["fold_scores"])),
"n_categorical_features": len(artifacts.get("cat_cols") or []),
"categorical_feature_names": artifacts.get("cat_cols") or [],
"data_provenance": artifacts.get("data_provenance") or {},
```

Write `experiment.json` as:

```python
resolved_config = {k: v for k, v in config.items() if k != "_config_path"}
experiment_payload = {
    **summary,
    "resolved_config": resolved_config,
    "feature_names": artifacts["feature_names"],
    "categorical_feature_names": artifacts.get("cat_cols") or [],
    "data_provenance": artifacts.get("data_provenance") or {},
}
```

- [ ] **Step 9: Run focused tests and commit**

Run: `pytest tests/test_config.py tests/test_smoke_train.py -v`

Expected: PASS.

```bash
git add s6e8/runtime.py s6e8/models/train.py tests/test_config.py tests/test_smoke_train.py
git commit -m "Enforce formal CV and enrich prediction artifacts"
```

---

### Task 2: Add exact categorical and structural budget features

**Files:**
- Create: `s6e8/structural_features.py`
- Create: `tests/test_structural_features.py`
- Modify: `s6e8/features.py:49-157`

**Interfaces:**
- Consumes: `features.exact_categorical`, `features.screen_budget`, and `features.decimal_lattice` YAML blocks.
- Produces: `add_structural_features(df, config) -> pd.DataFrame` and `categorical_feature_columns(df, config) -> list[str]`.

- [ ] **Step 1: Write failing exact-category tests**

Create `tests/test_structural_features.py`:

```python
import numpy as np
import pandas as pd

from s6e8.structural_features import add_exact_categorical_features


def test_exact_categories_are_canonical_and_missing_explicit():
    frame = pd.DataFrame({
        "age": [21.0, 21.0, np.nan],
        "daily_screen_time_hours": [3.1, 3.1000000000000001, np.nan],
    })
    config = {
        "features": {
            "numeric": ["age", "daily_screen_time_hours"],
            "exact_categorical": {
                "enabled": True,
                "columns": ["age", "daily_screen_time_hours"],
                "suffix": "__exact",
                "missing_token": "__MISSING__",
                "decimal_places": {"age": 0, "daily_screen_time_hours": 2},
            },
        }
    }
    out = add_exact_categorical_features(frame, config)
    assert out["age__exact"].tolist() == ["age=21", "age=21", "age=__MISSING__"]
    assert out["daily_screen_time_hours__exact"].tolist() == [
        "daily_screen_time_hours=3.10",
        "daily_screen_time_hours=3.10",
        "daily_screen_time_hours=__MISSING__",
    ]
```

- [ ] **Step 2: Run the exact-category test and verify red**

Run: `pytest tests/test_structural_features.py::test_exact_categories_are_canonical_and_missing_explicit -v`

Expected: FAIL because `s6e8.structural_features` does not exist.

- [ ] **Step 3: Implement deterministic exact categories**

Create `s6e8/structural_features.py` with:

```python
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _required_columns(df: pd.DataFrame, columns: list[str], block: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{block} columns are missing: {missing}")


def canonical_numeric_value(value: object, decimals: int, missing_token: str) -> str:
    if pd.isna(value):
        return missing_token
    number = float(value)
    if decimals == 0:
        return str(int(round(number)))
    return f"{number:.{decimals}f}"


def add_exact_categorical_features(
    df: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
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
            lambda value, c=column, d=decimals: f"{c}={canonical_numeric_value(value, d, missing_token)}"
        )
    return out
```

- [ ] **Step 4: Run the exact-category test and verify green**

Run: `pytest tests/test_structural_features.py::test_exact_categories_are_canonical_and_missing_explicit -v`

Expected: PASS.

- [ ] **Step 5: Write failing budget and lattice tests**

Append:

```python
from s6e8.structural_features import add_decimal_lattice_features, add_screen_budget_features


def test_budget_features_preserve_complete_and_observed_semantics():
    frame = pd.DataFrame({
        "daily_screen_time_hours": [8.0, 8.0, 0.0],
        "social_media_hours": [2.0, np.nan, 0.0],
        "gaming_hours": [1.0, 1.0, 0.0],
        "work_study_hours": [2.0, 2.0, 0.0],
        "weekend_screen_time": [10.0, 10.0, 1.0],
        "sleep_hours": [7.0, 7.0, 8.0],
    })
    config = {"features": {"screen_budget": {"enabled": True, "tolerance": 1e-9}}}
    out = add_screen_budget_features(frame, config)
    assert out.loc[0, "screen_component_sum_complete"] == 5.0
    assert out.loc[0, "screen_remainder_complete"] == 3.0
    assert out.loc[1, "screen_component_count"] == 2
    assert pd.isna(out.loc[1, "screen_component_sum_complete"])
    assert out.loc[1, "screen_component_sum_observed"] == 3.0
    assert out.loc[1, "screen_remainder_observed"] == 5.0
    assert pd.isna(out.loc[2, "screen_component_share_complete"])
    assert out.loc[0, "awake_non_screen_hours"] == 9.0


def test_decimal_lattice_is_an_isolated_first_digit_block():
    frame = pd.DataFrame({"daily_screen_time_hours": [3.27, np.nan]})
    config = {
        "features": {
            "decimal_lattice": {
                "enabled": True,
                "columns": ["daily_screen_time_hours"],
            }
        }
    }
    out = add_decimal_lattice_features(frame, config)
    assert np.isclose(out.loc[0, "daily_screen_time_hours__fraction"], 0.27)
    assert out.loc[0, "daily_screen_time_hours__first_decimal"] == 2
    assert pd.isna(out.loc[1, "daily_screen_time_hours__first_decimal"])
```

- [ ] **Step 6: Run the new tests and verify red**

Run: `pytest tests/test_structural_features.py -v`

Expected: FAIL because the budget and lattice functions are absent.

- [ ] **Step 7: Implement budget and lattice transforms**

Implement named constants for the three component columns and functions that:

```python
components = out[["social_media_hours", "gaming_hours", "work_study_hours"]]
complete = components.sum(axis=1, min_count=3)
observed = components.sum(axis=1, min_count=1)
count = components.notna().sum(axis=1).astype("int8")
daily = out["daily_screen_time_hours"]

out["screen_component_sum_complete"] = complete
out["screen_component_sum_observed"] = observed
out["screen_component_count"] = count
out["screen_remainder_complete"] = daily - complete
out["screen_remainder_observed"] = daily - observed
out["screen_component_share_complete"] = complete.div(daily.replace(0, np.nan))
out["screen_remainder_share_complete"] = (daily - complete).div(daily.replace(0, np.nan))
out["weekend_minus_component_sum"] = out["weekend_screen_time"] - complete
out["weekend_minus_remainder"] = out["weekend_screen_time"] - (daily - complete)
out["awake_non_screen_hours"] = 24.0 - out["sleep_hours"] - daily
remainder = out["screen_remainder_complete"]
out["screen_budget_boundary"] = remainder.abs().le(tolerance).where(
    remainder.notna()
).astype("Int8")
out["screen_budget_violation"] = remainder.lt(-tolerance).where(
    remainder.notna()
).astype("Int8")
```

For each lattice column, compute `fraction = value - floor(value)` and nullable `Int8` first digit as `floor(value * 10) % 10`, preserving missing values.

- [ ] **Step 8: Integrate structural functions and generated categorical columns**

In `s6e8/features.py`, import the three transforms, call them after existing row-wise engineering, and add:

```python
def categorical_feature_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    columns = [c for c in config["features"]["categorical"] if c in df.columns]
    block = config["features"].get("exact_categorical") or {}
    if bool(block.get("enabled", False)):
        source = block.get("columns", "auto_numeric")
        if source == "auto_numeric":
            source = config["features"]["numeric"]
        suffix = str(block.get("suffix", "__exact"))
        columns.extend(f"{column}{suffix}" for column in source if f"{column}{suffix}" in df.columns)
    return list(dict.fromkeys(columns))
```

Make `cast_categoricals` use this function. Ensure disabled blocks leave all existing config outputs unchanged.

- [ ] **Step 9: Run feature tests and commit**

Run: `pytest tests/test_structural_features.py tests/test_features.py -v`

Expected: PASS.

```bash
git add s6e8/structural_features.py s6e8/features.py tests/test_structural_features.py tests/test_features.py
git commit -m "Add exact categorical and screen budget features"
```

---

### Task 3: Route generated categories through CatBoost and add controlled configs

**Files:**
- Modify: `s6e8/models/train.py:167-177`
- Modify: `s6e8/models/train.py:360-398`
- Modify: `tests/test_smoke_train.py`
- Modify: `tests/test_config.py`
- Create: `configs/catboost_numeric_v1.yaml`
- Create: `configs/catboost_exactcat_v1.yaml`
- Create: `configs/catboost_exactcat_budget_v1.yaml`
- Create: `configs/catboost_exactcat_budget_lattice_v1.yaml`

**Interfaces:**
- Consumes: `categorical_feature_columns(df, config)` from Task 2.
- Produces: CatBoost-ready data frames, categorical counts, feature importances, and four isolated formal configs.

- [ ] **Step 1: Write failing generated-category routing test**

Add to `tests/test_smoke_train.py`:

```python
from s6e8.models.train import _prepare_xy


def test_prepare_xy_routes_exact_copies_as_catboost_categories(tmp_path, baseline_config_path):
    train_df, test_df = _synthetic_frames()
    y = train_df.pop("addicted_label")
    raw = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    raw["features"]["exact_categorical"] = {
        "enabled": True,
        "columns": ["age", "daily_screen_time_hours"],
        "suffix": "__exact",
        "decimal_places": {"age": 0, "daily_screen_time_hours": 2},
    }
    path = tmp_path / "exact.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(path)
    X, X_test, _, _, cat_cols, provenance = _prepare_xy(train_df, test_df, y, config)
    assert "age__exact" in cat_cols
    assert "daily_screen_time_hours__exact" in cat_cols
    assert list(X.columns) == list(X_test.columns)
    assert provenance == {}
```

- [ ] **Step 2: Run routing test and verify red**

Run: `pytest tests/test_smoke_train.py::test_prepare_xy_routes_exact_copies_as_catboost_categories -v`

Expected: FAIL because `_prepare_xy` only reads the base categorical list.

- [ ] **Step 3: Route the complete categorical list and CatBoost importance**

Import `categorical_feature_columns` into `s6e8/models/train.py` and use:

```python
cat_cols = [c for c in categorical_feature_columns(train_feat, config) if c in cols]
```

In `_fold_catboost`, store fold importances using `model.get_feature_importance()` and `X_tr.columns`, matching the existing LightGBM `{"gain": ...}` structure. Keep `_catboost_frame` missing strings explicit and leave numeric columns numeric.

- [ ] **Step 4: Run routing test and verify green**

Run: `pytest tests/test_smoke_train.py::test_prepare_xy_routes_exact_copies_as_catboost_categories -v`

Expected: PASS.

- [ ] **Step 5: Create the four formal configs**

Copy the shared data/CV/output schema from `configs/lgbm_nocat.yaml`. Each config sets:

```yaml
experiment:
  seed: 42
  formal: true
  validation_protocol: fixed5_seed42_v1
runtime:
  accelerator: gpu
model:
  name: catboost
  num_boost_round: 6000
  early_stopping_rounds: 200
  log_evaluation: 100
  params:
    learning_rate: 0.05
    depth: 8
    l2_leaf_reg: 6.0
    one_hot_max_size: 4
    max_ctr_complexity: 2
    random_strength: 1.0
    allow_writing_files: false
```

The exact isolated differences are:

```yaml
# catboost_numeric_v1
exact_categorical: {enabled: false}
screen_budget: {enabled: false}
decimal_lattice: {enabled: false}

# catboost_exactcat_v1
exact_categorical: {enabled: true, columns: auto_numeric, suffix: __exact, ...}
screen_budget: {enabled: false}
decimal_lattice: {enabled: false}

# catboost_exactcat_budget_v1
exact_categorical: {enabled: true, columns: auto_numeric, suffix: __exact, ...}
screen_budget: {enabled: true, tolerance: 1.0e-9}
decimal_lattice: {enabled: false}

# catboost_exactcat_budget_lattice_v1
exact_categorical: {enabled: true, columns: auto_numeric, suffix: __exact, ...}
screen_budget: {enabled: true, tolerance: 1.0e-9}
decimal_lattice:
  enabled: true
  columns: [daily_screen_time_hours, social_media_hours, gaming_hours, work_study_hours, sleep_hours, weekend_screen_time]
```

Use decimal precision `0` for `age`, `notifications_per_day`, and `app_opens_per_day`, and `2` for six hour-valued columns.

- [ ] **Step 6: Write and run config-isolation tests**

Add tests that load adjacent configs, remove `experiment.name`, `experiment.hypothesis`, `experiment.change`, and compare dictionaries after normalizing only the intended feature block. Also assert every new config has five folds, seed 42, formal protocol, CatBoost backend, and unique experiment name.

Run: `pytest tests/test_config.py -v && python scripts/validate_configs.py`

Expected: PASS and each new YAML prints `ok`.

- [ ] **Step 7: Add an optional CatBoost smoke test**

Add:

```python
def test_catboost_exact_category_smoke(tmp_path, baseline_config_path):
    pytest.importorskip("catboost")
    train_df, test_df = _synthetic_frames(120, 30)
    raw = yaml.safe_load(Path("configs/catboost_exactcat_budget_v1.yaml").read_text())
    raw["experiment"]["name"] = "catboost_smoke"
    raw["experiment"]["formal"] = False
    raw["runtime"]["accelerator"] = "cpu"
    raw["cv"]["n_splits"] = 2
    raw["model"]["num_boost_round"] = 10
    raw["model"]["early_stopping_rounds"] = 3
    raw["model"]["params"]["depth"] = 4
    raw["model"]["params"].pop("max_ctr_complexity", None)
    config = raw
    X_train, y = split_xy(train_df, config)
    artifacts = train_cv(X_train, test_df, y, config)
    assert len(artifacts["cat_cols"]) == 12
    assert np.isfinite(artifacts["oof"]).all()
```

Add required `numpy` and `pytest` imports.

Run: `pytest tests/test_smoke_train.py -v`

Expected: PASS, with the CatBoost test skipped only when the optional package is unavailable.

- [ ] **Step 8: Commit CatBoost routing and configs**

```bash
git add s6e8/models/train.py tests/test_smoke_train.py tests/test_config.py configs/catboost_*.yaml
git commit -m "Add controlled CatBoost structural experiments"
```

---

### Task 4: Add external reference distribution and label-aware transforms

**Files:**
- Create: `s6e8/reference_features.py`
- Create: `tests/test_reference_features.py`
- Modify: `s6e8/models/train.py:167-177`
- Create: `configs/catboost_exactcat_budget_refdist_v1.yaml`
- Create: `configs/catboost_exactcat_budget_reflabel_v1.yaml`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: transformed competition train/test frames, external CSV path, shared predictor columns, and optional source label.
- Produces: `apply_external_reference_features(train, test, config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]`.

- [ ] **Step 1: Write failing hash, deduplication, and overlap tests**

Create `tests/test_reference_features.py`:

```python
import pandas as pd

from s6e8.reference_features import canonical_row_hash, prepare_reference_rows

PREDICTORS = ["age", "daily_screen_time_hours", "gender"]


def test_prepare_reference_removes_duplicates_and_all_query_overlaps():
    reference = pd.DataFrame({
        "age": [20, 20, 30, 40],
        "daily_screen_time_hours": [3.1, 3.1, 5.2, 7.3],
        "gender": ["Male", "Male", "Female", "Other"],
        "addicted_label": [0, 0, 1, 1],
    })
    train = reference.iloc[[0]][PREDICTORS].copy()
    test = reference.iloc[[2]][PREDICTORS].copy()
    retained, provenance = prepare_reference_rows(reference, train, test, PREDICTORS)
    assert retained[PREDICTORS].to_dict("records") == [
        {"age": 40, "daily_screen_time_hours": 7.3, "gender": "Other"}
    ]
    assert provenance["raw_rows"] == 4
    assert provenance["duplicate_rows_removed"] == 1
    assert provenance["query_overlap_rows_removed"] == 2
    assert provenance["retained_rows"] == 1
    assert canonical_row_hash(train, PREDICTORS).iloc[0] not in set(
        canonical_row_hash(retained, PREDICTORS)
    )
```

- [ ] **Step 2: Run overlap test and verify red**

Run: `pytest tests/test_reference_features.py::test_prepare_reference_removes_duplicates_and_all_query_overlaps -v`

Expected: FAIL because the reference module does not exist.

- [ ] **Step 3: Implement canonical hashes and row filtering**

Create `s6e8/reference_features.py`. Canonicalize numeric columns with `round(8)`, categorical missing values with `__MISSING__`, and hash using:

```python
def canonical_row_hash(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    canonical = pd.DataFrame(index=frame.index)
    for column in columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            canonical[column] = pd.to_numeric(values, errors="coerce").round(8)
        else:
            canonical[column] = values.astype("string").fillna("__MISSING__")
    return pd.util.hash_pandas_object(canonical, index=False).astype("uint64")
```

Deduplicate by reference hash, form the union of train and test hashes, exclude all union matches, reset the retained index, and return the four provenance counts.

- [ ] **Step 4: Run overlap test and verify green**

Run: `pytest tests/test_reference_features.py::test_prepare_reference_removes_duplicates_and_all_query_overlaps -v`

Expected: PASS.

- [ ] **Step 5: Write failing transform and provenance tests**

Append tests that create a temporary source CSV with the full predictor schema and assert:

```python
train_out, test_out, provenance = apply_external_reference_features(train, test, config)
assert "ref_daily_screen_time_hours__cdf" in train_out
assert "ref_daily_screen_time_hours__robust_z" in train_out
assert "ref_daily_screen_time_hours__robust_abs_z" in test_out
assert list(train_out.columns) == list(test_out.columns)
assert provenance["external_supervision"] is False
assert len(provenance["sha256"]) == 64
```

For label-aware mode assert:

```python
assert "ref_daily_screen_time_hours__class_cdf_gap" in train_out
assert "ref_daily_screen_time_hours__source_target_mean" in train_out
assert provenance["external_supervision"] is True
```

Also assert that distribution mode produces identical features when source labels are permuted.

- [ ] **Step 6: Run transform tests and verify red**

Run: `pytest tests/test_reference_features.py -v`

Expected: FAIL because fitted reference transforms are absent.

- [ ] **Step 7: Implement target-free reference transforms**

Implement helpers:

```python
def _empirical_cdf(sorted_values: np.ndarray, query: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(query, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    valid = np.isfinite(numeric)
    result[valid] = np.searchsorted(sorted_values, numeric[valid], side="right") / len(sorted_values)
    return result


def _robust_location(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    median = float(np.median(numeric))
    q25, q75 = np.quantile(numeric, [0.25, 0.75])
    scale = float(q75 - q25)
    return median, scale if np.isfinite(scale) and scale > 0 else 1.0
```

For configured `cdf_columns`, add `ref_<column>__cdf`. For `distance_columns`, add signed and absolute robust z. For `frequency_columns`, map retained-source exact-value frequency divided by retained row count. Never read `addicted_label` in distribution mode.

- [ ] **Step 8: Implement label-aware transforms**

For each configured `class_cdf_columns`, fit sorted source values separately for label 0 and 1 and add `F1(x) - F0(x)`. For each `target_mean_columns`, create unique source quantile edges, assign source bins, and map smoothed bin means:

```python
smoothed = (bin_sum + smoothing * global_mean) / (bin_count + smoothing)
```

Unseen/out-of-range bins fall back to the source global mean. Reject label-aware mode if the target column is missing, non-binary, or only one class remains after overlap filtering.

Compute SHA-256 from the source file bytes and include source URL, mode, retained counts, selected columns, and `external_supervision` in provenance.

- [ ] **Step 9: Integrate reference transforms in `_prepare_xy`**

Call row-local `transform` first, then:

```python
train_feat, test_feat, reference_provenance = apply_external_reference_features(
    train_feat, test_feat, config
)
```

Return provenance from `_prepare_xy`, put it in `_run_cv` artifacts, and preserve empty provenance when the block is disabled. Assert train/test columns are identical before selecting features.

- [ ] **Step 10: Create and validate the two reference configs**

Copy `catboost_exactcat_budget_v1.yaml`. Distribution mode adds:

```yaml
external_reference:
  enabled: true
  path: /kaggle/input/smartphone-usage-and-addiction-prediction/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
  dataset_source: jayjoshi37/smartphone-usage-and-addiction-prediction
  source_url: https://www.kaggle.com/datasets/jayjoshi37/smartphone-usage-and-addiction-prediction
  mode: distribution
  predictor_columns: [age, gender, daily_screen_time_hours, social_media_hours, gaming_hours, work_study_hours, sleep_hours, notifications_per_day, app_opens_per_day, weekend_screen_time, stress_level, academic_work_impact]
  cdf_columns: [daily_screen_time_hours, weekend_screen_time, social_media_hours]
  distance_columns: [daily_screen_time_hours, weekend_screen_time, social_media_hours, notifications_per_day, app_opens_per_day]
  frequency_columns: []
  remove_query_overlaps: true
```

Label-aware mode changes only `mode`, adds `target_column: addicted_label`, `class_cdf_columns` equal to the five distance columns, and `target_mean_columns` for daily, weekend, notifications, and app opens with `n_bins: 50` and `smoothing: 20.0`.

Run: `pytest tests/test_reference_features.py tests/test_config.py -v && python scripts/validate_configs.py`

Expected: PASS.

- [ ] **Step 11: Commit reference infrastructure**

```bash
git add s6e8/reference_features.py s6e8/models/train.py tests/test_reference_features.py tests/test_config.py configs/catboost_exactcat_budget_refdist_v1.yaml configs/catboost_exactcat_budget_reflabel_v1.yaml
git commit -m "Add leakage-safe external reference features"
```

---

### Task 5: Add honest leave-one-fold-out blend evaluation

**Files:**
- Create: `s6e8/blending.py`
- Create: `tests/test_blending.py`
- Modify: `scripts/blend_oof.py`

**Interfaces:**
- Consumes: aligned prediction matrices, target vector, fold ids, test predictions, and a deterministic weight grid.
- Produces: honest meta-OOF predictions, aggregated test predictions, fold-selected weights, per-fold AUC, correlations, and JSON metrics.

- [ ] **Step 1: Write failing LOFO tests**

Create `tests/test_blending.py`:

```python
import numpy as np

from s6e8.blending import lofo_grid_blend


def test_lofo_weights_never_use_the_scored_fold():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    folds = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    pred_a = np.array([0.1, 0.9, 0.2, 0.8, 0.8, 0.2, 0.7, 0.3])
    pred_b = 1.0 - pred_a
    test = np.array([[0.2, 0.8], [0.7, 0.3]])
    result = lofo_grid_blend(
        y=y,
        fold_ids=folds,
        oof_matrix=np.vstack([pred_a, pred_b]),
        test_matrix=test,
        step=50,
    )
    assert result.oof.shape == y.shape
    assert result.test.shape == (2,)
    assert len(result.fold_weights) == 4
    assert all(np.isclose(sum(weights), 1.0) for weights in result.fold_weights)
    assert np.allclose(result.test, np.mean([
        np.dot(weights, test) for weights in result.fold_weights
    ], axis=0))
```

- [ ] **Step 2: Run blend test and verify red**

Run: `pytest tests/test_blending.py -v`

Expected: FAIL because `s6e8.blending` does not exist.

- [ ] **Step 3: Implement pure LOFO blending**

Create a frozen `BlendResult` dataclass with `oof`, `test`, `auc`, `fold_auc`, `fold_weights`, and `weight_mean`. For each held-out fold, search `_grid_weights(n_models, step)` on all other folds, break AUC ties lexicographically for determinism, score only the held-out rows, and accumulate one test prediction from the selected weights. Average fold test predictions for the final test vector.

Reject non-finite predictions, mismatched lengths, folds with one target class, and grids with no valid weights.

- [ ] **Step 4: Run blend unit tests and verify green**

Run: `pytest tests/test_blending.py -v`

Expected: PASS.

- [ ] **Step 5: Integrate `--method lofo_grid` into the CLI**

Update choices to include `lofo_grid` and make it the default. Load `fold` from OOF parquet. For legacy artifacts without `fold`, reconstruct exactly with `StratifiedKFold(5, shuffle=True, random_state=42)` in the first experiment's original row order, then align other experiments by id without re-sorting the reference order.

Save LOFO OOF with `fold`, test prediction, `fold_weights`, `weight_mean`, `fold_auc`, Pearson correlation, Spearman correlation, and per-fold component deltas. Label the old full-OOF `grid` method as `in_sample_weight_search: true` in metrics.

- [ ] **Step 6: Add CLI integration test and commit**

Use temporary two-model parquet/metrics fixtures, invoke the script through `subprocess.run`, and assert output metrics contain `method: lofo_grid`, four fold weights, and `in_sample_weight_search: false`.

Run: `pytest tests/test_blending.py -v`

Expected: PASS.

```bash
git add s6e8/blending.py scripts/blend_oof.py tests/test_blending.py
git commit -m "Add honest fold-wise blend evaluation"
```

---

### Task 6: Make Kaggle packaging backend- and dataset-aware

**Files:**
- Modify: `kaggle/runner.py:170-185`
- Modify: `kaggle/runner.py:252-263`
- Modify: `scripts/prepare_kaggle_kernel.py:116-152`
- Modify: `scripts/prepare_kaggle_kernel.py:155-201`
- Modify: `.github/workflows/kaggle-train.yml:3-33`
- Modify: `.github/workflows/kaggle-train.yml:141-185`
- Modify: `tests/test_cloud_helpers.py`
- Modify: `tests/test_prepare_kernel.py`

**Interfaces:**
- Consumes: selected config path, `model.name`, and top-level `external_reference.dataset_source`.
- Produces: required optional packages and Kaggle `dataset_sources` metadata.

- [ ] **Step 1: Write failing optional-dependency tests**

Add to `tests/test_cloud_helpers.py` using the loaded runner module:

```python
def test_optional_requirement_for_catboost():
    runner = load_script("../kaggle/runner.py")
    assert runner.optional_requirements({"model": {"name": "catboost"}}) == [
        ("catboost", "catboost>=1.2.8,<2")
    ]


def test_optional_requirement_for_lightgbm_is_empty():
    runner = load_script("../kaggle/runner.py")
    assert runner.optional_requirements({"model": {"name": "lightgbm"}}) == []
```

If `load_script` cannot resolve `../kaggle`, load it with `importlib.util.spec_from_file_location` using the repository root.

- [ ] **Step 2: Run dependency tests and verify red**

Run: `pytest tests/test_cloud_helpers.py -v`

Expected: FAIL because `optional_requirements` does not exist.

- [ ] **Step 3: Implement backend-specific installation**

Add:

```python
def optional_requirements(config: dict[str, Any]) -> list[tuple[str, str]]:
    backend = str(config["model"]["name"]).lower()
    if backend in {"catboost", "cat", "cb"}:
        return [("catboost", "catboost>=1.2.8,<2")]
    return []
```

Refactor installation to first ensure core `requirements.txt`, load the YAML config after core packages exist, then use `importlib.util.find_spec(module)` and install only missing optional requirements. Pass `ctx` to `maybe_install_requirements(root, ctx)` from `main`.

- [ ] **Step 4: Run dependency tests and verify green**

Run: `pytest tests/test_cloud_helpers.py -v`

Expected: PASS.

- [ ] **Step 5: Write failing dataset-source metadata test**

Extend the existing metadata test in `tests/test_prepare_kernel.py` with a config containing:

```python
config["external_reference"] = {
    "enabled": True,
    "dataset_source": "jayjoshi37/smartphone-usage-and-addiction-prediction",
}
```

Assert generated metadata contains exactly that slug in `dataset_sources` and still contains the official competition in `competition_sources`.

- [ ] **Step 6: Run metadata test and verify red**

Run: `pytest tests/test_prepare_kernel.py -v`

Expected: FAIL because `dataset_sources` is always empty.

- [ ] **Step 7: Add configured dataset sources to kernel metadata**

Add:

```python
def configured_dataset_sources(config: dict[str, Any]) -> list[str]:
    block = config.get("external_reference") or {}
    if not bool(block.get("enabled", False)):
        return []
    source = str(block.get("dataset_source", "")).strip()
    if not source or source.count("/") != 1:
        raise ValueError("external_reference.dataset_source must be owner/dataset")
    return [source]
```

Pass the list into `write_metadata` and store it in `metadata["dataset_sources"]`.

- [ ] **Step 8: Run cloud tests and commit**

Before the final cloud test, add a failing assertion that reads
`.github/workflows/kaggle-train.yml` and requires both
`"submit_to_kaggle" not in workflow_text` and
`"kaggle competitions submit" not in workflow_text`. Verify it fails, then
remove the submission input, `INPUT_SUBMIT`, and the complete optional-submit
step from the workflow. This preserves CSV generation as an artifact while
making leaderboard submission impossible through the repository workflow.

Run: `pytest tests/test_cloud_helpers.py tests/test_prepare_kernel.py -v`

Expected: PASS.

```bash
git add kaggle/runner.py scripts/prepare_kaggle_kernel.py .github/workflows/kaggle-train.yml tests/test_cloud_helpers.py tests/test_prepare_kernel.py
git commit -m "Package CatBoost and reference data for Kaggle"
```

---

### Task 7: Verify locally and publish the implementation commits

**Files:**
- Modify only files required by failures found during verification.

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: one reviewed GitHub branch state safe for formal cloud experiments.

- [ ] **Step 1: Run the full verification matrix**

Run:

```bash
pytest -q
python -m compileall -q s6e8 scripts kaggle
python scripts/validate_configs.py
python scripts/build_reports_index.py --check
python - <<'PY'
import json
from pathlib import Path
for path in Path("experiments").glob("*.json"):
    json.loads(path.read_text(encoding="utf-8"))
print("experiment json ok")
PY
git diff --check
```

Expected: every command exits zero. CatBoost smoke may be skipped only when CatBoost is not installed locally.

- [ ] **Step 2: Inspect repository scope**

Run: `git status --short && git diff --stat origin/agent/histgb-nocat-long-v1...HEAD`

Expected: `research_bundle/` remains untracked and no raw data, public notebook, OOF, test prediction, submission, credentials, or `.kernel-staging` file is staged.

- [ ] **Step 3: Request independent code review**

Use `superpowers:requesting-code-review` against the branch diff. Resolve every Critical or Important finding with a test-first fix and rerun Step 1.

- [ ] **Step 4: Publish commits to Draft PR #11**

Push with the GitHub connector when local HTTPS credentials are unavailable. Verify PR #11 remains open and Draft, and its head SHA matches the published implementation commit.

---

### Task 8: Run formal Kaggle ablations and record evidence

**Files:**
- Create after successful runs: `experiments/catboost_numeric_v1.json`
- Create after successful runs: `experiments/catboost_exactcat_v1.json`
- Create after successful runs: `experiments/catboost_exactcat_budget_v1.json`
- Create conditionally after successful runs: no-age, lattice, and reference experiment JSON files.
- Modify after successful runs: `experiments/LOG.md`
- Modify after successful runs: `README.md`
- Modify: PR #11 title/body through the GitHub connector.

**Interfaces:**
- Consumes: published configs and GitHub/Kaggle credentials already configured for the repository workflow.
- Produces: real cloud metrics, downloadable OOF/test artifacts, comparison tables, a tree-ceiling decision, and an updated Draft PR.

- [ ] **Step 1: Dispatch the control experiment**

Run the existing `Kaggle Train` workflow on the PR head with:

```text
config=configs/catboost_numeric_v1.yaml
accelerator=gpu
kernel_slug=s6e8-catboost-numeric-v1
gpu_machine_shape=NvidiaTeslaT4
```

The submission input has been removed in Task 6. If workflow dispatch is
unavailable to the connector, add a temporary push trigger scoped to the PR
branch and workflow file, hard-code the config, accelerator, slug, and GPU shape,
run it, and remove the trigger in the next commit. The temporary workflow must
not contain a competition-submit command.

- [ ] **Step 2: Validate the downloaded control artifact**

Require workflow success, Kaggle kernel completion, five fold scores, 691,369 OOF rows, 296,302 test rows, finite predictions, five stored fold ids, resolved config, and a matching git commit. A failed check blocks the next experiment.

- [ ] **Step 3: Dispatch exact-category and budget experiments sequentially**

Run `catboost_exactcat_v1`, validate it, then run `catboost_exactcat_budget_v1` and validate it. Use distinct kernel slugs. Download each artifact and calculate immediate-parent per-fold deltas plus LOFO blends with `lgbm_nocat`, `histgb_nocat_long_v1`, and the existing 60/40 blend components.

- [ ] **Step 4: Apply the promotion rule**

Promote an experiment when overall OOF improves without depending on a single exceptional fold and either at least three folds improve or honest LOFO blend AUC improves. Record negative and null results as well as positive results.

- [ ] **Step 5: Run the lattice ablation only after budget validation**

First run exactly one targeted no-age child of the promoted exact/budget frontier: retain numeric `age`, remove only `age__exact`, and do not start a general backward-selection sweep. Then dispatch the isolated lattice child of whichever internal parent is promoted. If its OOF and honest blend both fail to improve that parent, stop the lattice family and do not add more decimal digits.

- [ ] **Step 6: Run target-free reference distribution**

Dispatch a target-free reference-distribution child of the promoted internal parent. Verify provenance reports source SHA-256, raw/unique/duplicate/overlap/retained counts, source URL, and `external_supervision: false`.

- [ ] **Step 7: Gate the label-aware reference run**

Run `catboost_exactcat_budget_reflabel_v1` only when distribution features execute correctly and do not reveal suspicious exact lookup behavior. Require `external_supervision: true`; compare it separately and never silently replace the target-free result.

- [ ] **Step 8: Record real metrics**

For each successful formal run, copy the compact experiment summary into `experiments/<name>.json`, preserving the cloud run URL, git SHA, runtime, overall AUC, fold scores, best iterations, feature count, categorical count, and provenance. Do not commit large OOF/test/submission files.

- [ ] **Step 9: Decide the Lookup-Transformer gate**

Trigger the conditional Lookup plan when exact-grid signal is stable and two successive successful target-free tree candidates improve the best-so-far frontier by less than `0.00015`, or when a weaker tree improves honest LOFO by at least `0.00010` with at least three positive folds. Do not count label-aware source features toward the tree frontier. Record `lookup_transformer_decision`, evidence, and selected parent feature set in the experiment log.

- [ ] **Step 10: Update documentation and PR**

Update README and `experiments/LOG.md` with a five-fold table, per-fold deltas, correlations, LOFO blend results, external-supervision label, cloud links, and limitations. Update Draft PR #11 with verification evidence and the explicit sentence: `No Kaggle leaderboard submission was made.`

- [ ] **Step 11: Verify result records and commit**

Run the full Task 7 verification matrix, then:

```bash
git add experiments README.md
git commit -m "Record structural CatBoost cloud experiments"
```

Publish through the GitHub connector and keep the pull request Draft until final independent review completes.
