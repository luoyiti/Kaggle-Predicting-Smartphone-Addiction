"""Structural feature flags. Synthetic frames only."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from s6e8.data import load_config
from s6e8.features import categorical_feature_columns, feature_columns, transform
from s6e8.structural_features import (
    add_exact_categorical_features,
    add_screen_budget_features,
    canonical_numeric_value,
    format_exact_keys,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [0, 1, 2],
            "age": [24.0, np.nan, 18.4],
            "daily_screen_time_hours": [8.0, 6.0, 10.0],
            "social_media_hours": [2.0, 3.0, 1.0],
            "gaming_hours": [1.0, 0.5, 2.0],
            "work_study_hours": [2.0, 1.0, 1.0],
            "sleep_hours": [7.0, 6.0, 5.0],
            "notifications_per_day": [10.0, 20.0, 30.0],
            "app_opens_per_day": [5.0, 10.0, 15.0],
            "weekend_screen_time": [10.0, 12.0, 8.0],
            "gender": ["Male", "Female", "Other"],
            "stress_level": ["Low", "High", "Medium"],
            "academic_work_impact": ["Yes", "No", "Yes"],
            "addicted_label": [1, 1, 0],
        }
    )


def _config(tmp_path: Path, features_extra: dict):
    raw = yaml.safe_load(Path("configs/lgbm_nocat.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "struct_test"
    raw["paths"]["train"] = str(tmp_path / "train.csv")
    raw["paths"]["test"] = str(tmp_path / "test.csv")
    raw["paths"]["sample_submission"] = str(tmp_path / "missing.csv")
    raw["features"].update(features_extra)
    path = tmp_path / "feat.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def test_canonical_numeric_value_rounds_and_missing():
    assert canonical_numeric_value(24.4, 0, "__MISSING__") == "24"
    assert canonical_numeric_value(8.126, 2, "__MISSING__") == "8.13"
    assert canonical_numeric_value(np.nan, 2, "__MISSING__") == "__MISSING__"


def test_format_exact_keys_vectorized(tmp_path):
    s = pd.Series([8.126, np.nan, 8.1])
    keys = format_exact_keys(s, 2, "__MISSING__")
    assert list(keys) == ["8.13", "__MISSING__", "8.10"]


def test_exact_categorical_copies_and_drop(tmp_path):
    config = _config(
        tmp_path,
        {
            "exact_categorical": {
                "enabled": True,
                "columns": ["age", "daily_screen_time_hours"],
                "suffix": "__exact",
                "decimal_places": {"age": 0, "daily_screen_time_hours": 2},
            }
        },
    )
    out = transform(_frame(), config)
    cols = feature_columns(out, config)
    assert "age__exact" in cols
    assert "daily_screen_time_hours__exact" in cols
    assert "gender" not in cols
    cats = categorical_feature_columns(out, config)
    assert "age__exact" in cats
    assert "gender" not in cats
    assert str(out.loc[0, "age__exact"]) == "age=24"
    assert str(out.loc[1, "age__exact"]) == "age=__MISSING__"
    raw = add_exact_categorical_features(_frame(), config)
    assert raw.loc[0, "daily_screen_time_hours__exact"] == "daily_screen_time_hours=8.00"


def test_screen_budget_remainder(tmp_path):
    config = _config(
        tmp_path,
        {"screen_budget": {"enabled": True, "tolerance": 1e-9}},
    )
    out = add_screen_budget_features(_frame(), config)
    # row0: 8 - (2+1+2) = 3
    assert out.loc[0, "screen_component_sum_complete"] == 5.0
    assert out.loc[0, "screen_remainder_complete"] == 3.0
    assert out.loc[0, "screen_budget_violation"] == 0.0
    assert out.loc[0, "awake_non_screen_hours"] == 24.0 - 7.0 - 8.0


def test_disabled_structural_blocks_add_nothing(tmp_path):
    config = _config(tmp_path, {"exact_categorical": {"enabled": False}})
    out = transform(_frame(), config)
    assert "age__exact" not in out.columns
    assert "screen_remainder_complete" not in out.columns
