"""Unit tests for config-driven feature flags. Synthetic frames only."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from s6e8.data import load_config
from s6e8.features import add_engineered_features, feature_columns, transform


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "age": [20, 25, 30, 18],
            "daily_screen_time_hours": [8.0, np.nan, 4.0, 10.0],
            "social_media_hours": [2.0, 3.0, np.nan, 1.0],
            "gaming_hours": [1.0, 0.5, 0.0, 2.0],
            "work_study_hours": [2.0, 1.0, 3.0, 1.0],
            "sleep_hours": [7.0, 6.0, 8.0, 5.0],
            "notifications_per_day": [10, 20, 30, 40],
            "app_opens_per_day": [5, 10, 15, 20],
            "weekend_screen_time": [10.0, 12.0, 5.0, np.nan],
            "gender": ["Male", "Female", None, "Other"],
            "stress_level": ["Low", "High", "Medium", "Low"],
            "academic_work_impact": ["Yes", "No", "Yes", "No"],
            "addicted_label": [1, 1, 0, 1],
        }
    )


def _config(tmp_path: Path, engineering: dict, drop=None):
    raw = yaml.safe_load(Path("configs/baseline.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "feat_test"
    raw["paths"]["train"] = str(tmp_path / "train.csv")
    raw["paths"]["test"] = str(tmp_path / "test.csv")
    raw["paths"]["sample_submission"] = str(tmp_path / "missing.csv")
    raw["features"]["engineering"] = engineering
    if drop is not None:
        raw["features"]["drop"] = drop
    path = tmp_path / "feat.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def test_raw_engineering_adds_no_extra_columns(tmp_path):
    config = _config(
        tmp_path,
        {
            "add_n_missing": False,
            "add_missing_indicators": False,
            "add_leisure_hours": False,
            "add_screen_sleep_ratio": False,
            "add_weekend_weekday_ratio": False,
            "add_notif_per_open": False,
        },
    )
    out = add_engineered_features(_frame(), config)
    assert "n_missing" not in out.columns
    assert "leisure_hours" not in out.columns
    assert "strong3_row_mean" not in out.columns


def test_other_screen_is_nan_when_a_part_is_missing(tmp_path):
    config = _config(
        tmp_path,
        {
            "add_other_screen_hours": True,
            "other_screen": {
                "total": "daily_screen_time_hours",
                "parts": ["social_media_hours", "gaming_hours", "work_study_hours"],
            },
        },
    )
    out = add_engineered_features(_frame(), config)
    assert out.loc[0, "other_screen_hours"] == 3.0
    assert pd.isna(out.loc[1, "other_screen_hours"])
    assert pd.isna(out.loc[2, "other_screen_hours"])
    assert out.loc[3, "other_screen_hours"] == 6.0


def test_strong3_mean_covers_rows_with_partial_missing(tmp_path):
    config = _config(
        tmp_path,
        {
            "add_strong3_row_mean": True,
            "strong_usage_cols": [
                "daily_screen_time_hours",
                "weekend_screen_time",
                "social_media_hours",
            ],
        },
    )
    out = add_engineered_features(_frame(), config)
    assert out["strong3_row_mean"].notna().all()
    assert "strong3_row_mean" in feature_columns(transform(_frame(), config), config)


def test_drop_removes_columns(tmp_path):
    config = _config(
        tmp_path,
        {"add_n_missing": True},
        drop=["gender", "n_missing"],
    )
    cols = feature_columns(transform(_frame(), config), config)
    assert "gender" not in cols
    assert "n_missing" not in cols
    assert "daily_screen_time_hours" in cols
