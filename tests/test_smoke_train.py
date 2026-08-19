"""Tiny synthetic-data training smoke test. Not competition data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from s6e8.data import load_config, split_xy
from s6e8.models.train import _prepare_xy, assert_oof_available, save_artifacts, train_cv


def _synthetic_frames(n_train: int = 80, n_test: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = list(range(n_train + n_test))
    rows = []
    for i in rng:
        rows.append(
            {
                "id": i,
                "age": 18 + (i % 40),
                "daily_screen_time_hours": 2 + (i % 8),
                "social_media_hours": i % 5,
                "gaming_hours": i % 4,
                "work_study_hours": 1 + (i % 6),
                "sleep_hours": 5 + (i % 4),
                "notifications_per_day": 10 + i,
                "app_opens_per_day": 5 + (i % 12),
                "weekend_screen_time": 3 + (i % 5),
                "gender": ["Male", "Female", "Other"][i % 3],
                "stress_level": ["Low", "Medium", "High"][i % 3],
                "academic_work_impact": ["Yes", "No"][i % 2],
                "addicted_label": i % 2,
            }
        )
    df = pd.DataFrame(rows)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].drop(columns=["addicted_label"]).copy()


def test_prepare_xy_routes_exact_copies_as_catboost_categories(
    tmp_path, baseline_config_path
):
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


def test_catboost_exact_category_smoke(tmp_path, baseline_config_path):
    pytest.importorskip("catboost")
    train_df, test_df = _synthetic_frames(120, 30)
    raw = yaml.safe_load(
        Path("configs/catboost_exactcat_budget_v1.yaml").read_text(encoding="utf-8")
    )
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


def test_smoke_train_on_synthetic_data(tmp_path, baseline_config_path):
    train_df, test_df = _synthetic_frames()
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    raw = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "synthetic_smoke"
    raw["paths"]["train"] = str(train_path)
    raw["paths"]["test"] = str(test_path)
    raw["paths"]["sample_submission"] = str(tmp_path / "missing.csv")
    raw["paths"]["oof_dir"] = str(tmp_path / "oof")
    raw["paths"]["submission_dir"] = str(tmp_path / "submissions")
    raw["paths"]["experiments_dir"] = str(tmp_path / "experiments")
    raw["cv"]["n_splits"] = 2
    raw["model"]["num_boost_round"] = 20
    raw["model"]["early_stopping_rounds"] = 5
    raw["model"]["log_evaluation"] = 0

    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(cfg_path)

    assert_oof_available(config, overwrite=False)
    X_train, y = split_xy(train_df, config)
    artifacts = train_cv(X_train, test_df, y, config)
    artifacts["runtime_seconds"] = 0.01
    artifacts["git_commit"] = None
    written = save_artifacts(artifacts, config)

    assert Path(written["oof_npy"]).exists()
    assert Path(written["test_npy"]).exists()
    assert Path(written["metrics"]).exists()
    assert Path(written["submission"]).exists()
    assert 0.0 <= artifacts["oof_auc"] <= 1.0

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

    try:
        assert_oof_available(config, overwrite=False)
        raise AssertionError("expected existing OOF to be rejected")
    except FileExistsError:
        pass
    assert_oof_available(config, overwrite=True)


def test_smoke_histgb_and_logreg_backends(tmp_path, baseline_config_path):
    train_df, test_df = _synthetic_frames()
    raw = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    raw["paths"]["train"] = str(tmp_path / "train.csv")
    raw["paths"]["test"] = str(tmp_path / "test.csv")
    raw["paths"]["sample_submission"] = str(tmp_path / "missing.csv")
    raw["paths"]["oof_dir"] = str(tmp_path / "oof")
    raw["paths"]["submission_dir"] = str(tmp_path / "submissions")
    raw["paths"]["experiments_dir"] = str(tmp_path / "experiments")
    raw["cv"]["n_splits"] = 2
    raw["model"]["log_evaluation"] = 0
    raw["features"]["engineering"] = {
        "add_n_missing": False,
        "add_leisure_hours": False,
        "add_screen_sleep_ratio": False,
        "add_weekend_weekday_ratio": False,
        "add_notif_per_open": False,
    }
    train_df.to_csv(raw["paths"]["train"], index=False)
    test_df.to_csv(raw["paths"]["test"], index=False)

    for backend, params in (
        ("histgb", {"max_iter": 15, "learning_rate": 0.1, "early_stopping": True, "n_iter_no_change": 5}),
        ("logreg", {"max_iter": 200, "solver": "lbfgs"}),
    ):
        raw["experiment"]["name"] = f"synthetic_{backend}"
        raw["model"]["name"] = backend
        raw["model"]["params"] = params
        raw["model"]["num_boost_round"] = 15
        raw["model"]["early_stopping_rounds"] = 5
        cfg_path = tmp_path / f"{backend}.yaml"
        cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        config = load_config(cfg_path)
        X_train, y = split_xy(train_df, config)
        artifacts = train_cv(X_train, test_df, y, config)
        assert 0.0 <= artifacts["oof_auc"] <= 1.0
        assert len(artifacts["oof"]) == len(train_df)


def test_histgb_fit_accepts_missing_x_val():
    """Kaggle images may ship sklearn < 1.7, where HistGB.fit has no X_val."""
    from s6e8.models.train import _filter_init_kwargs, fit_histgb

    class OldHistGB:
        def __init__(self, learning_rate=0.1, max_iter=100):
            self.learning_rate = learning_rate
            self.max_iter = max_iter
            self.fit_args = None

        def fit(self, X, y, sample_weight=None):
            self.fit_args = {"X": X, "y": y, "sample_weight": sample_weight}
            return self

    class NewHistGB:
        def __init__(self, learning_rate=0.1, max_iter=100, categorical_features="from_dtype"):
            self.learning_rate = learning_rate
            self.max_iter = max_iter
            self.categorical_features = categorical_features
            self.fit_args = None

        def fit(self, X, y, sample_weight=None, *, X_val=None, y_val=None):
            self.fit_args = {"X": X, "y": y, "X_val": X_val, "y_val": y_val}
            return self

    old = OldHistGB()
    fit_histgb(old, "Xtr", "ytr", "Xva", "yva")
    assert old.fit_args == {"X": "Xtr", "y": "ytr", "sample_weight": None}

    new = NewHistGB()
    fit_histgb(new, "Xtr", "ytr", "Xva", "yva")
    assert new.fit_args == {"X": "Xtr", "y": "ytr", "X_val": "Xva", "y_val": "yva"}

    filtered = _filter_init_kwargs(OldHistGB, {"learning_rate": 0.06, "categorical_features": "from_dtype"})
    assert filtered == {"learning_rate": 0.06}


def test_diagnostic_override_renames_experiment(baseline_config_path):
    from s6e8.models.train import apply_diagnostic_overrides

    config = load_config(baseline_config_path)
    apply_diagnostic_overrides(config, max_train_rows=1000, n_splits=3)
    assert config["experiment"]["name"] == "baseline_diag1000"
    assert config["experiment"]["diagnostic"] is True
    assert config["runtime"]["max_train_rows"] == 1000
    assert config["cv"]["n_splits"] == 3


def test_diagnostic_override_demotes_formal_run(baseline_config_path):
    from s6e8.models.train import apply_diagnostic_overrides

    config = load_config(baseline_config_path)
    config["experiment"]["formal"] = True
    config["experiment"]["validation_protocol"] = "fixed5_seed42_v1"
    apply_diagnostic_overrides(config, n_splits=3)

    assert config["experiment"]["formal"] is False
    assert config["experiment"]["source_formal"] is True
    assert config["experiment"]["diagnostic"] is True
    assert config["cv"]["n_splits"] == 3
