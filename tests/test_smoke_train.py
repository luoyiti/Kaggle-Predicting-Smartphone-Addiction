"""Tiny synthetic-data training smoke test. Not competition data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from s6e8.data import load_config, split_xy
from s6e8.models.train import assert_oof_available, save_artifacts, train_cv


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


def test_smoke_mlp_backend(tmp_path, baseline_config_path):
    train_df, test_df = _synthetic_frames(n_train=120, n_test=30)
    raw = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "synthetic_mlp"
    raw["paths"]["train"] = str(tmp_path / "train.csv")
    raw["paths"]["test"] = str(tmp_path / "test.csv")
    raw["paths"]["sample_submission"] = str(tmp_path / "missing.csv")
    raw["paths"]["oof_dir"] = str(tmp_path / "oof")
    raw["paths"]["submission_dir"] = str(tmp_path / "submissions")
    raw["paths"]["experiments_dir"] = str(tmp_path / "experiments")
    raw["cv"]["n_splits"] = 2
    raw["model"]["name"] = "mlp"
    raw["model"]["params"] = {
        "hidden_layer_sizes": [8],
        "max_iter": 40,
        "early_stopping": True,
        "validation_fraction": 0.2,
    }
    raw["features"]["drop"] = ["gender", "stress_level", "academic_work_impact"]
    train_df.to_csv(raw["paths"]["train"], index=False)
    test_df.to_csv(raw["paths"]["test"], index=False)
    cfg_path = tmp_path / "mlp.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(cfg_path)
    X_train, y = split_xy(train_df, config)
    artifacts = train_cv(X_train, test_df, y, config)
    assert 0.0 <= artifacts["oof_auc"] <= 1.0
    assert len(artifacts["oof"]) == len(train_df)


def test_smoke_frequency_encoding_adds_fold_features(tmp_path, baseline_config_path):
    train_df, test_df = _synthetic_frames()
    raw = yaml.safe_load(Path("configs/lgbm_freq_v1.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "synthetic_freq"
    raw["paths"]["train"] = str(tmp_path / "train.csv")
    raw["paths"]["test"] = str(tmp_path / "test.csv")
    raw["paths"]["sample_submission"] = str(tmp_path / "missing.csv")
    raw["paths"]["oof_dir"] = str(tmp_path / "oof")
    raw["paths"]["submission_dir"] = str(tmp_path / "submissions")
    raw["paths"]["experiments_dir"] = str(tmp_path / "experiments")
    raw["cv"]["n_splits"] = 2
    raw["model"]["num_boost_round"] = 20
    raw["model"]["early_stopping_rounds"] = 5
    raw["model"]["log_evaluation"] = 0
    train_df.to_csv(raw["paths"]["train"], index=False)
    test_df.to_csv(raw["paths"]["test"], index=False)
    cfg_path = tmp_path / "freq.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(cfg_path)
    X_train, y = split_xy(train_df, config)
    artifacts = train_cv(X_train, test_df, y, config)
    assert any(name.endswith("_freq") for name in artifacts["feature_names"])
    assert artifacts.get("freq_fold_stats")


def test_smoke_reference_features_on_synthetic_original(tmp_path, baseline_config_path):
    train_df, test_df = _synthetic_frames()
    original_rows = []
    for i in range(30):
        original_rows.append(
            {
                "age": 20 + (i % 15),
                "gender": ["Male", "Female", "Other"][i % 3],
                "daily_screen_time_hours": 1.5 + (i % 7),
                "social_media_hours": 0.5 + (i % 4),
                "gaming_hours": 0.2 + (i % 3),
                "work_study_hours": 1.0 + (i % 5),
                "sleep_hours": 6.0 + (i % 3),
                "notifications_per_day": 8 + i,
                "app_opens_per_day": 4 + (i % 9),
                "weekend_screen_time": 2.0 + (i % 6),
                "stress_level": ["Low", "Medium", "High"][i % 3],
                "academic_work_impact": ["Yes", "No"][i % 2],
                "addicted_label": i % 2,
                "addiction_level": ["Low", "Moderate", "Severe"][i % 3],
            }
        )
    original_path = tmp_path / "original.csv"
    pd.DataFrame(original_rows).to_csv(original_path, index=False)

    raw = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "synthetic_refdist"
    raw["paths"]["train"] = str(tmp_path / "train.csv")
    raw["paths"]["test"] = str(tmp_path / "test.csv")
    raw["paths"]["sample_submission"] = str(tmp_path / "missing.csv")
    raw["paths"]["oof_dir"] = str(tmp_path / "oof")
    raw["paths"]["submission_dir"] = str(tmp_path / "submissions")
    raw["paths"]["experiments_dir"] = str(tmp_path / "experiments")
    raw["cv"]["n_splits"] = 2
    raw["model"]["num_boost_round"] = 20
    raw["model"]["early_stopping_rounds"] = 5
    raw["model"]["log_evaluation"] = 0
    raw["features"]["drop"] = ["gender", "stress_level", "academic_work_impact"]
    raw["features"]["engineering"] = {
        "add_n_missing": False,
        "add_leisure_hours": False,
        "add_screen_sleep_ratio": False,
        "add_weekend_weekday_ratio": False,
        "add_notif_per_open": False,
    }
    raw["features"]["reference"] = {
        "enabled": True,
        "path": str(original_path),
        "dataset_source": "fixture/original",
        "mode": "distribution",
        "use_labels": False,
        "remove_query_overlaps": True,
        "predictor_columns": [
            "age",
            "gender",
            "daily_screen_time_hours",
            "social_media_hours",
            "gaming_hours",
            "work_study_hours",
            "sleep_hours",
            "notifications_per_day",
            "app_opens_per_day",
            "weekend_screen_time",
            "stress_level",
            "academic_work_impact",
        ],
        "cdf_columns": ["daily_screen_time_hours"],
        "distance_columns": ["daily_screen_time_hours"],
        "frequency_columns": ["notifications_per_day"],
        "knn": {
            "enabled": True,
            "n_neighbors": 3,
            "columns": [
                "daily_screen_time_hours",
                "social_media_hours",
                "gaming_hours",
                "work_study_hours",
                "weekend_screen_time",
            ],
        },
    }
    train_df.to_csv(raw["paths"]["train"], index=False)
    test_df.to_csv(raw["paths"]["test"], index=False)
    cfg_path = tmp_path / "ref.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(cfg_path)
    X_train, y = split_xy(train_df, config)
    artifacts = train_cv(X_train, test_df, y, config)
    assert any(name.startswith("ref_") for name in artifacts["feature_names"])
    assert "ref_knn_mean_dist" in artifacts["feature_names"]
    assert artifacts.get("reference_features", {}).get("external_supervision") is False
    assert 0.0 <= artifacts["oof_auc"] <= 1.0


def test_smoke_exactcat_lightgbm(tmp_path):
    train_df, test_df = _synthetic_frames()
    raw = yaml.safe_load(Path("configs/lgbm_exactcat_v1.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "synthetic_exactcat"
    raw["paths"]["train"] = str(tmp_path / "train.csv")
    raw["paths"]["test"] = str(tmp_path / "test.csv")
    raw["paths"]["sample_submission"] = str(tmp_path / "missing.csv")
    raw["paths"]["oof_dir"] = str(tmp_path / "oof")
    raw["paths"]["submission_dir"] = str(tmp_path / "submissions")
    raw["paths"]["experiments_dir"] = str(tmp_path / "experiments")
    raw["cv"]["n_splits"] = 2
    raw["model"]["num_boost_round"] = 20
    raw["model"]["early_stopping_rounds"] = 5
    raw["model"]["log_evaluation"] = 0
    train_df.to_csv(raw["paths"]["train"], index=False)
    test_df.to_csv(raw["paths"]["test"], index=False)
    cfg_path = tmp_path / "exactcat.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(cfg_path)
    X_train, y = split_xy(train_df, config)
    artifacts = train_cv(X_train, test_df, y, config)
    assert any(name.endswith("__exact") for name in artifacts["feature_names"])
    assert artifacts["cat_cols"]
    assert all(c.endswith("__exact") for c in artifacts["cat_cols"])


def test_monotone_constraints_follow_feature_order():
    from s6e8.models.train import apply_monotone_constraints

    params = {"learning_rate": 0.05}
    config = {
        "model": {
            "monotone_constraints": {
                "daily_screen_time_hours": 1,
                "sleep_hours": -1,
            }
        }
    }
    names = ["age", "daily_screen_time_hours", "sleep_hours"]
    out = apply_monotone_constraints(params, names, config, "lightgbm")
    assert out["monotone_constraints"] == [0, 1, -1]
    xgb = apply_monotone_constraints(params, names, config, "xgboost")
    assert xgb["monotone_constraints"] == "(0,1,-1)"


def test_smoke_catboost_exactcat_if_installed(tmp_path):
    pytest.importorskip("catboost")
    train_df, test_df = _synthetic_frames()
    raw = yaml.safe_load(Path("configs/catboost_exactcat_v1.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "synthetic_cb"
    raw["paths"]["train"] = str(tmp_path / "train.csv")
    raw["paths"]["test"] = str(tmp_path / "test.csv")
    raw["paths"]["sample_submission"] = str(tmp_path / "missing.csv")
    raw["paths"]["oof_dir"] = str(tmp_path / "oof")
    raw["paths"]["submission_dir"] = str(tmp_path / "submissions")
    raw["paths"]["experiments_dir"] = str(tmp_path / "experiments")
    raw["cv"]["n_splits"] = 2
    raw["model"]["num_boost_round"] = 20
    raw["model"]["early_stopping_rounds"] = 5
    raw["model"]["log_evaluation"] = 0
    raw["model"]["params"]["allow_writing_files"] = False
    train_df.to_csv(raw["paths"]["train"], index=False)
    test_df.to_csv(raw["paths"]["test"], index=False)
    cfg_path = tmp_path / "cb.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(cfg_path)
    X_train, y = split_xy(train_df, config)
    artifacts = train_cv(X_train, test_df, y, config)
    assert 0.0 <= artifacts["oof_auc"] <= 1.0
    assert any(name.endswith("__exact") for name in artifacts["feature_names"])


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
