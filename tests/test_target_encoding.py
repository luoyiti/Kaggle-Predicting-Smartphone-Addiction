"""Fold-safe exact-value target encoding. Synthetic frames only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from s6e8.data import load_config
from s6e8.features import add_engineered_features
from s6e8.target_encoding import (
    ExactValueTargetEncoder,
    apply_fold_target_encoding,
    parse_exact_te_config,
    te_feature_name,
)


def test_parse_disabled_or_empty_returns_none():
    assert parse_exact_te_config({"features": {}}) is None
    assert (
        parse_exact_te_config(
            {"features": {"engineering": {"exact_target_encoding": {"enabled": False, "columns": ["age"]}}}}
        )
        is None
    )
    assert (
        parse_exact_te_config(
            {"features": {"engineering": {"exact_target_encoding": {"enabled": True, "columns": []}}}}
        )
        is None
    )


def test_loo_does_not_use_a_row_own_label():
    X = pd.DataFrame({"x": [1.0, 1.0, 1.0, 2.0]})
    y = np.array([1.0, 0.0, 1.0, 0.0])
    enc = ExactValueTargetEncoder(
        columns=["x"], smoothing=0.0, min_count=1, round_decimals=2, leave_one_out_train=True
    )
    enc.fit(X, y)
    out = enc.transform(X, y, leave_one_out=True)
    te = out["x_exact_te"].to_numpy()
    # key=1 appears three times with labels 1,0,1. LOO for row0 uses (0+1)/2 = 0.5
    assert te[0] == 0.5
    # row1 (label 0) uses (1+1)/2 = 1.0
    assert te[1] == 1.0
    # key=2 appears once: LOO count=0 -> prior 0.5
    assert te[3] == enc.prior


def test_validation_encoding_ignores_validation_labels():
    X_tr = pd.DataFrame({"x": [1.11, 1.11, 2.22]})
    y_tr = np.array([1.0, 0.0, 1.0])
    X_va = pd.DataFrame({"x": [1.11, 3.33]})
    X_te = pd.DataFrame({"x": [1.11]})
    cfg = {
        "columns": ["x"],
        "smoothing": 0.0,
        "min_count": 1,
        "round_decimals": 2,
        "suffix": "_exact_te",
        "leave_one_out_train": True,
    }
    tr, va, te, stats = apply_fold_target_encoding(X_tr, y_tr, X_va, X_te, cfg)
    # train-fold mean for 1.11 is 0.5, independent of whatever the val labels would be
    assert va["x_exact_te"].iloc[0] == 0.5
    assert te["x_exact_te"].iloc[0] == 0.5
    # unseen 3.33 -> prior (2/3)
    assert abs(va["x_exact_te"].iloc[1] - (2.0 / 3.0)) < 1e-12
    assert stats["columns"]["x"]["n_val_unseen"] == 1


def test_missing_stays_missing_and_smoothing_shrinks_rare_values():
    X = pd.DataFrame({"x": [1.0, 1.0, np.nan]})
    y = np.array([1.0, 1.0, 0.0])
    enc = ExactValueTargetEncoder(columns=["x"], smoothing=20.0, min_count=5, round_decimals=2)
    enc.fit(X, y)
    out = enc.transform(X, leave_one_out=False)
    assert pd.isna(out["x_exact_te"].iloc[2])
    # count=2 < min_count=5 -> prior, even though both observed labels are 1
    assert out["x_exact_te"].iloc[0] == enc.prior
    assert out["x_exact_te"].iloc[1] == enc.prior


def test_row_wise_transform_does_not_add_te_columns(tmp_path):
    raw = yaml.safe_load(open("configs/lgbm_nocat_exact_te_v1.yaml", encoding="utf-8"))
    raw["experiment"]["name"] = "te_flag_test"
    raw["paths"]["train"] = str(tmp_path / "train.csv")
    raw["paths"]["test"] = str(tmp_path / "test.csv")
    path = tmp_path / "te.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(path)
    df = pd.DataFrame(
        {
            "id": [0, 1],
            "age": [20, 25],
            "daily_screen_time_hours": [8.0, 4.0],
            "social_media_hours": [2.0, 3.0],
            "gaming_hours": [1.0, 0.5],
            "work_study_hours": [2.0, 1.0],
            "sleep_hours": [7.0, 6.0],
            "notifications_per_day": [10, 20],
            "app_opens_per_day": [5, 10],
            "weekend_screen_time": [10.0, 12.0],
            "gender": ["Male", "Female"],
            "stress_level": ["Low", "High"],
            "academic_work_impact": ["Yes", "No"],
            "addicted_label": [1, 0],
        }
    )
    out = add_engineered_features(df, config)
    assert "notifications_per_day_exact_te" not in out.columns
    parsed = parse_exact_te_config(config)
    assert parsed is not None
    assert "notifications_per_day" in parsed["columns"]
    assert te_feature_name("notifications_per_day") == "notifications_per_day_exact_te"


def test_smoke_train_with_exact_te(tmp_path, baseline_config_path):
    from s6e8.data import split_xy
    from s6e8.models.train import save_artifacts, train_cv
    from tests.test_smoke_train import _synthetic_frames

    train_df, test_df = _synthetic_frames(n_train=80, n_test=20)
    raw = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "synthetic_exact_te"
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
    raw["features"]["engineering"] = {
        "add_n_missing": False,
        "add_missing_indicators": False,
        "add_leisure_hours": False,
        "add_screen_sleep_ratio": False,
        "add_weekend_weekday_ratio": False,
        "add_notif_per_open": False,
        "exact_target_encoding": {
            "enabled": True,
            "smoothing": 10.0,
            "min_count": 2,
            "round_decimals": 2,
            "leave_one_out_train": True,
            "columns": ["notifications_per_day", "app_opens_per_day"],
        },
    }
    train_df.to_csv(raw["paths"]["train"], index=False)
    test_df.to_csv(raw["paths"]["test"], index=False)
    cfg_path = tmp_path / "te_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(cfg_path)
    X_train, y = split_xy(train_df, config)
    artifacts = train_cv(X_train, test_df, y, config)
    artifacts["runtime_seconds"] = 0.01
    artifacts["git_commit"] = None
    written = save_artifacts(artifacts, config)
    assert "notifications_per_day_exact_te" in artifacts["feature_names"]
    assert "app_opens_per_day_exact_te" in artifacts["feature_names"]
    assert artifacts["te_fold_stats"]
    assert 0.0 <= artifacts["oof_auc"] <= 1.0
    import json

    metrics = json.loads((tmp_path / "oof" / "synthetic_exact_te" / "metrics.json").read_text())
    assert "target_encoding" in metrics
    assert "feature_importance_gain_mean" in metrics
    assert (tmp_path / written["oof_npy"]).exists() or written["oof_npy"].endswith("oof.npy")
