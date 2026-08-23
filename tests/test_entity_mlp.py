from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from s6e8.data import load_config, split_xy
from s6e8.models.entity_mlp import hashed_bucket_ids, torch_available
from s6e8.models.train import resolve_backend, train_cv


def test_hashed_bucket_ids_are_stable_and_reserve_zero_for_na():
    s = pd.Series(["age=24", "age=24", np.nan, "age=18"])
    a = hashed_bucket_ids(s, 16)
    b = hashed_bucket_ids(s, 16)
    np.testing.assert_array_equal(a, b)
    assert a[2] == 0
    assert a[0] == a[1]
    assert a[0] != 0
    assert a[3] != a[0]


def test_entity_mlp_backend_is_registered():
    config = load_config("configs/entity_mlp_hash_v1.yaml")
    assert resolve_backend(config) == "entity_mlp"
    assert config["experiment"]["name"] == "entity_mlp_hash_v1"
    assert config["features"]["exact_categorical"]["enabled"] is True


def test_entity_mlp_raises_without_torch():
    if torch_available():
        pytest.skip("torch is installed; ImportError path is tested only when missing")

    from s6e8.models.entity_mlp import fold_predict

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b__exact": ["a=1", "a=2", "a=3", "a=4"]})
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ImportError, match="Install torch"):
        fold_predict(
            frame,
            y,
            frame,
            y,
            frame,
            cat_cols=["b__exact"],
            seed=0,
            accelerator="cpu",
            params={"epochs": 1, "batch_size": 2, "hash_buckets": 8, "hidden_dims": [4]},
        )


def _synthetic_frames(n_train: int = 80, n_test: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i in range(n_train + n_test):
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


def test_smoke_entity_mlp_if_torch_installed(tmp_path):
    pytest.importorskip("torch")
    train_df, test_df = _synthetic_frames(n_train=60, n_test=16)
    raw = yaml.safe_load(Path("configs/entity_mlp_hash_v1.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "synthetic_entity_mlp"
    raw["paths"]["train"] = str(tmp_path / "train.csv")
    raw["paths"]["test"] = str(tmp_path / "test.csv")
    raw["paths"]["sample_submission"] = str(tmp_path / "missing.csv")
    raw["paths"]["oof_dir"] = str(tmp_path / "oof")
    raw["paths"]["submission_dir"] = str(tmp_path / "submissions")
    raw["paths"]["experiments_dir"] = str(tmp_path / "experiments")
    raw["cv"]["n_splits"] = 2
    raw["model"]["params"]["epochs"] = 2
    raw["model"]["params"]["batch_size"] = 16
    raw["model"]["params"]["hidden_dims"] = [8]
    raw["model"]["params"]["hash_buckets"] = 16
    raw["model"]["params"]["patience"] = 2
    train_df.to_csv(raw["paths"]["train"], index=False)
    test_df.to_csv(raw["paths"]["test"], index=False)
    cfg_path = tmp_path / "entity.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(cfg_path)
    X_train, y = split_xy(train_df, config)
    artifacts = train_cv(X_train, test_df, y, config)
    assert 0.0 <= artifacts["oof_auc"] <= 1.0
    assert artifacts["backend"] == "entity_mlp"
    assert any(name.endswith("__exact") for name in artifacts["feature_names"])
    assert len(artifacts["oof"]) == len(train_df)
    assert len(artifacts["test_pred"]) == len(test_df)
