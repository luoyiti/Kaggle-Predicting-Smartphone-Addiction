from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from s6e8.data import load_config
from s6e8.models.train import save_artifacts
from s6e8.runtime import dependency_versions


def test_dependency_versions_are_deterministic_and_backend_specific(monkeypatch):
    installed = {
        "numpy": "2.0.0",
        "pandas": "2.2.0",
        "scikit-learn": "1.5.0",
        "lightgbm": "4.5.0",
        "catboost": "1.2.8",
    }

    monkeypatch.setattr(
        "s6e8.runtime._installed_distribution_version", installed.get
    )
    versions = dependency_versions("catboost")

    assert list(versions) == [
        "python",
        "numpy",
        "pandas",
        "scikit-learn",
        "pyarrow",
        "lightgbm",
        "s6e8",
        "catboost",
    ]
    assert versions["catboost"] == "1.2.8"
    assert versions["lightgbm"] == "4.5.0"
    assert versions["pyarrow"] is None


def test_catboost_dependency_versions_are_written_to_both_json_artifacts(
    tmp_path, baseline_config_path, monkeypatch
):
    config = load_config(baseline_config_path)
    config["experiment"]["name"] = "catboost_artifact_contract"
    config["model"]["name"] = "catboost"
    config["paths"]["oof_dir"] = str(tmp_path / "oof")
    config["paths"]["submission_dir"] = str(tmp_path / "submissions")
    config["paths"]["experiments_dir"] = str(tmp_path / "experiments")
    config["output"]["save_submission"] = False

    expected = {
        "python": "3.11.0",
        "numpy": "2.0.0",
        "pandas": "2.2.0",
        "scikit-learn": "1.5.0",
        "pyarrow": None,
        "lightgbm": None,
        "s6e8": "0.1.0",
        "catboost": "1.2.8",
    }
    monkeypatch.setattr(
        "s6e8.models.train.dependency_versions", lambda _backend: expected
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=False: Path(path).write_bytes(b"parquet-stub"),
    )
    artifacts = {
        "oof": np.array([0.1, 0.9]),
        "test_pred": np.array([0.4]),
        "fold_ids": np.array([0, 1]),
        "train_ids": np.array([1, 2]),
        "test_ids": np.array([3]),
        "y": np.array([0, 1]),
        "fold_scores": [1.0, 1.0],
        "best_iterations": [1, 1],
        "cv_mean": 1.0,
        "cv_std": 0.0,
        "oof_auc": 1.0,
        "feature_names": ["age"],
        "cat_cols": [],
        "data_provenance": {},
        "backend": "catboost",
    }

    written = save_artifacts(artifacts, config)
    metrics = json.loads(Path(written["metrics"]).read_text(encoding="utf-8"))
    experiment = json.loads(Path(written["experiment"]).read_text(encoding="utf-8"))

    assert metrics["dependency_versions"] == expected
    assert experiment["dependency_versions"] == expected
    assert "catboost" in metrics["dependency_versions"]
