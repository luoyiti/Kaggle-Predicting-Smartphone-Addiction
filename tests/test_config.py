from __future__ import annotations

from s6e8.data import load_config
from s6e8.runtime import validate_config


def test_baseline_yaml_loads(baseline_config_path):
    config = load_config(baseline_config_path)
    validate_config(config)
    assert config["experiment"]["name"] == "baseline"
    assert config["experiment"]["seed"] == 42
    assert config["runtime"]["accelerator"] == "cpu"
    assert config["model"]["name"] == "lightgbm"


def test_new_train_configs_document_hypothesis():
    from pathlib import Path

    required = (
        "lgbm_nocat_missflags.yaml",
        "lgbm_nocat_interactions.yaml",
        "catboost_nocat.yaml",
        "extratrees_nocat.yaml",
        "xgb_nocat.yaml",
    )
    for name in required:
        config = load_config(Path("configs") / name)
        assert config["experiment"].get("hypothesis")
        assert config["experiment"].get("change")
        assert config["experiment"].get("primary_variable")


def test_experiment_names_are_unique():
    from pathlib import Path

    names = []
    for path in sorted(Path("configs").glob("*.yaml")):
        config = load_config(path)
        validate_config(config)
        names.append(config["experiment"]["name"])
    assert names
    assert len(names) == len(set(names))


def test_train_help_mentions_config():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/train.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in proc.stdout
    assert "--accelerator" in proc.stdout
    assert "--max-train-rows" in proc.stdout
