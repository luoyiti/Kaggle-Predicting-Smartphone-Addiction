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


def test_experiment_names_are_unique():
    from pathlib import Path

    names = []
    for path in sorted(Path("configs").glob("*.yaml")):
        config = load_config(path)
        validate_config(config)
        names.append(config["experiment"]["name"])
    assert names
    assert len(names) == len(set(names))


def test_histgb_long_is_a_training_budget_only_experiment():
    """Catch feature/CV drift that would make the long-run comparison invalid."""
    from pathlib import Path

    control = load_config(Path("configs/histgb_nocat.yaml"))
    candidate = load_config(Path("configs/histgb_nocat_long_v1.yaml"))
    validate_config(candidate)

    assert candidate["experiment"]["name"] == "histgb_nocat_long_v1"
    assert candidate["experiment"]["seed"] == control["experiment"]["seed"]
    assert candidate["cv"] == control["cv"]
    assert candidate["features"] == control["features"]
    assert candidate["model"]["name"] == control["model"]["name"] == "histgb"

    control_model = dict(control["model"])
    candidate_model = dict(candidate["model"])
    control_params = dict(control_model.pop("params"))
    candidate_params = dict(candidate_model.pop("params"))
    assert candidate_model["num_boost_round"] > control_model["num_boost_round"]
    candidate_model["num_boost_round"] = control_model["num_boost_round"]
    assert candidate_model == control_model
    assert candidate_params == control_params


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
