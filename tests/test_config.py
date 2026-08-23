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


def test_new_modeling_configs_declare_a_hypothesis():
    from pathlib import Path

    required = {
        "catboost_exactcat_v1",
        "catboost_numeric_v1",
        "catboost_exactcat_budget_v1",
        "catboost_exactcat_budget_refdist_v1",
        "catboost_exactcat_budget_seed7",
        "catboost_exactcat_budget_seed2026",
        "catboost_exactcat_budget_gpu_v1",
        "catboost_exactcat_budget_gpu_depth6_v1",
        "entity_mlp_hash_v1",
        "histgb_nocat_long_v1",
        "lgbm_freq_v1",
        "lgbm_nocat_mono",
        "mlp_nocat",
        "xgb_nocat",
    }
    found = set()
    for path in Path("configs").glob("*.yaml"):
        config = load_config(path)
        name = config["experiment"]["name"]
        found.add(name)
        if name in required:
            assert config["experiment"].get("hypothesis")
            assert config["experiment"].get("change")
            assert config["output"]["save_oof"] is True
            assert config["output"]["save_test"] is True
    assert required.issubset(found)


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


def test_blend_help_mentions_stack_methods():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/blend_oof.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--method" in proc.stdout
    assert "stack_logistic" in proc.stdout
    assert "stack_ridge" in proc.stdout
    assert "mean" in proc.stdout
    assert "catboost_exactcat_budget_seedavg" in proc.stdout


def test_catboost_seed_configs_differ_only_by_seed_and_name():
    from copy import deepcopy
    from pathlib import Path

    import yaml

    base = yaml.safe_load(Path("configs/catboost_exactcat_budget_v1.yaml").read_text(encoding="utf-8"))
    variants = {
        7: yaml.safe_load(Path("configs/catboost_exactcat_budget_seed7.yaml").read_text(encoding="utf-8")),
        2026: yaml.safe_load(
            Path("configs/catboost_exactcat_budget_seed2026.yaml").read_text(encoding="utf-8")
        ),
    }
    for seed, cfg in variants.items():
        assert cfg["experiment"]["seed"] == seed
        assert cfg["experiment"]["name"] == f"catboost_exactcat_budget_seed{seed}"
        left = deepcopy(base)
        right = deepcopy(cfg)
        for blob in (left, right):
            blob["experiment"].pop("name")
            blob["experiment"].pop("seed")
            blob["experiment"].pop("hypothesis")
            blob["experiment"].pop("change")
        assert left == right


def test_gpu_catboost_configs_isolate_accelerator_and_depth():
    from pathlib import Path

    import yaml

    cpu = yaml.safe_load(Path("configs/catboost_exactcat_budget_v1.yaml").read_text(encoding="utf-8"))
    gpu = yaml.safe_load(Path("configs/catboost_exactcat_budget_gpu_v1.yaml").read_text(encoding="utf-8"))
    depth6 = yaml.safe_load(
        Path("configs/catboost_exactcat_budget_gpu_depth6_v1.yaml").read_text(encoding="utf-8")
    )
    assert cpu["runtime"]["accelerator"] == "cpu"
    assert gpu["runtime"]["accelerator"] == "gpu"
    assert depth6["runtime"]["accelerator"] == "gpu"
    assert gpu["model"]["params"]["depth"] == cpu["model"]["params"]["depth"] == 8
    assert depth6["model"]["params"]["depth"] == 6
    assert gpu["model"]["name"] == "catboost"
    assert "device_type" not in gpu["model"]["params"]

