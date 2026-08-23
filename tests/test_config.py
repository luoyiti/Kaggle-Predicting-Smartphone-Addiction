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
        "catboost_exactcat_budget_joint_v1",
        "catboost_exactcat_budget_identity_v1",
        "catboost_exactcat_budget_plain_v1",
        "catboost_exactcat_budget_bernoulli_v1",
        "entity_mlp_hash_v1",
        "histgb_nocat_long_v1",
        "lgbm_freq_v1",
        "lgbm_nocat_dart_v1",
        "lgbm_nocat_hardband_v1",
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


def test_joint_budget_config_isolates_joint_pairs():
    from copy import deepcopy
    from pathlib import Path

    import yaml

    base = yaml.safe_load(Path("configs/catboost_exactcat_budget_v1.yaml").read_text(encoding="utf-8"))
    joint = yaml.safe_load(
        Path("configs/catboost_exactcat_budget_joint_v1.yaml").read_text(encoding="utf-8")
    )
    assert joint["experiment"]["name"] == "catboost_exactcat_budget_joint_v1"
    assert joint["features"]["exact_categorical"]["joint_pairs"] == [
        ["notifications_per_day", "app_opens_per_day"]
    ]
    assert "joint_pairs" not in (base["features"].get("exact_categorical") or {})
    left = deepcopy(base)
    right = deepcopy(joint)
    for blob in (left, right):
        blob["experiment"].pop("name")
        blob["experiment"].pop("hypothesis")
        blob["experiment"].pop("change")
        blob["experiment"].pop("feature_version")
        blob["features"]["exact_categorical"].pop("joint_pairs", None)
        blob["features"]["exact_categorical"].pop("joint_suffix", None)
    assert left == right


def test_identity_budget_config_isolates_exact_columns():
    from copy import deepcopy
    from pathlib import Path

    import yaml

    base = yaml.safe_load(Path("configs/catboost_exactcat_budget_v1.yaml").read_text(encoding="utf-8"))
    identity = yaml.safe_load(
        Path("configs/catboost_exactcat_budget_identity_v1.yaml").read_text(encoding="utf-8")
    )
    assert identity["experiment"]["name"] == "catboost_exactcat_budget_identity_v1"
    assert identity["features"]["exact_categorical"]["columns"] == [
        "notifications_per_day",
        "app_opens_per_day",
    ]
    assert base["features"]["exact_categorical"]["columns"] == "auto_numeric"
    left = deepcopy(base)
    right = deepcopy(identity)
    for blob in (left, right):
        blob["experiment"].pop("name")
        blob["experiment"].pop("hypothesis")
        blob["experiment"].pop("change")
        blob["experiment"].pop("feature_version")
        blob["features"]["exact_categorical"].pop("columns")
    assert left == right


def test_plain_budget_config_isolates_boosting_type():
    from copy import deepcopy
    from pathlib import Path

    import yaml

    base = yaml.safe_load(Path("configs/catboost_exactcat_budget_v1.yaml").read_text(encoding="utf-8"))
    plain = yaml.safe_load(
        Path("configs/catboost_exactcat_budget_plain_v1.yaml").read_text(encoding="utf-8")
    )
    assert plain["experiment"]["name"] == "catboost_exactcat_budget_plain_v1"
    assert plain["model"]["params"]["boosting_type"] == "Plain"
    assert "boosting_type" not in (base["model"]["params"] or {})
    left = deepcopy(base)
    right = deepcopy(plain)
    for blob in (left, right):
        blob["experiment"].pop("name")
        blob["experiment"].pop("hypothesis")
        blob["experiment"].pop("change")
        blob["experiment"].pop("model_version")
        blob["model"]["params"].pop("boosting_type", None)
    assert left == right


def test_bernoulli_budget_config_isolates_bootstrap_type():
    from copy import deepcopy
    from pathlib import Path

    import yaml

    base = yaml.safe_load(Path("configs/catboost_exactcat_budget_v1.yaml").read_text(encoding="utf-8"))
    bernoulli = yaml.safe_load(
        Path("configs/catboost_exactcat_budget_bernoulli_v1.yaml").read_text(encoding="utf-8")
    )
    assert bernoulli["experiment"]["name"] == "catboost_exactcat_budget_bernoulli_v1"
    assert bernoulli["model"]["params"]["bootstrap_type"] == "Bernoulli"
    assert "bootstrap_type" not in (base["model"]["params"] or {})
    left = deepcopy(base)
    right = deepcopy(bernoulli)
    for blob in (left, right):
        blob["experiment"].pop("name")
        blob["experiment"].pop("hypothesis")
        blob["experiment"].pop("change")
        blob["experiment"].pop("model_version")
        blob["model"]["params"].pop("bootstrap_type", None)
    assert left == right


def test_hardband_config_isolates_hard_band():
    from copy import deepcopy
    from pathlib import Path

    import yaml

    base = yaml.safe_load(Path("configs/lgbm_nocat.yaml").read_text(encoding="utf-8"))
    hard = yaml.safe_load(Path("configs/lgbm_nocat_hardband_v1.yaml").read_text(encoding="utf-8"))
    assert hard["experiment"]["name"] == "lgbm_nocat_hardband_v1"
    hb = hard["features"]["hard_band"]
    assert hb["enabled"] is True
    assert hb["mode"] == "train_only"
    assert hb["source_experiment"] == "lgbm_nocat"
    assert hb["lo"] == 0.3
    assert hb["hi"] == 0.7
    left = deepcopy(base)
    right = deepcopy(hard)
    for blob in (left, right):
        blob["experiment"].pop("name")
        blob["experiment"].pop("hypothesis")
        blob["experiment"].pop("change")
        blob["experiment"].pop("feature_version")
        blob["features"].pop("hard_band", None)
    assert left == right


def test_dart_config_isolates_boosting_type():
    from copy import deepcopy
    from pathlib import Path

    import yaml

    base = yaml.safe_load(Path("configs/lgbm_nocat.yaml").read_text(encoding="utf-8"))
    dart = yaml.safe_load(Path("configs/lgbm_nocat_dart_v1.yaml").read_text(encoding="utf-8"))
    assert dart["experiment"]["name"] == "lgbm_nocat_dart_v1"
    assert dart["model"]["params"]["boosting_type"] == "dart"
    assert base["model"]["params"]["boosting_type"] == "gbdt"
    left = deepcopy(base)
    right = deepcopy(dart)
    for blob in (left, right):
        blob["experiment"].pop("name")
        blob["experiment"].pop("hypothesis")
        blob["experiment"].pop("change")
        blob["experiment"].pop("model_version")
        blob["model"]["params"].pop("boosting_type")
    assert left == right

