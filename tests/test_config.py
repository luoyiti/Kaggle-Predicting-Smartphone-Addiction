from __future__ import annotations

import copy

import pytest

from s6e8.data import load_config
from s6e8.runtime import experiment_summary, validate_config


def test_baseline_yaml_loads(baseline_config_path):
    config = load_config(baseline_config_path)
    validate_config(config)
    assert config["experiment"]["name"] == "baseline"
    assert config["experiment"]["seed"] == 42
    assert config["runtime"]["accelerator"] == "cpu"
    assert config["model"]["name"] == "lightgbm"


def test_formal_protocol_rejects_fold_or_seed_drift(baseline_config_path):
    config = load_config(baseline_config_path)
    config["experiment"]["formal"] = True
    config["experiment"]["validation_protocol"] = "fixed5_seed42_v1"
    validate_config(config)

    bad_folds = copy.deepcopy(config)
    bad_folds["cv"]["n_splits"] = 4
    with pytest.raises(ValueError, match="fixed5_seed42_v1"):
        validate_config(bad_folds)

    bad_seed = copy.deepcopy(config)
    bad_seed["experiment"]["seed"] = 7
    with pytest.raises(ValueError, match="fixed5_seed42_v1"):
        validate_config(bad_seed)


@pytest.mark.parametrize("output_key", ["save_oof", "save_test"])
def test_formal_protocol_requires_prediction_outputs(baseline_config_path, output_key):
    config = load_config(baseline_config_path)
    config["experiment"]["formal"] = True
    config["experiment"]["validation_protocol"] = "fixed5_seed42_v1"
    config["output"][output_key] = False

    with pytest.raises(ValueError, match="Formal output requirements"):
        validate_config(config)


def test_experiment_summary_records_source_formal_only_when_true(baseline_config_path):
    config = load_config(baseline_config_path)
    assert "source_formal" not in experiment_summary(config)

    config["experiment"]["source_formal"] = True
    assert experiment_summary(config)["source_formal"] is True


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


CATBOOST_STRUCTURAL_CONFIGS = (
    "catboost_numeric_v1",
    "catboost_exactcat_v1",
    "catboost_exactcat_budget_v1",
    "catboost_exactcat_budget_lattice_v1",
)


def _without_experiment_metadata(config):
    normalized = copy.deepcopy(config)
    normalized.pop("_config_path", None)
    for key in ("name", "hypothesis", "change"):
        normalized["experiment"].pop(key, None)
    return normalized


def test_catboost_structural_configs_are_formal_and_keep_base_categories():
    from pathlib import Path

    configs = [
        load_config(Path("configs") / f"{name}.yaml")
        for name in CATBOOST_STRUCTURAL_CONFIGS
    ]
    expected_categories = ["gender", "stress_level", "academic_work_impact"]
    expected_precision = {
        "age": 0,
        "daily_screen_time_hours": 2,
        "social_media_hours": 2,
        "gaming_hours": 2,
        "work_study_hours": 2,
        "sleep_hours": 2,
        "notifications_per_day": 0,
        "app_opens_per_day": 0,
        "weekend_screen_time": 2,
    }

    for config in configs:
        validate_config(config)
        assert config["experiment"]["formal"] is True
        assert config["experiment"]["seed"] == 42
        assert config["experiment"]["validation_protocol"] == "fixed5_seed42_v1"
        assert config["cv"]["n_splits"] == 5
        assert config["model"]["name"] == "catboost"
        assert config["runtime"]["accelerator"] == "gpu"
        assert config["output"]["save_oof"] is True
        assert config["output"]["save_test"] is True
        assert config["features"]["categorical"] == expected_categories
        assert "drop" not in config["features"]

    for config in configs[1:]:
        assert config["features"]["exact_categorical"]["decimal_places"] == expected_precision
    assert [config["experiment"]["name"] for config in configs] == list(
        CATBOOST_STRUCTURAL_CONFIGS
    )


@pytest.mark.parametrize(
    ("control_name", "candidate_name", "feature_block"),
    (
        ("catboost_numeric_v1", "catboost_exactcat_v1", "exact_categorical"),
        ("catboost_exactcat_v1", "catboost_exactcat_budget_v1", "screen_budget"),
        (
            "catboost_exactcat_budget_v1",
            "catboost_exactcat_budget_lattice_v1",
            "decimal_lattice",
        ),
    ),
)
def test_adjacent_catboost_configs_are_isolated_feature_experiments(
    control_name, candidate_name, feature_block
):
    from pathlib import Path

    control = _without_experiment_metadata(
        load_config(Path("configs") / f"{control_name}.yaml")
    )
    candidate = _without_experiment_metadata(
        load_config(Path("configs") / f"{candidate_name}.yaml")
    )

    candidate["features"][feature_block] = control["features"][feature_block]
    assert candidate == control


REFERENCE_CONFIGS = (
    "catboost_exactcat_budget_refdist_v1",
    "catboost_exactcat_budget_reflabel_v1",
)


def test_reference_configs_are_isolated_children_and_keep_base_categories():
    """Catch unrelated model/feature drift in either external-reference child."""
    from pathlib import Path

    base = _without_experiment_metadata(
        load_config(Path("configs/catboost_exactcat_budget_v1.yaml"))
    )
    for name in REFERENCE_CONFIGS:
        candidate = load_config(Path("configs") / f"{name}.yaml")
        validate_config(candidate)
        assert candidate["experiment"]["name"] == name
        assert candidate["features"]["categorical"] == [
            "gender",
            "stress_level",
            "academic_work_impact",
        ]
        reference = candidate["external_reference"]
        assert reference["enabled"] is True
        assert reference["remove_query_overlaps"] is True
        normalized = _without_experiment_metadata(candidate)
        normalized.pop("external_reference")
        assert normalized == base


def test_reference_configs_select_only_shared_predictors_and_expected_supervision():
    """Catch a source-only identifier leaking into features or mislabeled mode."""
    from pathlib import Path

    expected_predictors = [
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
    ]
    expected_distance = [
        "daily_screen_time_hours",
        "weekend_screen_time",
        "social_media_hours",
        "notifications_per_day",
        "app_opens_per_day",
    ]
    distribution = load_config(Path("configs") / f"{REFERENCE_CONFIGS[0]}.yaml")[
        "external_reference"
    ]
    label_aware = load_config(Path("configs") / f"{REFERENCE_CONFIGS[1]}.yaml")[
        "external_reference"
    ]

    assert distribution["mode"] == "distribution"
    assert distribution["predictor_columns"] == expected_predictors
    assert distribution["distance_columns"] == expected_distance
    assert distribution["frequency_columns"] == []
    assert label_aware["mode"] == "label_aware"
    assert label_aware["target_column"] == "addicted_label"
    assert label_aware["class_cdf_columns"] == expected_distance
    assert label_aware["target_mean_columns"] == [
        "daily_screen_time_hours",
        "weekend_screen_time",
        "notifications_per_day",
        "app_opens_per_day",
    ]
    assert label_aware["n_bins"] == 50
    assert label_aware["smoothing"] == 20.0


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
