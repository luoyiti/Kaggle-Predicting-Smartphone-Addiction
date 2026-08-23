from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from s6e8.data import load_config
from s6e8.reference_features import (
    apply_reference_features,
    canonical_row_hash,
    dataset_sources_for_kernel,
    parse_reference_block,
    prepare_reference_rows,
    resolve_reference_path,
)


PREDICTORS = ["age", "daily_screen_time_hours", "gender"]
FULL_PREDICTORS = [
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


def test_prepare_reference_removes_duplicates_and_all_query_overlaps():
    reference = pd.DataFrame(
        {
            "age": [20, 20, 30, 40],
            "daily_screen_time_hours": [3.1, 3.1, 5.2, 7.3],
            "gender": ["Male", "Male", "Female", "Other"],
            "addicted_label": [0, 0, 1, 1],
        }
    )
    train = reference.iloc[[0]][PREDICTORS].copy()
    test = reference.iloc[[2]][PREDICTORS].copy()

    retained, provenance = prepare_reference_rows(reference, train, test, PREDICTORS)

    assert retained[PREDICTORS].to_dict("records") == [
        {"age": 40, "daily_screen_time_hours": 7.3, "gender": "Other"}
    ]
    assert provenance["raw_rows"] == 4
    assert provenance["unique_rows"] == 3
    assert provenance["duplicate_rows_removed"] == 1
    assert provenance["query_overlap_rows_removed"] == 2
    assert provenance["retained_rows"] == 1
    assert canonical_row_hash(train, PREDICTORS).iloc[0] not in set(
        canonical_row_hash(retained, PREDICTORS)
    )


def _full_predictor_row(seed: int) -> dict[str, object]:
    return {
        "age": 20 + seed,
        "gender": ["Male", "Female", "Other"][seed % 3],
        "daily_screen_time_hours": 3.25 + seed,
        "social_media_hours": 1.0 + seed / 10,
        "gaming_hours": 0.5 + seed / 10,
        "work_study_hours": 4.0 + seed / 10,
        "sleep_hours": 7.0 - seed / 10,
        "notifications_per_day": 20 + seed,
        "app_opens_per_day": 10 + seed,
        "weekend_screen_time": 5.5 + seed,
        "stress_level": ["Low", "Medium", "High"][seed % 3],
        "academic_work_impact": ["No", "Yes"][seed % 2],
    }


def _float_numeric_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    categorical = {"gender", "stress_level", "academic_work_impact"}
    numeric = [column for column in FULL_PREDICTORS if column not in categorical]
    out[numeric] = out[numeric].astype("float64")
    return out


def test_full_predictor_overlap_hash_is_numeric_dtype_independent():
    reference = pd.DataFrame([_full_predictor_row(seed) for seed in range(3)])
    train = _float_numeric_predictors(reference.iloc[[0]][FULL_PREDICTORS])
    test = _float_numeric_predictors(reference.iloc[[1]][FULL_PREDICTORS])

    assert canonical_row_hash(reference.iloc[[0]], FULL_PREDICTORS).iloc[0] == (
        canonical_row_hash(train, FULL_PREDICTORS).iloc[0]
    )
    retained, provenance = prepare_reference_rows(
        reference, train, test, FULL_PREDICTORS
    )
    assert provenance["query_overlap_rows_removed"] == 2
    assert provenance["retained_rows"] == 1
    assert canonical_row_hash(retained, FULL_PREDICTORS).iloc[0] == (
        canonical_row_hash(reference.iloc[[2]], FULL_PREDICTORS).iloc[0]
    )


def test_canonical_hash_normalizes_numeric_missing_and_signed_zero_only():
    nullable_integer = pd.DataFrame({"value": pd.Series([0, None], dtype="Int64")})
    floating = pd.DataFrame({"value": [-0.0, np.nan]}, dtype="float64")
    categorical = pd.DataFrame({"value": ["0", None]}, dtype="string")

    pd.testing.assert_series_equal(
        canonical_row_hash(nullable_integer, ["value"]),
        canonical_row_hash(floating, ["value"]),
        check_names=False,
    )
    assert canonical_row_hash(categorical, ["value"]).iloc[0] != (
        canonical_row_hash(floating.iloc[[0]], ["value"]).iloc[0]
    )


def _reference_frame() -> pd.DataFrame:
    rows = [_full_predictor_row(seed) for seed in range(8)]
    frame = pd.DataFrame(rows)
    frame["addicted_label"] = [0, 0, 0, 1, 1, 1, 1, 0]
    frame["addiction_level"] = ["Low", "Low", "Medium", "High", "High", "High", "High", "Low"]
    return frame


def _query_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame([_full_predictor_row(seed) for seed in (20, 21)])
    train.loc[1, "daily_screen_time_hours"] = np.nan
    test = pd.DataFrame([_full_predictor_row(22)])
    return train, test


def _reference_config(path: Path) -> dict:
    return {
        "features": {
            "reference": {
                "enabled": True,
                "path": str(path),
                "dataset_source": "jayjoshi37/smartphone-usage-and-addiction-prediction",
                "source_url": "https://example.test/reference",
                "mode": "distribution",
                "use_labels": False,
                "predictor_columns": FULL_PREDICTORS,
                "cdf_columns": ["daily_screen_time_hours"],
                "distance_columns": ["daily_screen_time_hours"],
                "frequency_columns": ["gender", "notifications_per_day"],
                "remove_query_overlaps": True,
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
        }
    }


def test_reference_path_prefers_existing_configured_file(tmp_path):
    legacy_path = tmp_path / "legacy" / "reference.csv"
    legacy_path.parent.mkdir()
    legacy_path.write_text("age\n20\n", encoding="utf-8")
    typed_path = (
        tmp_path / "input" / "datasets" / "jayjoshi37"
        / "smartphone-usage-and-addiction-prediction" / "reference.csv"
    )
    typed_path.parent.mkdir(parents=True)
    typed_path.write_text("age\n30\n", encoding="utf-8")

    resolved = resolve_reference_path(
        {
            "path": str(legacy_path),
            "dataset_source": "jayjoshi37/smartphone-usage-and-addiction-prediction",
        },
        kaggle_input_root=tmp_path / "input",
    )
    assert resolved == legacy_path


def test_reference_path_falls_back_to_typed_kaggle_dataset_mount(tmp_path):
    configured_path = tmp_path / "missing" / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
    typed_path = (
        tmp_path / "input" / "datasets" / "jayjoshi37"
        / "smartphone-usage-and-addiction-prediction"
        / "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
    )
    typed_path.parent.mkdir(parents=True)
    typed_path.write_text("age\n30\n", encoding="utf-8")

    resolved = resolve_reference_path(
        {
            "path": str(configured_path),
            "dataset_source": "jayjoshi37/smartphone-usage-and-addiction-prediction",
        },
        kaggle_input_root=tmp_path / "input",
    )
    assert resolved == typed_path


def test_reference_path_classic_kaggle_mount(tmp_path):
    configured_path = tmp_path / "missing" / "original.csv"
    classic = tmp_path / "input" / "smartphone-usage-and-addiction-prediction" / "original.csv"
    classic.parent.mkdir(parents=True)
    classic.write_text("age\n1\n", encoding="utf-8")
    resolved = resolve_reference_path(
        {
            "path": str(configured_path),
            "dataset_source": "jayjoshi37/smartphone-usage-and-addiction-prediction",
        },
        kaggle_input_root=tmp_path / "input",
    )
    assert resolved == classic


def test_reference_path_fails_clearly_when_absent(tmp_path):
    configured_path = tmp_path / "missing" / "reference.csv"
    with pytest.raises(FileNotFoundError, match="kaggle datasets download"):
        resolve_reference_path(
            {
                "path": str(configured_path),
                "dataset_source": "jayjoshi37/smartphone-usage-and-addiction-prediction",
            },
            kaggle_input_root=tmp_path / "input",
        )


def test_apply_reference_adds_target_free_columns_and_drops_labels(tmp_path):
    path = tmp_path / "original.csv"
    _reference_frame().to_csv(path, index=False)
    train, test = _query_frames()
    config = _reference_config(path)

    train_out, test_out, provenance = apply_reference_features(train, test, config)

    expected = {
        "ref_daily_screen_time_hours__cdf",
        "ref_daily_screen_time_hours__robust_z",
        "ref_daily_screen_time_hours__robust_abs_z",
        "ref_gender__frequency",
        "ref_notifications_per_day__frequency",
        "ref_knn_mean_dist",
        "ref_knn_min_dist",
    }
    assert expected.issubset(set(train_out.columns))
    assert expected.issubset(set(test_out.columns))
    assert "addicted_label" not in train_out.columns
    assert "addiction_level" not in train_out.columns
    assert provenance["external_supervision"] is False
    assert provenance["mode"] == "distribution"
    assert provenance["retained_rows"] == 8
    assert list(train_out.columns) == list(test_out.columns)
    assert train_out["ref_knn_mean_dist"].notna().all()
    assert test_out["ref_knn_min_dist"].notna().all()
    # Missing daily screen still gets a CDF nan, not a fabricated label.
    assert pd.isna(train_out.loc[1, "ref_daily_screen_time_hours__cdf"])


def test_label_aware_mode_is_rejected(tmp_path):
    path = tmp_path / "original.csv"
    _reference_frame().to_csv(path, index=False)
    train, test = _query_frames()
    config = _reference_config(path)
    config["features"]["reference"]["mode"] = "label_aware"
    with pytest.raises(ValueError, match="target-free"):
        apply_reference_features(train, test, config)


def test_disable_overlap_filter_is_rejected(tmp_path):
    path = tmp_path / "original.csv"
    _reference_frame().to_csv(path, index=False)
    train, test = _query_frames()
    config = _reference_config(path)
    config["features"]["reference"]["remove_query_overlaps"] = False
    with pytest.raises(ValueError, match="remove_query_overlaps"):
        apply_reference_features(train, test, config)


def test_dataset_sources_for_kernel_from_yaml():
    config = load_config("configs/catboost_exactcat_budget_refdist_v1.yaml")
    block = parse_reference_block(config)
    assert block is not None
    assert block["enabled"] is True
    assert dataset_sources_for_kernel(config) == [
        "jayjoshi37/smartphone-usage-and-addiction-prediction"
    ]
    assert dataset_sources_for_kernel(load_config("configs/catboost_exactcat_budget_v1.yaml")) == []


def test_prepare_xy_includes_reference_columns(tmp_path):
    from s6e8.data import split_xy
    from s6e8.models.train import _prepare_xy

    raw = yaml.safe_load(Path("configs/lgbm_nocat.yaml").read_text(encoding="utf-8"))
    path = tmp_path / "original.csv"
    _reference_frame().to_csv(path, index=False)
    raw["experiment"]["name"] = "ref_smoke"
    raw["features"]["reference"] = _reference_config(path)["features"]["reference"]
    cfg_path = tmp_path / "ref.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(cfg_path)

    train = pd.DataFrame([_full_predictor_row(i) | {"id": i, "addicted_label": i % 2} for i in range(12)])
    test = pd.DataFrame([_full_predictor_row(100 + i) | {"id": 100 + i} for i in range(4)])
    X, y = split_xy(train, config)
    X_tr, X_te, y_np, cols, cat_cols, meta = _prepare_xy(X, test, y, config)
    assert any(c.startswith("ref_") for c in cols)
    assert "ref_knn_mean_dist" in cols
    assert meta["retained_rows"] >= 1
    assert len(X_tr) == len(train)
    assert len(X_te) == len(test)
    assert len(y_np) == len(train)
