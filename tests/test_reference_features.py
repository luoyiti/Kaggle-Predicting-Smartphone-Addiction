from __future__ import annotations

import pandas as pd

import numpy as np
import pytest

from s6e8.reference_features import (
    apply_external_reference_features,
    canonical_row_hash,
    prepare_reference_rows,
    resolve_external_reference_path,
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
    """Catch either duplicate reference rows or train/test matches being retained."""
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

    retained, provenance = prepare_reference_rows(
        reference, train, test, PREDICTORS
    )

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
    """An int source row must match the same float-valued competition row."""
    reference = pd.DataFrame([_full_predictor_row(seed) for seed in range(3)])
    train = _float_numeric_predictors(reference.iloc[[0]][FULL_PREDICTORS])
    test = _float_numeric_predictors(reference.iloc[[1]][FULL_PREDICTORS])

    assert canonical_row_hash(reference.iloc[[0]], FULL_PREDICTORS).iloc[0] == (
        canonical_row_hash(train, FULL_PREDICTORS).iloc[0]
    )
    assert canonical_row_hash(reference.iloc[[1]], FULL_PREDICTORS).iloc[0] == (
        canonical_row_hash(test, FULL_PREDICTORS).iloc[0]
    )

    retained, provenance = prepare_reference_rows(
        reference, train, test, FULL_PREDICTORS
    )

    assert retained[FULL_PREDICTORS].to_dict("records") == [
        reference.iloc[2][FULL_PREDICTORS].to_dict()
    ]
    assert provenance == {
        "raw_rows": 3,
        "unique_rows": 3,
        "duplicate_rows_removed": 0,
        "query_overlap_rows_removed": 2,
        "retained_rows": 1,
    }


def test_canonical_hash_normalizes_numeric_missing_and_signed_zero_only():
    nullable_integer = pd.DataFrame(
        {"value": pd.Series([0, None], dtype="Int64")}
    )
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
    return pd.DataFrame(
        {
            "transaction_id": np.arange(100, 106),
            "user_id": np.arange(200, 206),
            "age": [18, 19, 20, 21, 22, 23],
            "gender": ["Male", "Female", "Other", "Male", "Female", "Male"],
            "daily_screen_time_hours": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "social_media_hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "gaming_hours": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            "work_study_hours": [8.0, 7.5, 7.0, 6.5, 6.0, 5.5],
            "sleep_hours": [8.0, 7.8, 7.6, 7.4, 7.2, 7.0],
            "notifications_per_day": [10, 20, 30, 40, 50, 60],
            "app_opens_per_day": [5, 10, 15, 20, 25, 30],
            "weekend_screen_time": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
            "stress_level": ["Low", "Low", "Medium", "Medium", "High", "High"],
            "academic_work_impact": ["No", "No", "No", "Yes", "Yes", "Yes"],
            "addiction_level": ["Low", "Low", "Medium", "High", "High", "High"],
            "addicted_label": [0, 0, 0, 1, 1, 1],
        }
    )


def _query_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "age": [30, 31],
            "gender": ["Male", "Female"],
            "daily_screen_time_hours": [5.0, np.nan],
            "social_media_hours": [1.25, 2.75],
            "gaming_hours": [0.5, 1.1],
            "work_study_hours": [7.25, 5.75],
            "sleep_hours": [7.7, 7.1],
            "notifications_per_day": [25, 55],
            "app_opens_per_day": [12, 28],
            "weekend_screen_time": [6.0, 12.0],
            "stress_level": ["Low", "High"],
            "academic_work_impact": ["No", "Yes"],
        }
    )
    test = pd.DataFrame(
        {
            "age": [32],
            "gender": ["Other"],
            "daily_screen_time_hours": [14.0],
            "social_media_hours": [3.5],
            "gaming_hours": [1.4],
            "work_study_hours": [5.0],
            "sleep_hours": [6.8],
            "notifications_per_day": [70],
            "app_opens_per_day": [35],
            "weekend_screen_time": [15.0],
            "stress_level": ["High"],
            "academic_work_impact": ["Yes"],
        }
    )
    return train, test


def _reference_config(path, *, mode: str = "distribution") -> dict:
    block = {
        "enabled": True,
        "path": str(path),
        "dataset_source": "fixture/reference",
        "source_url": "https://example.test/reference",
        "mode": mode,
        "predictor_columns": FULL_PREDICTORS,
        "cdf_columns": ["daily_screen_time_hours"],
        "distance_columns": ["daily_screen_time_hours"],
        "frequency_columns": ["gender"],
        "remove_query_overlaps": True,
    }
    if mode == "label_aware":
        block.update(
            {
                "target_column": "addicted_label",
                "class_cdf_columns": ["daily_screen_time_hours"],
                "target_mean_columns": ["daily_screen_time_hours"],
                "n_bins": 3,
                "smoothing": 2.0,
            }
        )
    return {"external_reference": block}


def test_reference_path_prefers_existing_configured_legacy_mount(tmp_path):
    legacy_path = tmp_path / "legacy" / "reference.csv"
    legacy_path.parent.mkdir()
    legacy_path.write_text("age\n20\n", encoding="utf-8")
    typed_path = tmp_path / "input" / "datasets" / "fixture" / "reference" / "reference.csv"
    typed_path.parent.mkdir(parents=True)
    typed_path.write_text("age\n30\n", encoding="utf-8")

    resolved = resolve_external_reference_path(
        _reference_config(legacy_path)["external_reference"],
        kaggle_input_root=tmp_path / "input",
    )

    assert resolved == legacy_path


def test_reference_path_falls_back_to_typed_kaggle_dataset_mount(tmp_path):
    configured_path = tmp_path / "missing" / "reference.csv"
    typed_path = tmp_path / "input" / "datasets" / "fixture" / "reference" / "reference.csv"
    typed_path.parent.mkdir(parents=True)
    typed_path.write_text("age\n30\n", encoding="utf-8")

    resolved = resolve_external_reference_path(
        _reference_config(configured_path)["external_reference"],
        kaggle_input_root=tmp_path / "input",
    )

    assert resolved == typed_path


def test_reference_path_fails_clearly_when_configured_and_typed_paths_absent(tmp_path):
    configured_path = tmp_path / "missing" / "reference.csv"
    with pytest.raises(FileNotFoundError, match="reference.csv") as exc_info:
        resolve_external_reference_path(
            _reference_config(configured_path)["external_reference"],
            kaggle_input_root=tmp_path / "input",
        )

    message = str(exc_info.value)
    assert str(configured_path) in message
    assert "datasets/fixture/reference/reference.csv" in message


def test_reference_features_record_actual_typed_fallback_path(tmp_path):
    configured_path = tmp_path / "missing" / "reference.csv"
    input_root = tmp_path / "input"
    typed_path = input_root / "datasets" / "fixture" / "reference" / "reference.csv"
    typed_path.parent.mkdir(parents=True)
    _reference_frame().to_csv(typed_path, index=False)

    _train_out, _test_out, provenance = apply_external_reference_features(
        *_query_frames(),
        _reference_config(configured_path),
        kaggle_input_root=input_root,
    )

    assert provenance["path"] == str(typed_path)
    assert provenance["configured_path"] == str(configured_path)


def test_distribution_features_are_target_free_and_schema_aligned(tmp_path):
    """Catch source-label access or divergent train/test feature construction."""
    source_path = tmp_path / "reference.csv"
    reference = _reference_frame()
    reference.to_csv(source_path, index=False)
    train, test = _query_frames()
    config = _reference_config(source_path)

    train_out, test_out, provenance = apply_external_reference_features(
        train, test, config
    )

    assert "ref_daily_screen_time_hours__cdf" in train_out
    assert "ref_daily_screen_time_hours__robust_z" in train_out
    assert "ref_daily_screen_time_hours__robust_abs_z" in test_out
    assert train_out.loc[0, "ref_daily_screen_time_hours__cdf"] == pytest.approx(2 / 6)
    assert train_out.loc[0, "ref_gender__frequency"] == pytest.approx(3 / 6)
    assert list(train_out.columns) == list(test_out.columns)
    assert provenance["external_supervision"] is False
    assert provenance["mode"] == "distribution"
    assert provenance["raw_rows"] == provenance["unique_rows"] == 6
    assert provenance["retained_rows"] == 6
    assert provenance["source_url"] == "https://example.test/reference"
    assert len(provenance["sha256"]) == 64

    reference["addicted_label"] = reference["addicted_label"].iloc[::-1].to_numpy()
    reference.to_csv(source_path, index=False)
    permuted_train, permuted_test, _ = apply_external_reference_features(
        train, test, config
    )
    pd.testing.assert_frame_equal(train_out, permuted_train)
    pd.testing.assert_frame_equal(test_out, permuted_test)


def test_label_aware_features_use_only_retained_source_labels(tmp_path):
    """Catch absent aggregate supervision or a direct query-row label lookup."""
    source_path = tmp_path / "reference.csv"
    reference = _reference_frame()
    overlapping = _query_frames()[0].iloc[[0]].copy()
    for column in ("transaction_id", "user_id", "addiction_level"):
        overlapping[column] = "source-only"
    overlapping["addicted_label"] = 1
    reference = pd.concat([reference, overlapping[reference.columns]], ignore_index=True)
    reference.to_csv(source_path, index=False)
    train, test = _query_frames()

    train_out, test_out, provenance = apply_external_reference_features(
        train, test, _reference_config(source_path, mode="label_aware")
    )

    assert "ref_daily_screen_time_hours__class_cdf_gap" in train_out
    assert "ref_daily_screen_time_hours__source_target_mean" in train_out
    assert list(train_out.columns) == list(test_out.columns)
    assert provenance["external_supervision"] is True
    assert provenance["query_overlap_rows_removed"] == 1
    assert provenance["retained_rows"] == 6
    assert provenance["selected_columns"]["target_mean"] == [
        "daily_screen_time_hours"
    ]


@pytest.mark.parametrize(
    ("labels", "message"),
    (
        ([0, 1, 2, 0, 1, 0], "binary"),
        ([1, 1, 1, 1, 1, 1], "both classes"),
    ),
)
def test_label_aware_rejects_invalid_retained_labels(tmp_path, labels, message):
    """Catch fitting supervised transforms to invalid or single-class targets."""
    source_path = tmp_path / "reference.csv"
    reference = _reference_frame()
    reference["addicted_label"] = labels
    reference.to_csv(source_path, index=False)

    with pytest.raises(ValueError, match=message):
        apply_external_reference_features(
            *_query_frames(), _reference_config(source_path, mode="label_aware")
        )
