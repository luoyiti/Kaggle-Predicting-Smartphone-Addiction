"""Competition prediction and data contracts (Kaggle-native, not online serving).

These constants are the schema/leakage policy for playground-series-s6e8.
Model hyperparameters still live in YAML; this module does not train.
"""

from __future__ import annotations

from typing import Any

COMPETITION_SLUG = "playground-series-s6e8"
TASK = "binary_classification"
TARGET = "addicted_label"
ID_COL = "id"
METRIC = "roc_auc"
POSITIVE_CLASS = 1
SUBMIT_COLUMN = "addicted_label"  # P(class=1), not a hard label
ENTITY_GRAIN = "one row = one synthetic smartphone user (id)"

NUMERIC_COLUMNS = (
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
)

CATEGORICAL_COLUMNS = (
    "gender",
    "stress_level",
    "academic_work_impact",
)

CATEGORICAL_VALUES: dict[str, frozenset[str]] = {
    "gender": frozenset({"Male", "Female", "Other"}),
    "stress_level": frozenset({"Low", "Medium", "High"}),
    "academic_work_impact": frozenset({"Yes", "No"}),
}

# Inclusive soft bounds for audit warnings (not clip thresholds).
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (10.0, 90.0),
    "daily_screen_time_hours": (0.0, 24.0),
    "social_media_hours": (0.0, 24.0),
    "gaming_hours": (0.0, 24.0),
    "work_study_hours": (0.0, 24.0),
    "sleep_hours": (0.0, 24.0),
    "notifications_per_day": (0.0, 5000.0),
    "app_opens_per_day": (0.0, 5000.0),
    "weekend_screen_time": (0.0, 48.0),
}

COMPONENT_SUM_PARTS = (
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
)
COMPONENT_SUM_TOTAL = "daily_screen_time_hours"

SPLIT_POLICY = {
    "cv": "StratifiedKFold on addicted_label",
    "shuffle": True,
    "default_n_splits": 5,
    "default_seed": 42,
    "holdout": "Kaggle public/private test; local eval uses OOF only",
    "no_random_row_leak": "Do not shuffle train into test; id is sequential and unused as a feature",
}

FORBIDDEN_FEATURES = (
    "id",  # sequential split index; univariate leak check only
    TARGET,
)

ORIGINAL_SOURCE_POLICY = (
    "Do not concatenate the original ~7,500-row source into playground train. "
    "Component dependence differs (original daily ⟂ social+gaming+work; "
    "playground never violates daily >= sum). Mixing is a leakage / shift risk."
)

SLICE_COLUMNS = (
    "gender",
    "stress_level",
    "academic_work_impact",
)

MISSINGNESS_COHORT_COLUMNS = (
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
)

PREDICTION_CONTRACT: dict[str, Any] = {
    "decision": "Kaggle submission ranks users by P(addicted_label=1) for ROC-AUC",
    "output_schema": {"id": "int", SUBMIT_COLUMN: "float in [0, 1]"},
    "grain": ENTITY_GRAIN,
    "serving_mode": "batch (Kaggle Kernel → submission.csv)",
    "fallback": "keep the last promoted OOF/submission (lgbm_nocat or blend_nocat); do not retrain to roll back",
    "latency": "offline; Kernel wall time is the budget, not p95 serving latency",
    "calibration": "optional post-hoc isotonic/platt on OOF; must not reduce OOF AUC materially",
}

DEFAULT_PROMOTION_GATES: dict[str, Any] = {
    "baseline_experiment": "lgbm_nocat",
    "min_oof_auc_delta": 0.0,
    "max_oof_auc_drop": 0.0005,
    "max_ece": 0.08,
    "min_slice_n": 200,
    "min_slice_auc": 0.50,
    "require_slices": False,
    "require_oof_and_test": True,
    "fail_closed_on_missing": True,
}


def required_train_columns() -> tuple[str, ...]:
    return (ID_COL, *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, TARGET)


def required_test_columns() -> tuple[str, ...]:
    return (ID_COL, *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS)
