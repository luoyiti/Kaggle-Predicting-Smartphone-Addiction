import numpy as np
import pandas as pd

from s6e8.structural_features import (
    add_decimal_lattice_features,
    add_exact_categorical_features,
    add_screen_budget_features,
)


def test_exact_categories_are_canonical_and_missing_explicit():
    frame = pd.DataFrame(
        {
            "age": [21.0, 21.0, np.nan],
            "daily_screen_time_hours": [3.1, 3.1000000000000001, np.nan],
        }
    )
    config = {
        "features": {
            "numeric": ["age", "daily_screen_time_hours"],
            "exact_categorical": {
                "enabled": True,
                "columns": ["age", "daily_screen_time_hours"],
                "suffix": "__exact",
                "missing_token": "__MISSING__",
                "decimal_places": {"age": 0, "daily_screen_time_hours": 2},
            },
        }
    }
    out = add_exact_categorical_features(frame, config)
    assert out["age__exact"].tolist() == [
        "age=21",
        "age=21",
        "age=__MISSING__",
    ]
    assert out["daily_screen_time_hours__exact"].tolist() == [
        "daily_screen_time_hours=3.10",
        "daily_screen_time_hours=3.10",
        "daily_screen_time_hours=__MISSING__",
    ]


def test_budget_features_preserve_complete_and_observed_semantics():
    frame = pd.DataFrame(
        {
            "daily_screen_time_hours": [8.0, 8.0, 0.0],
            "social_media_hours": [2.0, np.nan, 0.0],
            "gaming_hours": [1.0, 1.0, 0.0],
            "work_study_hours": [2.0, 2.0, 0.0],
            "weekend_screen_time": [10.0, 10.0, 1.0],
            "sleep_hours": [7.0, 7.0, 8.0],
        }
    )
    config = {"features": {"screen_budget": {"enabled": True, "tolerance": 1e-9}}}
    out = add_screen_budget_features(frame, config)
    assert out.loc[0, "screen_component_sum_complete"] == 5.0
    assert out.loc[0, "screen_remainder_complete"] == 3.0
    assert out.loc[1, "screen_component_count"] == 2
    assert pd.isna(out.loc[1, "screen_component_sum_complete"])
    assert out.loc[1, "screen_component_sum_observed"] == 3.0
    assert out.loc[1, "screen_remainder_observed"] == 5.0
    assert pd.isna(out.loc[2, "screen_component_share_complete"])
    assert out.loc[0, "awake_non_screen_hours"] == 9.0


def test_decimal_lattice_is_an_isolated_first_digit_block():
    frame = pd.DataFrame({"daily_screen_time_hours": [3.27, np.nan]})
    config = {
        "features": {
            "decimal_lattice": {
                "enabled": True,
                "columns": ["daily_screen_time_hours"],
            }
        }
    }
    out = add_decimal_lattice_features(frame, config)
    assert np.isclose(out.loc[0, "daily_screen_time_hours__fraction"], 0.27)
    assert out.loc[0, "daily_screen_time_hours__first_decimal"] == 2
    assert pd.isna(out.loc[1, "daily_screen_time_hours__first_decimal"])
