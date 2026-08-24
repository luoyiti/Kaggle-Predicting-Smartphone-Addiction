"""Slice, ECE, and calibration helpers. Synthetic arrays only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from s6e8.metrics import (
    calibration_summary,
    expected_calibration_error,
    hard_band_mask,
    missingness_cohorts,
    safe_roc_auc,
    slice_metrics,
)


def test_safe_roc_auc_none_when_one_class():
    y = np.ones(30)
    p = np.linspace(0.1, 0.9, 30)
    assert safe_roc_auc(y, p, min_n=10) is None


def test_ece_zero_when_perfectly_calibrated():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    p = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=2) < 1e-9


def test_slice_metrics_by_gender_and_hard_band():
    frame = pd.DataFrame(
        {
            "gender": ["Male", "Male", "Female", "Female"] * 20,
            "stress_level": ["Low"] * 80,
            "academic_work_impact": ["Yes"] * 80,
            "daily_screen_time_hours": np.linspace(1, 10, 80),
            "weekend_screen_time": np.linspace(1, 10, 80),
            "social_media_hours": np.linspace(0, 5, 80),
        }
    )
    y = np.array([0, 1] * 40)
    pred = y.astype(float) * 0.9 + 0.05
    out = slice_metrics(y, pred, frame, min_n=10)
    assert out["overall_auc"] > 0.9
    assert "Male" in out["gender"]
    assert out["hard_band"]["n"] >= 0
    assert "complete" in out["missingness"] or "daily_screen_missing" in out["missingness"]


def test_missingness_cohorts_flags_daily():
    df = pd.DataFrame(
        {
            "daily_screen_time_hours": [1.0, np.nan, 3.0],
            "weekend_screen_time": [1.0, 2.0, np.nan],
            "social_media_hours": [1.0, 1.0, 1.0],
        }
    )
    labels = missingness_cohorts(df).tolist()
    assert labels[0] == "complete"
    assert labels[1] == "daily_screen_missing"
    assert labels[2] == "other_usage_missing"


def test_hard_band_mask():
    p = np.array([0.1, 0.5, 0.9])
    assert hard_band_mask(p).tolist() == [False, True, False]


def test_calibration_summary_keys():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    p = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.4, 0.9, 0.1])
    summary = calibration_summary(y, p)
    assert set(summary) >= {"brier", "ece", "mean_pred", "positive_rate", "auc"}
    assert 0.0 <= summary["ece"] <= 1.0
