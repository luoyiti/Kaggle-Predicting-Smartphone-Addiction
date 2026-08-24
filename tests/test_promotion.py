"""Fail-closed promotion gates."""

from __future__ import annotations

import pytest

from s6e8.promotion import evaluate_promotion


def test_missing_oof_auc_fails_closed():
    result = evaluate_promotion({"n_train": 10, "n_test": 5}, {"oof_auc": 0.96})
    assert result.passed is False
    assert any("oof_auc" in f or "missing" in f for f in result.failures)


def test_diagnostic_cannot_promote():
    result = evaluate_promotion(
        {"oof_auc": 0.99, "n_train": 100, "n_test": 50, "diagnostic": True},
        {"oof_auc": 0.96, "n_train": 100, "n_test": 50},
    )
    assert result.passed is False
    assert any("diagnostic" in f for f in result.failures)


def test_candidate_below_baseline_fails():
    result = evaluate_promotion(
        {"oof_auc": 0.95, "n_train": 100, "n_test": 50},
        {"oof_auc": 0.96, "n_train": 100, "n_test": 50},
        {"min_oof_auc_delta": 0.0, "require_ece": False},
    )
    assert result.passed is False
    with pytest.raises(ValueError, match="promotion gates failed"):
        result.raise_if_failed()


def test_equal_or_better_auc_passes_without_slices():
    result = evaluate_promotion(
        {"oof_auc": 0.9638, "n_train": 691369, "n_test": 296302},
        {"oof_auc": 0.963771, "n_train": 691369, "n_test": 296302},
        {"min_oof_auc_delta": 0.0, "require_ece": False, "require_slices": False},
    )
    assert result.passed is True
    result.raise_if_failed()


def test_ece_gate_when_present():
    result = evaluate_promotion(
        {"oof_auc": 0.97, "n_train": 100, "n_test": 50, "ece": 0.2},
        {"oof_auc": 0.96},
        {"max_ece": 0.05, "require_ece": True, "min_oof_auc_delta": 0.0},
    )
    assert result.passed is False


def test_slice_floor():
    candidate = {
        "oof_auc": 0.97,
        "n_train": 1000,
        "n_test": 100,
        "slices": {
            "gender": {"Male": {"n": 500, "auc": 0.4, "positive_rate": 0.7}},
        },
    }
    result = evaluate_promotion(
        candidate,
        {"oof_auc": 0.96},
        {"min_slice_auc": 0.5, "min_slice_n": 200, "require_ece": False},
    )
    assert result.passed is False
    assert any("slice" in f for f in result.failures)
