"""Hard-band specialist mask / weights. Synthetic frames only."""

import numpy as np
import pandas as pd
import pytest

from s6e8.error_band import band_mask
from s6e8.hard_band import (
    align_frozen_pred,
    apply_hard_band_fold,
    gated_blend,
    parse_hard_band_config,
    row_weights,
)


def test_parse_disabled_returns_none():
    assert parse_hard_band_config({"features": {}}) is None
    assert parse_hard_band_config({"features": {"hard_band": {"enabled": False}}}) is None


def test_parse_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode"):
        parse_hard_band_config(
            {
                "competition": {"id_col": "id"},
                "features": {"hard_band": {"enabled": True, "mode": "leak"}},
            }
        )


def test_align_frozen_pred_left_joins_and_marks_missing():
    frozen = pd.DataFrame({"id": [1, 2, 3], "pred": [0.1, 0.5, 0.9]})
    pred, meta = align_frozen_pred(
        np.array([2, 9, 1]), frozen, id_col="id", pred_col="pred"
    )
    np.testing.assert_allclose(pred[0], 0.5)
    assert np.isnan(pred[1])
    np.testing.assert_allclose(pred[2], 0.1)
    assert meta["n_missing"] == 1
    assert meta["n_matched"] == 2


def test_train_only_keeps_band_rows_and_full_length_val_predictions_are_separate():
    X_tr = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y_tr = np.array([0, 1, 0, 1])
    X_va = pd.DataFrame({"x": [10.0, 11.0, 12.0]})
    y_va = np.array([0, 0, 1])
    tr_pred = np.array([0.05, 0.5, 0.55, 0.99])
    va_pred = np.array([0.4, 0.1, 0.6])
    hb = {
        "lo": 0.3,
        "hi": 0.7,
        "mode": "train_only",
        "eval_on": "band",
        "min_train_rows": 2,
        "min_eval_rows": 2,
        "in_band_weight": 4.0,
        "out_band_weight": 1.0,
    }
    out = apply_hard_band_fold(X_tr, y_tr, X_va, y_va, tr_pred, va_pred, hb)
    assert len(out["y_tr"]) == 2
    np.testing.assert_array_equal(out["X_tr"]["x"].to_numpy(), [1.0, 2.0])
    assert out["train_weight"] is None
    assert len(out["y_va_eval"]) == 2
    np.testing.assert_array_equal(out["X_va_eval"]["x"].to_numpy(), [10.0, 12.0])
    assert out["stats"]["used_train_only"] is True
    assert out["stats"]["used_eval_band"] is True
    # Original val frame is untouched so callers can still score every row.
    assert len(y_va) == 3


def test_reweight_keeps_all_rows():
    X_tr = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y_tr = np.array([0, 1, 0])
    X_va = pd.DataFrame({"x": [0.0]})
    y_va = np.array([1])
    tr_pred = np.array([0.1, 0.5, 0.2])
    hb = {
        "lo": 0.3,
        "hi": 0.7,
        "mode": "reweight",
        "eval_on": "all",
        "min_train_rows": 1,
        "min_eval_rows": 1,
        "in_band_weight": 4.0,
        "out_band_weight": 1.0,
    }
    out = apply_hard_band_fold(X_tr, y_tr, X_va, y_va, tr_pred, np.array([0.2]), hb)
    assert len(out["y_tr"]) == 3
    np.testing.assert_allclose(out["train_weight"], [1.0, 4.0, 1.0])
    assert out["stats"]["used_train_only"] is False


def test_gated_blend_replaces_only_the_base_hard_band():
    base = np.array([0.1, 0.5, 0.9])
    spec = np.array([0.2, 0.8, 0.3])
    out = gated_blend(base, spec, lo=0.3, hi=0.7, mix=1.0)
    np.testing.assert_allclose(out, [0.1, 0.8, 0.9])
    mixed = gated_blend(base, spec, lo=0.3, hi=0.7, mix=0.5)
    np.testing.assert_allclose(mixed, [0.1, 0.65, 0.9])


def test_row_weights_use_open_interval():
    pred = np.array([0.3, 0.30001, 0.7])
    w = row_weights(pred, lo=0.3, hi=0.7, in_band_weight=5.0, out_band_weight=1.0)
    np.testing.assert_allclose(w, [1.0, 5.0, 1.0])
    np.testing.assert_array_equal(band_mask(pred, 0.3, 0.7), [False, True, False])
