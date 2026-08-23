"""Fold-safe frequency encoding. Synthetic frames only."""

import numpy as np
import pandas as pd

from s6e8.frequency_encoding import (
    ExactValueFrequencyEncoder,
    apply_fold_frequency_encoding,
    parse_frequency_config,
)


def test_parse_disabled_or_empty_returns_none():
    assert parse_frequency_config({"features": {}}) is None
    assert (
        parse_frequency_config(
            {"features": {"frequency_encoding": {"enabled": False, "columns": ["age"]}}}
        )
        is None
    )
    assert (
        parse_frequency_config(
            {"features": {"frequency_encoding": {"enabled": True, "columns": []}}}
        )
        is None
    )


def test_train_counts_do_not_use_validation_rows():
    X_tr = pd.DataFrame({"x": [1.0, 1.0, 2.0]})
    X_va = pd.DataFrame({"x": [1.0, 3.0]})
    X_te = pd.DataFrame({"x": [1.0]})
    cfg = {
        "columns": ["x"],
        "round_decimals": 2,
        "suffix": "_freq",
        "log1p": False,
        "unseen_value": 0.0,
    }
    tr, va, te, stats = apply_fold_frequency_encoding(X_tr, X_va, X_te, cfg)
    # value 1 appears twice in train
    assert tr["x_freq"].iloc[0] == 2.0
    assert va["x_freq"].iloc[0] == 2.0
    # unseen 3 -> 0
    assert va["x_freq"].iloc[1] == 0.0
    assert stats["columns"]["x"]["n_val_unseen"] == 1


def test_missing_stays_missing_and_log1p():
    X = pd.DataFrame({"x": [1.0, 1.0, np.nan]})
    enc = ExactValueFrequencyEncoder(columns=["x"], round_decimals=2, log1p=True)
    enc.fit(X)
    out = enc.transform(X)
    assert pd.isna(out["x_freq"].iloc[2])
    assert abs(out["x_freq"].iloc[0] - np.log1p(2.0)) < 1e-12
