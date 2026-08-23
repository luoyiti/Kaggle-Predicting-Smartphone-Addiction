"""CLI loaders for postprocess scripts (help + tiny OOF blend)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s6e8.oof_io import load_experiment_oof, write_prediction_bundle
from tests.scripts_loader import load_script


def _bundle(tmp_path: Path, name: str, y, pred, test_pred):
    ids = np.arange(len(y))
    test_ids = np.arange(1000, 1000 + len(test_pred))
    write_prediction_bundle(
        name=name,
        train_ids=ids,
        y=y,
        oof_pred=pred,
        test_ids=test_ids,
        test_pred=test_pred,
        metrics={"experiment": name, "oof_auc": 0.9, "n_train": len(y), "n_test": len(test_pred)},
        oof_root=tmp_path / "oof",
        submission_dir=tmp_path / "sub",
        write_experiment_record=False,
    )


def test_oof_io_roundtrip(tmp_path):
    y = np.array([0, 1, 0, 1, 1, 0])
    pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    test = np.array([0.4, 0.6])
    _bundle(tmp_path, "exp_a", y, pred, test)
    loaded = load_experiment_oof("exp_a", tmp_path / "oof")
    assert np.array_equal(loaded["y"], y)
    assert loaded["metrics"]["experiment"] == "exp_a"


def test_missing_oof_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment_oof("does_not_exist", tmp_path / "oof")


@pytest.mark.parametrize(
    "script",
    [
        "blend_oof.py",
        "stack_oof.py",
        "calibrate_oof.py",
        "promote.py",
        "audit_data.py",
        "eval_slices.py",
        "error_analysis.py",
    ],
)
def test_postprocess_scripts_help(script):
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, f"scripts/{script}", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout
    assert load_script(script) is not None
