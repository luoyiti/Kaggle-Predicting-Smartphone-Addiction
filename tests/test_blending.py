from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s6e8.blending import lofo_grid_blend
from scripts_loader import load_script


def test_lofo_weights_never_use_the_scored_fold():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    folds = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    pred_a = np.array([0.1, 0.9, 0.2, 0.8, 0.8, 0.2, 0.7, 0.3])
    pred_b = 1.0 - pred_a
    test = np.array([[0.2, 0.8], [0.7, 0.3]])

    result = lofo_grid_blend(
        y=y,
        fold_ids=folds,
        oof_matrix=np.vstack([pred_a, pred_b]),
        test_matrix=test,
        step=50,
    )

    assert result.oof.shape == y.shape
    assert result.test.shape == (2,)
    assert len(result.fold_weights) == 4
    assert all(np.isclose(sum(weights), 1.0) for weights in result.fold_weights)
    assert np.allclose(
        result.test,
        np.mean([np.dot(weights, test) for weights in result.fold_weights], axis=0),
    )


def test_lofo_selected_weight_is_invariant_to_held_out_predictions():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    folds = np.repeat(np.arange(4), 2)
    pred_a = np.tile([0.1, 0.9], 4)
    pred_b = 1.0 - pred_a
    baseline = lofo_grid_blend(
        y=y,
        fold_ids=folds,
        oof_matrix=np.vstack([pred_a, pred_b]),
        test_matrix=np.array([[0.2, 0.8], [0.8, 0.2]]),
        step=50,
    )

    changed_a = pred_a.copy()
    changed_b = pred_b.copy()
    changed_a[:2] = [0.99, 0.01]
    changed_b[:2] = [0.01, 0.99]
    changed = lofo_grid_blend(
        y=y,
        fold_ids=folds,
        oof_matrix=np.vstack([changed_a, changed_b]),
        test_matrix=np.array([[0.2, 0.8], [0.8, 0.2]]),
        step=50,
    )

    assert baseline.fold_weights[0] == changed.fold_weights[0]


@pytest.mark.parametrize(
    ("oof_matrix", "test_matrix", "fold_ids", "match"),
    [
        (np.array([[0.1, np.nan], [0.9, 0.1]]), np.ones((2, 2)), np.array([0, 1]), "finite"),
        (np.ones((2, 3)), np.ones((2, 2)), np.array([0, 1]), "length"),
        (np.ones((2, 4)), np.ones((3, 2)), np.array([0, 0, 1, 1]), "test matrix"),
        (np.ones((2, 4)), np.ones((2, 2)), np.array([0, 0, 1, 1]), "both classes"),
    ],
)
def test_lofo_rejects_invalid_inputs(oof_matrix, test_matrix, fold_ids, match):
    with pytest.raises(ValueError, match=match):
        lofo_grid_blend(
            y=np.array([0, 1, 0, 0]),
            fold_ids=fold_ids,
            oof_matrix=oof_matrix,
            test_matrix=test_matrix,
            step=50,
        )


def test_lofo_rejects_empty_weight_grid():
    with pytest.raises(ValueError, match="step"):
        lofo_grid_blend(
            y=np.array([0, 1, 0, 1]),
            fold_ids=np.array([0, 0, 1, 1]),
            oof_matrix=np.array([[0.1, 0.9, 0.2, 0.8]]),
            test_matrix=np.array([[0.2, 0.8]]),
            step=0,
        )


def test_partial_stored_fold_ids_are_rejected_instead_of_reconstructed():
    blend_script = load_script("blend_oof.py")
    frame = pd.DataFrame(
        {
            "id": [3, 1, 2, 4],
            "addicted_label": [0, 1, 0, 1],
            "pred": [0.1, 0.9, 0.2, 0.8],
            "fold": [0, np.nan, 1, 2],
        }
    )
    with pytest.raises(ValueError, match="missing fold"):
        blend_script._reference_fold_ids(frame)


@pytest.mark.parametrize("artifact", ["OOF", "test"])
def test_alignment_rejects_extra_ids(artifact):
    blend_script = load_script("blend_oof.py")
    frame = pd.DataFrame({"id": [2, 1, 999], "pred": [0.2, 0.1, 0.9]})
    with pytest.raises(ValueError, match="ids do not align"):
        blend_script._align_to_reference(
            np.array([1, 2]),
            frame,
            experiment="extra_ids",
            artifact=artifact,
        )


def test_alignment_preserves_reference_order_after_valid_reordering():
    blend_script = load_script("blend_oof.py")
    frame = pd.DataFrame({"id": [2, 1], "pred": [0.2, 0.1]})
    aligned = blend_script._align_to_reference(
        np.array([1, 2]),
        frame,
        experiment="reordered",
        artifact="OOF",
    )
    assert aligned["id"].tolist() == [1, 2]
    assert aligned["pred"].tolist() == [0.1, 0.2]


def test_cli_lofo_grid_records_honest_fold_metadata(tmp_path):
    pytest.importorskip("pyarrow")
    root = Path(__file__).resolve().parents[1]
    oof_root = tmp_path / "oof"
    ids = np.arange(8)
    y = np.tile([0, 1], 4)
    folds = np.repeat(np.arange(4), 2)
    test_ids = np.array([100, 101])
    for name, pred in {
        "model_a": np.tile([0.1, 0.9], 4),
        "model_b": np.tile([0.9, 0.1], 4),
    }.items():
        folder = oof_root / name
        folder.mkdir(parents=True)
        pd.DataFrame({"id": ids, "addicted_label": y, "pred": pred, "fold": folds}).to_parquet(
            folder / "oof.parquet", index=False
        )
        pd.DataFrame({"id": test_ids, "pred": [0.2, 0.8]}).to_parquet(
            folder / "test.parquet", index=False
        )
        (folder / "metrics.json").write_text(json.dumps({"oof_auc": 1.0}), encoding="utf-8")

    name = "test_lofo_grid_metadata"
    record = root / "experiments" / f"{name}.json"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "blend_oof.py"),
                "--experiments",
                "model_a",
                "model_b",
                "--oof-dir",
                str(oof_root),
                "--name",
                name,
                "--submission-dir",
                str(tmp_path / "submissions"),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        )
        assert "lofo_grid" in completed.stdout
        metrics = json.loads((oof_root / name / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["method"] == "lofo_grid"
        assert len(metrics["fold_weights"]) == 4
        assert metrics["in_sample_weight_search"] is False
    finally:
        record.unlink(missing_ok=True)
        shutil.rmtree(oof_root / name, ignore_errors=True)
