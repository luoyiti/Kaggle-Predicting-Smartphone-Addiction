#!/usr/bin/env python3
"""Blend saved OOF predictions with honest leave-one-fold-out evaluation.

The default ``lofo_grid`` method chooses weights using four folds and scores
the fifth.  The historical ``grid`` method is retained only as a clearly
marked in-sample diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from s6e8.blending import _grid_weights, lofo_grid_blend
from s6e8.data import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend experiment OOF / test predictions")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--oof-dir", default="oof")
    parser.add_argument(
        "--method",
        choices=["mean", "rank", "logit", "grid", "lofo_grid"],
        default="lofo_grid",
    )
    parser.add_argument("--grid-step", type=int, default=5)
    parser.add_argument("--name", default=None, help="Output experiment name")
    parser.add_argument("--submission-dir", default="submissions")
    return parser.parse_args()


def _prediction_file(folder: Path, kind: str) -> Path:
    candidates = [folder / f"{kind}.parquet", folder / f"{kind}_predictions.parquet"]
    for path in candidates:
        if path.is_file():
            return path
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing {kind} prediction artifact: {expected}")


def _load(exp: str, oof_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    folder = oof_root / exp
    oof = pd.read_parquet(_prediction_file(folder, "oof"))
    test = pd.read_parquet(_prediction_file(folder, "test"))
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    return oof, test, metrics


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy() / (len(x) + 1.0)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _legacy_folds(y: np.ndarray) -> np.ndarray:
    """Recreate the repository's canonical five-fold assignment exactly."""
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        fold_ids[valid_idx] = fold
    if np.any(fold_ids < 0):
        raise RuntimeError("Legacy fold reconstruction did not assign every row")
    return fold_ids


def _reference_fold_ids(oof: pd.DataFrame) -> np.ndarray:
    """Use persisted folds, reconstructing only genuinely legacy artifacts."""
    if "fold" not in oof.columns:
        return _legacy_folds(oof["addicted_label"].to_numpy())
    if oof["fold"].isna().any():
        raise ValueError("OOF artifact has missing fold ids")
    return oof["fold"].to_numpy()


def _require_prediction_columns(frame: pd.DataFrame, *, target: bool, artifact: str) -> None:
    required = {"id", "pred"}
    if target:
        required.add("addicted_label")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{artifact} is missing required columns: {sorted(missing)}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{artifact} contains duplicate ids")


def _align_to_reference(
    reference_ids: np.ndarray,
    frame: pd.DataFrame,
    *,
    experiment: str,
    artifact: str,
) -> pd.DataFrame:
    """Align by id without mutating the first experiment's original row order."""
    reference_index = pd.Index(reference_ids)
    artifact_index = pd.Index(frame["id"])
    if (
        len(artifact_index) != len(reference_index)
        or not artifact_index.isin(reference_index).all()
        or not reference_index.isin(artifact_index).all()
    ):
        raise ValueError(f"{experiment} {artifact} ids do not align with the reference")
    indexed = frame.set_index("id", drop=False)
    try:
        aligned = indexed.loc[reference_ids].reset_index(drop=True)
    except KeyError as exc:
        raise ValueError(f"{experiment} {artifact} ids do not align with the reference") from exc
    if len(aligned) != len(reference_ids):
        raise ValueError(f"{experiment} {artifact} ids do not align with the reference")
    return aligned


def _fold_scores(
    y: np.ndarray, predictions: np.ndarray, folds: np.ndarray
) -> tuple[tuple[Any, ...], tuple[float, ...]]:
    values = tuple(np.unique(folds).tolist())
    scores: list[float] = []
    for fold in values:
        mask = folds == fold
        if len(np.unique(y[mask])) < 2:
            raise ValueError(f"fold {fold!r} must contain both classes")
        scores.append(float(roc_auc_score(y[mask], predictions[mask])))
    return values, tuple(scores)


def _full_oof_grid(y: np.ndarray, stacked: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Historical, in-sample grid search retained for backward compatibility."""
    best_weights: tuple[float, ...] | None = None
    best_auc = -np.inf
    best_prediction: np.ndarray | None = None
    for candidate in _grid_weights(stacked.shape[0], step):
        prediction = np.asarray(candidate) @ stacked
        auc = float(roc_auc_score(y, prediction))
        if auc > best_auc or (auc == best_auc and (best_weights is None or candidate < best_weights)):
            best_auc = auc
            best_weights = candidate
            best_prediction = prediction
    if best_weights is None or best_prediction is None:
        raise ValueError("weight grid has no valid weights")
    return np.asarray(best_weights), best_prediction


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _correlation_payload(
    stacked: np.ndarray, experiments: list[str], method: str
) -> dict[str, dict[str, float | None]]:
    frame = pd.DataFrame(stacked.T, columns=experiments).corr(method=method)
    return {
        str(row): {str(col): _json_number(frame.loc[row, col]) for col in frame.columns}
        for row in frame.index
    }


def _weight_map(experiments: list[str], weights: np.ndarray | tuple[float, ...]) -> dict[str, float]:
    return {name: float(weight) for name, weight in zip(experiments, weights)}


def main() -> None:
    args = parse_args()
    oof_root = Path(args.oof_dir)
    if not oof_root.is_absolute():
        oof_root = PROJECT_ROOT / oof_root

    oofs: list[np.ndarray] = []
    tests: list[np.ndarray] = []
    component_rows: list[tuple[str, float]] = []
    y: np.ndarray | None = None
    ids: np.ndarray | None = None
    test_ids: np.ndarray | None = None
    fold_ids: np.ndarray | None = None

    for index, exp in enumerate(args.experiments):
        oof, test, component_metrics = _load(exp, oof_root)
        _require_prediction_columns(oof, target=True, artifact=f"{exp} OOF")
        _require_prediction_columns(test, target=False, artifact=f"{exp} test")
        if index == 0:
            # Preserve this order. Legacy folds are reconstructed against this
            # unmodified frame, then every following component is aligned to it.
            y = oof["addicted_label"].to_numpy()
            ids = oof["id"].to_numpy()
            test_ids = test["id"].to_numpy()
            fold_ids = _reference_fold_ids(oof)
        else:
            assert ids is not None and test_ids is not None and y is not None and fold_ids is not None
            oof = _align_to_reference(ids, oof, experiment=exp, artifact="OOF")
            test = _align_to_reference(test_ids, test, experiment=exp, artifact="test")
            if not np.array_equal(y, oof["addicted_label"].to_numpy()):
                raise ValueError(f"{exp} OOF target does not align with the reference")
            if "fold" in oof.columns:
                other_folds = _reference_fold_ids(oof)
                if not np.array_equal(fold_ids, other_folds):
                    raise ValueError(f"{exp} OOF fold ids do not align with the reference")

        oofs.append(oof["pred"].to_numpy(dtype=float))
        tests.append(test["pred"].to_numpy(dtype=float))
        component_rows.append((exp, float(component_metrics.get("oof_auc", np.nan))))
        print(f"{exp}: oof_auc={component_metrics.get('oof_auc')} n_train={len(oof)}")

    assert y is not None and ids is not None and test_ids is not None and fold_ids is not None
    stacked = np.vstack(oofs)
    test_matrix = np.vstack(tests)
    pearson = _correlation_payload(stacked, args.experiments, "pearson")
    spearman = _correlation_payload(stacked, args.experiments, "spearman")
    print("OOF Pearson correlation:")
    print(pd.DataFrame(pearson).T.round(4))

    if args.method == "lofo_grid":
        result = lofo_grid_blend(
            y=y,
            fold_ids=fold_ids,
            oof_matrix=stacked,
            test_matrix=test_matrix,
            step=args.grid_step,
        )
        blend_oof = result.oof
        blend_test = result.test
        weights = np.asarray(result.weight_mean)
        fold_values = result.fold_values
        fold_auc = result.fold_auc
        fold_weight_payload = [
            {"fold": fold, "weights": _weight_map(args.experiments, fold_weights)}
            for fold, fold_weights in zip(fold_values, result.fold_weights)
        ]
        auc = result.auc
        in_sample_weight_search = False
        lofo_selection = True
    else:
        if args.method == "mean":
            weights = np.ones(len(args.experiments), dtype=float) / len(args.experiments)
            blend_oof = stacked.mean(axis=0)
            blend_test = test_matrix.mean(axis=0)
        elif args.method == "rank":
            weights = np.ones(len(args.experiments), dtype=float) / len(args.experiments)
            blend_oof = np.mean([_rank(x) for x in oofs], axis=0)
            blend_test = np.mean([_rank(x) for x in tests], axis=0)
        elif args.method == "logit":
            weights = np.ones(len(args.experiments), dtype=float) / len(args.experiments)
            blend_oof = _sigmoid(np.mean([_logit(x) for x in oofs], axis=0))
            blend_test = _sigmoid(np.mean([_logit(x) for x in tests], axis=0))
        else:
            weights, blend_oof = _full_oof_grid(y, stacked, args.grid_step)
            blend_test = weights @ test_matrix
        fold_values, fold_auc = _fold_scores(y, blend_oof, fold_ids)
        fold_weight_payload = [
            {"fold": fold, "weights": _weight_map(args.experiments, weights)}
            for fold in fold_values
        ]
        auc = float(roc_auc_score(y, blend_oof))
        in_sample_weight_search = args.method == "grid"
        lofo_selection = False

    component_fold_scores = np.asarray(
        [_fold_scores(y, prediction, fold_ids)[1] for prediction in oofs], dtype=float
    )
    component_deltas = [
        {
            "fold": fold,
            "deltas": {
                exp: float(fold_auc[fold_index] - component_fold_scores[component_index, fold_index])
                for component_index, exp in enumerate(args.experiments)
            },
        }
        for fold_index, fold in enumerate(fold_values)
    ]

    weight_map = _weight_map(args.experiments, weights)
    print(f"blend method={args.method} weights={weight_map} oof_auc={auc:.6f}")
    singles = ", ".join(f"{name}={score:.6f}" for name, score in component_rows)
    print(f"components: {singles}")

    name = args.name or ("blend_" + "_".join(args.experiments))
    out_dir = oof_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", blend_oof)
    np.save(out_dir / "test.npy", blend_test)
    pd.DataFrame({"id": ids, "addicted_label": y, "pred": blend_oof, "fold": fold_ids}).to_parquet(
        out_dir / "oof.parquet", index=False
    )
    pd.DataFrame({"id": test_ids, "pred": blend_test}).to_parquet(out_dir / "test.parquet", index=False)
    metrics = {
        "experiment": name,
        "method": args.method,
        "components": args.experiments,
        "weights": weight_map,
        "weight_mean": weight_map,
        "fold_weights": fold_weight_payload,
        "oof_auc": float(auc),
        "fold_auc": {str(fold): float(score) for fold, score in zip(fold_values, fold_auc)},
        "component_auc": {component: score for component, score in component_rows},
        "oof_pearson_corr": pearson,
        "oof_spearman_corr": spearman,
        "per_fold_component_deltas": component_deltas,
        "in_sample_weight_search": in_sample_weight_search,
        "lofo_selection": lofo_selection,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    records_dir = PROJECT_ROOT / "experiments"
    records_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": name,
        "cv_auc": float(auc),
        "method": args.method,
        "components": args.experiments,
        "weights": weight_map,
        "fold_weights": fold_weight_payload,
        "fold_auc": metrics["fold_auc"],
        "component_auc": metrics["component_auc"],
        "in_sample_weight_search": in_sample_weight_search,
        "lofo_selection": lofo_selection,
        "n_train": int(len(blend_oof)),
        "n_test": int(len(blend_test)),
        "diagnostic": False,
        "change": f"{args.method} blend of {', '.join(args.experiments)}",
    }
    (records_dir / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    sub_dir = Path(args.submission_dir)
    if not sub_dir.is_absolute():
        sub_dir = PROJECT_ROOT / sub_dir
    sub_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_ids, "addicted_label": blend_test}).to_csv(
        sub_dir / f"{name}.csv", index=False
    )
    print(f"wrote oof/{name}/, experiments/{name}.json, and submissions/{name}.csv")


if __name__ == "__main__":
    main()
