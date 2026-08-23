"""Guards for harvested OOF dumps and blend inputs.

``oof/catboost_exactcat_budget_v1/`` is overloaded: it currently holds the
PR #11 GPU orig-cats dump (accelerator=gpu, n_cat=12). This-branch CPU
drop-cats harvest must not replace that folder unless the incoming metrics
are accelerator=cpu and n_categorical_features=9.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

FULL_N_TRAIN = 691369
FULL_N_TEST = 296302
CPU_DROPCATS_EXPERIMENT = "catboost_exactcat_budget_v1"
CPU_DROPCATS_N_CAT = 9
PR11_ORIGCATS_ALIAS = "catboost_exactcat_budget_pr11_origcats"
REQUIRED_OOF_FILES = ("oof.parquet", "test.parquet", "metrics.json")


class OofProtocolError(ValueError):
    """Incoming OOF dump does not match the required training protocol."""


def n_categorical_features(metrics: Mapping[str, Any]) -> int | None:
    if metrics.get("n_categorical_features") is not None:
        return int(metrics["n_categorical_features"])
    names = metrics.get("categorical_feature_names")
    if names is not None:
        return int(len(names))
    return None


def as_1d_float(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} pred shape {arr.shape} is not 1-d")
    if arr.size == 0:
        raise ValueError(f"{name} pred is empty")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} pred dtype {arr.dtype} is not numeric")
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name} pred contains {n_bad} NaN/Inf values")
    return np.asarray(arr, dtype=np.float64)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns {missing}; have {list(frame.columns)}")


def validate_prediction_frames(
    experiment: str,
    oof: pd.DataFrame,
    test: pd.DataFrame,
    metrics: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fail loud on id/pred shape problems. Returns 1-d OOF and test preds."""
    _require_columns(oof, ("id", "pred"), label=f"{experiment} OOF")
    _require_columns(test, ("id", "pred"), label=f"{experiment} test")
    if oof["id"].duplicated().any():
        raise ValueError(f"{experiment} OOF ids are not unique")
    if test["id"].duplicated().any():
        raise ValueError(f"{experiment} test ids are not unique")
    oof_pred = as_1d_float(f"{experiment} OOF", oof["pred"].to_numpy())
    test_pred = as_1d_float(f"{experiment} test", test["pred"].to_numpy())
    if oof_pred.shape != (len(oof),):
        raise ValueError(
            f"{experiment} OOF pred length {oof_pred.shape} != n_rows {(len(oof),)}"
        )
    if test_pred.shape != (len(test),):
        raise ValueError(
            f"{experiment} test pred length {test_pred.shape} != n_rows {(len(test),)}"
        )
    if metrics is not None:
        n_train = metrics.get("n_train")
        n_test = metrics.get("n_test")
        if n_train is not None and int(n_train) != len(oof):
            raise ValueError(
                f"{experiment} metrics n_train={n_train} != OOF rows {len(oof)}"
            )
        if n_test is not None and int(n_test) != len(test):
            raise ValueError(
                f"{experiment} metrics n_test={n_test} != test rows {len(test)}"
            )
    return oof_pred, test_pred


def check_npy_shape(folder: Path, which: str, pred: np.ndarray) -> None:
    path = folder / f"{which}.npy"
    if not path.is_file():
        return
    loaded = np.asarray(np.load(path, allow_pickle=False))
    if loaded.ndim != 1 or loaded.shape != pred.shape:
        raise ValueError(
            f"{folder.name} {which}.npy shape {loaded.shape} != parquet pred {pred.shape}"
        )


def validate_cpu_dropcats_budget_metrics(
    metrics: Mapping[str, Any],
    *,
    experiment: str = CPU_DROPCATS_EXPERIMENT,
) -> None:
    """Incoming dump that may replace ``oof/catboost_exactcat_budget_v1/``."""
    accel = str(metrics.get("accelerator") or "").strip().lower()
    n_cat = n_categorical_features(metrics)
    problems: list[str] = []
    if accel != "cpu":
        problems.append(f"accelerator={metrics.get('accelerator')!r} (need cpu)")
    if n_cat != CPU_DROPCATS_N_CAT:
        problems.append(
            f"n_categorical_features={n_cat} (need {CPU_DROPCATS_N_CAT} exact copies; orig cats dropped)"
        )
    if metrics.get("diagnostic") is True:
        problems.append("diagnostic=true")
    n_splits = metrics.get("n_splits")
    if n_splits is not None and int(n_splits) != 5:
        problems.append(f"n_splits={n_splits} (need 5)")
    n_train = metrics.get("n_train")
    if n_train is not None and int(n_train) != FULL_N_TRAIN:
        problems.append(f"n_train={n_train} (need {FULL_N_TRAIN})")
    n_test = metrics.get("n_test")
    if n_test is not None and int(n_test) != FULL_N_TEST:
        problems.append(f"n_test={n_test} (need {FULL_N_TEST})")
    if problems:
        raise OofProtocolError(
            f"Refuse to install/overwrite oof/{experiment}/: "
            + "; ".join(problems)
            + ". That folder may still be the PR #11 GPU orig-cats dump "
            "(accelerator=gpu, n_cat=12). Use scripts/install_kernel_oof.py "
            f"and preserve it as {PR11_ORIGCATS_ALIAS}."
        )


def looks_like_pr11_origcats(metrics: Mapping[str, Any]) -> bool:
    accel = str(metrics.get("accelerator") or "").strip().lower()
    n_cat = n_categorical_features(metrics)
    return accel == "gpu" and n_cat == 12 and metrics.get("diagnostic") is not True


def is_cpu_dropcats_budget(metrics: Mapping[str, Any]) -> bool:
    try:
        validate_cpu_dropcats_budget_metrics(metrics)
    except OofProtocolError:
        return False
    return True


def validate_blend_protocol(components: list[tuple[str, Mapping[str, Any]]]) -> None:
    """Refuse diagnostic/full mixes and disagreeing n_train / n_splits / n_test."""
    diag_true = [name for name, m in components if m.get("diagnostic") is True]
    diag_false = [name for name, m in components if m.get("diagnostic") is False]
    if diag_true and diag_false:
        raise ValueError(
            "Refuse to mix diagnostic OOF with full 5-fold OOF: "
            f"diagnostic={diag_true} full={diag_false}"
        )

    def _group(key: str) -> dict[str, Any]:
        return {name: m[key] for name, m in components if m.get(key) is not None}

    n_trains = _group("n_train")
    if len(set(int(v) for v in n_trains.values())) > 1:
        raise ValueError(f"n_train mismatch across blend components: {n_trains}")
    n_tests = _group("n_test")
    if len(set(int(v) for v in n_tests.values())) > 1:
        raise ValueError(f"n_test mismatch across blend components: {n_tests}")
    n_splits = _group("n_splits")
    if len(set(int(v) for v in n_splits.values())) > 1:
        raise ValueError(f"n_splits mismatch across blend components: {n_splits}")


def warn_if_budget_v1_not_cpu_dropcats(experiment: str, metrics: Mapping[str, Any]) -> str | None:
    if experiment != CPU_DROPCATS_EXPERIMENT:
        return None
    try:
        validate_cpu_dropcats_budget_metrics(metrics)
    except OofProtocolError as exc:
        return str(exc)
    return None
