"""K-fold training with mandatory OOF / test prediction dumps."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from s6e8 import __version__ as package_version
from s6e8.data import PROJECT_ROOT, load_sample_submission, resolve_path
from s6e8.features import feature_columns, transform
from s6e8.runtime import (
    apply_model_device,
    experiment_summary,
    get_accelerator,
    get_git_commit,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _oof_dir(config: dict[str, Any]) -> Path:
    exp = config["experiment"]["name"]
    out = resolve_path(config["paths"]["oof_dir"]) / exp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _submission_path(config: dict[str, Any]) -> Path:
    exp = config["experiment"]["name"]
    out_dir = resolve_path(config["paths"]["submission_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{exp}.csv"


def _float_dtype(config: dict[str, Any]) -> np.dtype:
    name = config.get("output", {}).get("dtype", "float64")
    return np.dtype(name)


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def oof_exists(config: dict[str, Any]) -> bool:
    oof_dir = resolve_path(config["paths"]["oof_dir"]) / config["experiment"]["name"]
    return (oof_dir / "oof.npy").exists() or (oof_dir / "oof.parquet").exists()


def assert_oof_available(config: dict[str, Any], overwrite: bool = False) -> None:
    if overwrite or not oof_exists(config):
        return
    exp = config["experiment"]["name"]
    raise FileExistsError(
        f"OOF already exists for experiment {exp!r}. "
        "Create a new config/experiment name, or pass --overwrite."
    )


def train_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    backend = str(config["model"]["name"]).lower()
    if backend in {"lightgbm", "lgbm", "lgb"}:
        return _train_lightgbm_cv(train_df, test_df, y, config)
    raise ValueError(
        f"Unsupported model backend {backend!r}. "
        "Add a trainer in s6e8/models/train.py; keep hyperparameters in YAML."
    )


def _train_lightgbm_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    train_feat = transform(train_df, config)
    test_feat = transform(test_df, config)
    cols = feature_columns(train_feat, config)
    cat_cols = [c for c in config["features"]["categorical"] if c in cols]

    X = train_feat[cols]
    X_test = test_feat[cols]
    y_np = y.to_numpy()

    cv_cfg = config["cv"]
    splitter = StratifiedKFold(
        n_splits=int(cv_cfg["n_splits"]),
        shuffle=bool(cv_cfg.get("shuffle", True)),
        random_state=seed,
    )

    dtype = _float_dtype(config)
    oof = np.zeros(len(X), dtype=dtype)
    test_pred = np.zeros(len(X_test), dtype=dtype)
    fold_scores: list[float] = []
    best_iterations: list[int] = []

    model_cfg = config["model"]
    accelerator = get_accelerator(config)
    params = apply_model_device(dict(model_cfg["params"]), model_cfg["name"], accelerator)
    params["seed"] = seed
    params["feature_fraction_seed"] = seed
    params["bagging_seed"] = seed
    print(f"model={model_cfg['name']} accelerator={accelerator}")

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y_np), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        dtrain = lgb.Dataset(
            X_tr,
            label=y_tr,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_va,
            label=y_va,
            categorical_feature=cat_cols,
            reference=dtrain,
            free_raw_data=False,
        )

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=int(model_cfg["num_boost_round"]),
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(int(model_cfg["early_stopping_rounds"])),
                lgb.log_evaluation(int(model_cfg.get("log_evaluation", 100))),
            ],
        )

        va_pred = booster.predict(X_va, num_iteration=booster.best_iteration)
        te_pred = booster.predict(X_test, num_iteration=booster.best_iteration)
        oof[va_idx] = va_pred.astype(dtype, copy=False)
        test_pred += te_pred.astype(dtype, copy=False) / splitter.n_splits

        auc = float(roc_auc_score(y_va, va_pred))
        fold_scores.append(auc)
        best_iterations.append(int(booster.best_iteration or 0))
        print(f"[fold {fold}] AUC={auc:.6f} best_iter={booster.best_iteration}")

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    oof_auc = float(roc_auc_score(y_np, oof))
    print(f"[cv] mean={cv_mean:.6f} std={cv_std:.6f} oof={oof_auc:.6f}")

    artifacts = {
        "oof": oof,
        "test_pred": test_pred,
        "feature_names": cols,
        "fold_scores": fold_scores,
        "best_iterations": best_iterations,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "oof_auc": oof_auc,
        "train_ids": train_df[config["competition"]["id_col"]].to_numpy(),
        "test_ids": test_df[config["competition"]["id_col"]].to_numpy(),
        "y": y_np,
    }
    return artifacts


def save_artifacts(
    artifacts: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, str]:
    output_cfg = config.get("output", {})
    oof_dir = _oof_dir(config)
    id_col = config["competition"]["id_col"]
    target = config["competition"]["target"]
    written: dict[str, str] = {}

    if output_cfg.get("save_oof", True):
        oof_npy = oof_dir / "oof.npy"
        np.save(oof_npy, artifacts["oof"])
        oof_pq = oof_dir / "oof.parquet"
        pd.DataFrame(
            {
                id_col: artifacts["train_ids"],
                target: artifacts["y"],
                "pred": artifacts["oof"],
            }
        ).to_parquet(oof_pq, index=False)
        written["oof_npy"] = _relpath(oof_npy)
        written["oof_parquet"] = _relpath(oof_pq)

    if output_cfg.get("save_test", True):
        test_npy = oof_dir / "test.npy"
        np.save(test_npy, artifacts["test_pred"])
        test_pq = oof_dir / "test.parquet"
        pd.DataFrame(
            {
                id_col: artifacts["test_ids"],
                "pred": artifacts["test_pred"],
            }
        ).to_parquet(test_pq, index=False)
        written["test_npy"] = _relpath(test_npy)
        written["test_parquet"] = _relpath(test_pq)

    git_commit = artifacts.get("git_commit", get_git_commit())
    runtime_seconds = artifacts.get("runtime_seconds")
    summary = experiment_summary(
        config,
        cv_auc=artifacts["oof_auc"],
        runtime_seconds=runtime_seconds,
        git_commit=git_commit,
    )
    metrics = {
        **summary,
        "package_version": package_version,
        "lightgbm_version": lgb.__version__,
        "metric": config["competition"]["metric"],
        "cv_mean": artifacts["cv_mean"],
        "cv_std": artifacts["cv_std"],
        "oof_auc": artifacts["oof_auc"],
        "fold_scores": artifacts["fold_scores"],
        "best_iterations": artifacts["best_iterations"],
        "n_features": len(artifacts["feature_names"]),
        "feature_names": artifacts["feature_names"],
        "n_train": int(len(artifacts["oof"])),
        "n_test": int(len(artifacts["test_pred"])),
        "n_splits": int(config["cv"]["n_splits"]),
        "model_name": config["model"]["name"],
        "config_path": config.get("_config_path"),
    }
    metrics_path = oof_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    written["metrics"] = _relpath(metrics_path)

    experiment_path = oof_dir / "experiment.json"
    experiment_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written["experiment"] = _relpath(experiment_path)

    records_dir = resolve_path(config["paths"].get("experiments_dir", "experiments"))
    records_dir.mkdir(parents=True, exist_ok=True)
    record_path = records_dir / f"{config['experiment']['name']}.json"
    record_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written["experiment_record"] = _relpath(record_path)

    if output_cfg.get("save_submission", True):
        sub_path = _submission_path(config)
        sample = load_sample_submission(config)
        pred_col = target
        if sample is not None:
            pred_col = [c for c in sample.columns if c != id_col][0]
            sub = sample.copy()
            sub[pred_col] = artifacts["test_pred"]
        else:
            sub = pd.DataFrame(
                {id_col: artifacts["test_ids"], pred_col: artifacts["test_pred"]}
            )
        sub.to_csv(sub_path, index=False)
        written["submission"] = _relpath(sub_path)

    return written
