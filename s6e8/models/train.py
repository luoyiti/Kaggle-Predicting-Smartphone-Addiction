"""K-fold training with mandatory OOF / test prediction dumps."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from s6e8 import __version__ as package_version
from s6e8.data import PROJECT_ROOT, load_sample_submission, resolve_path
from s6e8.features import feature_columns, transform
from s6e8.runtime import (
    apply_model_device,
    experiment_summary,
    get_accelerator,
    get_git_commit,
)

BACKEND_ALIASES = {
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "lgb": "lightgbm",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "catboost": "catboost",
    "cb": "catboost",
    "cat": "catboost",
    "histgb": "histgb",
    "histgradientboosting": "histgb",
    "sklearn_histgb": "histgb",
    "logreg": "logreg",
    "logistic": "logreg",
    "logisticregression": "logreg",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_backend(config: dict[str, Any]) -> str:
    name = str(config["model"]["name"]).lower()
    if name not in BACKEND_ALIASES:
        supported = ", ".join(sorted(set(BACKEND_ALIASES.values())))
        raise ValueError(
            f"Unsupported model backend {name!r}. Supported: {supported}. "
            "Add a trainer in s6e8/models/train.py; keep hyperparameters in YAML."
        )
    return BACKEND_ALIASES[name]


def _oof_dir(config: dict[str, Any]) -> Path:
    exp = config["experiment"]["name"]
    out = resolve_path(config["paths"]["oof_dir"]) / exp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _submission_path(config: dict[str, Any]):
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


def stratified_subsample(
    df: pd.DataFrame,
    target: str,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    if n_rows >= len(df):
        return df
    sampled, _ = train_test_split(
        df,
        train_size=n_rows,
        stratify=df[target],
        random_state=seed,
        shuffle=True,
    )
    return sampled.reset_index(drop=True)


def apply_diagnostic_overrides(
    config: dict[str, Any],
    *,
    max_train_rows: int | None = None,
    n_splits: int | None = None,
) -> dict[str, Any]:
    """CLI/runtime overrides for cheap ranking runs. Recorded on the config."""
    runtime = dict(config.get("runtime") or {})
    experiment = dict(config["experiment"])
    diagnostic = False

    if max_train_rows is not None:
        runtime["max_train_rows"] = int(max_train_rows)
        diagnostic = True
    if n_splits is not None:
        cv = dict(config["cv"])
        cv["n_splits"] = int(n_splits)
        config["cv"] = cv
        diagnostic = True

    if diagnostic:
        experiment["diagnostic"] = True
        name = str(experiment["name"])
        rows = runtime.get("max_train_rows")
        suffix = f"_diag{rows}" if rows else "_diag"
        if suffix not in name:
            experiment["name"] = f"{name}{suffix}"
        print(
            "WARNING: diagnostic run "
            f"max_train_rows={runtime.get('max_train_rows')} "
            f"n_splits={config['cv']['n_splits']} "
            f"experiment={experiment['name']}. "
            "Do not treat this AUC as a competition score."
        )

    config["runtime"] = runtime
    config["experiment"] = experiment
    return config


FoldTrainer = Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]],
    tuple[np.ndarray, np.ndarray, int],
]


def _prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[str]]:
    train_feat = transform(train_df, config)
    test_feat = transform(test_df, config)
    cols = feature_columns(train_feat, config)
    cat_cols = [c for c in config["features"]["categorical"] if c in cols]
    return train_feat[cols], test_feat[cols], y.to_numpy(), cols, cat_cols


def _run_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
    fold_fn: FoldTrainer,
) -> dict[str, Any]:
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    X, X_test, y_np, cols, cat_cols = _prepare_xy(train_df, test_df, y, config)
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
    ctx = {
        "config": config,
        "cols": cols,
        "cat_cols": cat_cols,
        "seed": seed,
        "accelerator": get_accelerator(config),
    }
    print(
        f"model={config['model']['name']} backend={resolve_backend(config)} "
        f"accelerator={ctx['accelerator']} n_features={len(cols)}"
    )

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y_np), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]
        va_pred, te_pred, best_iter = fold_fn(X_tr, y_tr, X_va, y_va, X_test, ctx)
        oof[va_idx] = va_pred.astype(dtype, copy=False)
        test_pred += te_pred.astype(dtype, copy=False) / splitter.n_splits
        auc = float(roc_auc_score(y_va, va_pred))
        fold_scores.append(auc)
        best_iterations.append(int(best_iter))
        print(f"[fold {fold}] AUC={auc:.6f} best_iter={best_iter}")

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    oof_auc = float(roc_auc_score(y_np, oof))
    print(f"[cv] mean={cv_mean:.6f} std={cv_std:.6f} oof={oof_auc:.6f}")
    return {
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
        "backend": resolve_backend(config),
    }


def train_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    backend = resolve_backend(config)
    trainers = {
        "lightgbm": _fold_lightgbm,
        "xgboost": _fold_xgboost,
        "catboost": _fold_catboost,
        "histgb": _fold_histgb,
        "logreg": _fold_logreg,
    }
    return _run_cv(train_df, test_df, y, config, trainers[backend])


def _fold_lightgbm(X_tr, y_tr, X_va, y_va, X_test, ctx):
    model_cfg = ctx["config"]["model"]
    params = apply_model_device(
        dict(model_cfg["params"]), model_cfg["name"], ctx["accelerator"]
    )
    params["seed"] = ctx["seed"]
    params["feature_fraction_seed"] = ctx["seed"]
    params["bagging_seed"] = ctx["seed"]
    cat_cols = ctx["cat_cols"]
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
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
    best_iter = int(booster.best_iteration or 0)
    va_pred = booster.predict(X_va, num_iteration=booster.best_iteration)
    te_pred = booster.predict(X_test, num_iteration=booster.best_iteration)
    return va_pred, te_pred, best_iter


def _fold_xgboost(X_tr, y_tr, X_va, y_va, X_test, ctx):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("Install xgboost to use model.name: xgboost") from exc

    model_cfg = ctx["config"]["model"]
    params = apply_model_device(
        dict(model_cfg["params"]), model_cfg["name"], ctx["accelerator"]
    )
    params.setdefault("seed", ctx["seed"])
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(model_cfg["num_boost_round"]),
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=int(model_cfg["early_stopping_rounds"]),
        verbose_eval=int(model_cfg.get("log_evaluation", 100)),
    )
    best_iter = int(getattr(booster, "best_iteration", 0) or 0)
    end = best_iter + 1 if best_iter > 0 else None
    iteration_range = (0, end) if end is not None else None
    kwargs = {"iteration_range": iteration_range} if iteration_range else {}
    va_pred = booster.predict(dvalid, **kwargs)
    te_pred = booster.predict(dtest, **kwargs)
    return va_pred, te_pred, best_iter


def _catboost_frame(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = X.copy()
    for col in cat_cols:
        out[col] = out[col].astype("string").fillna("__NA__")
    return out


def _fold_catboost(X_tr, y_tr, X_va, y_va, X_test, ctx):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("Install catboost to use model.name: catboost") from exc

    model_cfg = ctx["config"]["model"]
    params = apply_model_device(
        dict(model_cfg["params"]), model_cfg["name"], ctx["accelerator"]
    )
    params.setdefault("random_seed", ctx["seed"])
    params.setdefault("loss_function", "Logloss")
    params.setdefault("eval_metric", "AUC")
    params.setdefault("verbose", int(model_cfg.get("log_evaluation", 100)))
    iterations = int(params.pop("iterations", model_cfg.get("num_boost_round", 1000)))
    cat_cols = ctx["cat_cols"]
    X_tr_c = _catboost_frame(X_tr, cat_cols)
    X_va_c = _catboost_frame(X_va, cat_cols)
    X_te_c = _catboost_frame(X_test, cat_cols)
    model = CatBoostClassifier(iterations=iterations, **params)
    model.fit(
        X_tr_c,
        y_tr,
        eval_set=(X_va_c, y_va),
        cat_features=cat_cols,
        early_stopping_rounds=int(model_cfg["early_stopping_rounds"]),
        use_best_model=True,
    )
    best_iter = int(model.get_best_iteration() or 0)
    va_pred = model.predict_proba(X_va_c)[:, 1]
    te_pred = model.predict_proba(X_te_c)[:, 1]
    return va_pred, te_pred, best_iter


def _fold_histgb(X_tr, y_tr, X_va, y_va, X_test, ctx):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model_cfg = ctx["config"]["model"]
    params = dict(model_cfg.get("params") or {})
    params.setdefault("random_state", ctx["seed"])
    if "max_iter" not in params and "num_boost_round" in model_cfg:
        params["max_iter"] = int(model_cfg["num_boost_round"])
    if "n_iter_no_change" not in params and "early_stopping_rounds" in model_cfg:
        params["n_iter_no_change"] = int(model_cfg["early_stopping_rounds"])
        params.setdefault("early_stopping", True)
    params.setdefault("categorical_features", "from_dtype")
    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    best_iter = int(getattr(clf, "n_iter_", 0) or 0)
    va_pred = clf.predict_proba(X_va)[:, 1]
    te_pred = clf.predict_proba(X_test)[:, 1]
    return va_pred, te_pred, best_iter


def _fold_logreg(X_tr, y_tr, X_va, y_va, X_test, ctx):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    model_cfg = ctx["config"]["model"]
    params = dict(model_cfg.get("params") or {})
    params.setdefault("max_iter", 400)
    params.setdefault("solver", "lbfgs")
    cat_cols = ctx["cat_cols"]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ]
    )
    clf = Pipeline([("pre", pre), ("model", LogisticRegression(random_state=ctx["seed"], **params))])
    clf.fit(X_tr, y_tr)
    va_pred = clf.predict_proba(X_va)[:, 1]
    te_pred = clf.predict_proba(X_test)[:, 1]
    return va_pred, te_pred, 0


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
    extra = {
        "cv_std": artifacts.get("cv_std"),
        "hypothesis": config["experiment"].get("hypothesis"),
        "change": config["experiment"].get("change"),
        "diagnostic": bool(config["experiment"].get("diagnostic", False)),
        "max_train_rows": (config.get("runtime") or {}).get("max_train_rows"),
        "backend": artifacts.get("backend") or resolve_backend(config),
    }
    extra = {k: v for k, v in extra.items() if v is not None}
    summary = experiment_summary(
        config,
        cv_auc=artifacts["oof_auc"],
        runtime_seconds=runtime_seconds,
        git_commit=git_commit,
        extra=extra,
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

    is_diagnostic = bool(config["experiment"].get("diagnostic", False))
    if not is_diagnostic:
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
            if len(sub) != len(artifacts["test_pred"]):
                sub = pd.DataFrame(
                    {id_col: artifacts["test_ids"], pred_col: artifacts["test_pred"]}
                )
            else:
                sub[pred_col] = artifacts["test_pred"]
        else:
            sub = pd.DataFrame(
                {id_col: artifacts["test_ids"], pred_col: artifacts["test_pred"]}
            )
        sub.to_csv(sub_path, index=False)
        written["submission"] = _relpath(sub_path)

    return written
