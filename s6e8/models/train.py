"""K-fold training with mandatory OOF / test prediction dumps."""

from __future__ import annotations

import inspect
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from s6e8 import __version__ as package_version
from s6e8.data import PROJECT_ROOT, load_sample_submission, resolve_path
from s6e8.features import categorical_feature_columns, feature_columns, transform
from s6e8.reference_features import apply_external_reference_features
from s6e8.runtime import (
    apply_model_device,
    dependency_versions,
    experiment_summary,
    get_accelerator,
    get_git_commit,
)
from s6e8.target_encoding import apply_fold_target_encoding, parse_exact_te_config

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
    "lookup_transformer": "lookup_transformer",
    "lookup": "lookup_transformer",
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
        if bool(experiment.get("formal", False)):
            experiment["source_formal"] = True
            experiment["formal"] = False
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
) -> tuple[
    pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[str], dict[str, Any]
]:
    train_feat = transform(train_df, config)
    test_feat = transform(test_df, config)
    train_feat, test_feat, data_provenance = apply_external_reference_features(
        train_feat, test_feat, config
    )
    if list(train_feat.columns) != list(test_feat.columns):
        raise ValueError("Train and test feature schemas must match before selection")
    cols = feature_columns(train_feat, config)
    cat_cols = [
        c for c in categorical_feature_columns(train_feat, config) if c in cols
    ]
    return train_feat[cols], test_feat[cols], y.to_numpy(), cols, cat_cols, data_provenance


def _assert_target_free_provenance(
    value: Any,
    *,
    path: tuple[str, ...] = ("preprocessing_provenance",),
) -> None:
    """Reject target-related keys anywhere in predictor preprocessing metadata."""
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            lowered = key_text.casefold()
            if "target" in lowered or "label" in lowered:
                location = ".".join((*path, key_text))
                raise ValueError(
                    "Lookup preprocessing provenance must be target-free; "
                    f"forbidden key at {location}"
                )
            _assert_target_free_provenance(
                nested,
                path=(*path, key_text),
            )
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_target_free_provenance(
                nested,
                path=(*path, str(index)),
            )
    elif isinstance(value, str):
        lowered = value.casefold()
        if "target" in lowered or "label" in lowered:
            location = ".".join(path)
            raise ValueError(
                "Lookup preprocessing provenance must be target-free; "
                f"forbidden string value at {location}"
            )


def _lookup_model_columns(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, int]]:
    """Validate and resolve the explicitly configured Lookup token schema."""
    model_cfg = config["model"]

    def column_list(key: str) -> list[str]:
        raw = model_cfg.get(key)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"model.{key} must be a non-empty list")
        if any(not isinstance(column, str) or not column for column in raw):
            raise ValueError(f"model.{key} must contain non-empty strings")
        columns = list(raw)
        if len(columns) != len(set(columns)):
            raise ValueError(f"model.{key} must not contain duplicate columns")
        return columns

    lookup_columns = column_list("lookup_columns")
    numeric_columns = column_list("numeric_token_columns")
    if numeric_columns[: len(lookup_columns)] != lookup_columns:
        raise ValueError(
            "model.numeric_token_columns must start with model.lookup_columns "
            "in identical order"
        )

    restricted = {
        str(config["competition"]["id_col"]),
        str(config["competition"]["target"]),
    }
    selected = list(dict.fromkeys([*lookup_columns, *numeric_columns]))
    forbidden = sorted(restricted.intersection(selected))
    if forbidden:
        raise ValueError(
            "Lookup token columns cannot contain the identifier or target: "
            f"{forbidden}"
        )

    unknown_train = [column for column in selected if column not in train_feat]
    unknown_test = [column for column in selected if column not in test_feat]
    if unknown_train or unknown_test:
        raise ValueError(
            "Lookup token configuration contains unknown transformed columns: "
            f"train={unknown_train}, test={unknown_test}"
        )

    raw_precision = model_cfg.get("decimal_places") or {}
    if not isinstance(raw_precision, dict):
        raise ValueError("model.decimal_places must be a mapping")
    decimal_places: dict[str, int] = {}
    for column in lookup_columns:
        if column not in raw_precision:
            raise ValueError(
                f"model.decimal_places is missing lookup column {column!r}"
            )
        decimals = raw_precision[column]
        if type(decimals) is not int or decimals < 0:
            raise ValueError(
                "model.decimal_places values must be non-negative integers"
            )
        decimal_places[column] = decimals
    extra_precision = sorted(set(raw_precision) - set(lookup_columns))
    if extra_precision:
        raise ValueError(
            "model.decimal_places contains non-lookup columns: "
            f"{extra_precision}"
        )
    return lookup_columns, numeric_columns, decimal_places


def _run_lookup_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run target-isolated fixed-fold CV for the Lookup-Transformer backend."""
    from s6e8.models import lookup_transformer as lookup_backend

    reference_cfg = config.get("external_reference") or {}
    if bool(reference_cfg.get("enabled", False)):
        raise ValueError(
            "Lookup-Transformer does not accept external reference features"
        )

    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    train_feat = transform(train_df, config)
    test_feat = transform(test_df, config)
    lookup_columns, numeric_columns, decimal_places = _lookup_model_columns(
        train_feat,
        test_feat,
        config,
    )
    predictor_columns = list(dict.fromkeys([*lookup_columns, *numeric_columns]))
    predictor_train = train_feat.loc[:, predictor_columns]
    predictor_test = test_feat.loc[:, predictor_columns]
    preprocessor = lookup_backend.LookupPreprocessor(
        lookup_columns=lookup_columns,
        numeric_columns=numeric_columns,
        decimal_places=decimal_places,
    ).fit(predictor_train, predictor_test)
    preprocessing_provenance = preprocessor.provenance()
    _assert_target_free_provenance(preprocessing_provenance)
    train_arrays = preprocessor.transform(predictor_train)
    test_arrays = preprocessor.transform(predictor_test)

    y_np = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    y_np = np.asarray(y_np).reshape(-1)
    if len(y_np) != len(train_arrays.lookup_ids):
        raise ValueError("Training labels must align with transformed predictors")

    cv_cfg = config["cv"]
    splitter = StratifiedKFold(
        n_splits=int(cv_cfg["n_splits"]),
        shuffle=bool(cv_cfg.get("shuffle", True)),
        random_state=seed,
    )
    dtype = _float_dtype(config)
    oof = np.zeros(len(train_arrays.lookup_ids), dtype=dtype)
    test_pred = np.zeros(len(test_arrays.lookup_ids), dtype=dtype)
    fold_ids = np.full(len(train_arrays.lookup_ids), -1, dtype=np.int16)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    fold_diagnostics: list[dict[str, Any]] = []
    accelerator = get_accelerator(config)
    device = "cuda" if accelerator == "gpu" else "cpu"
    params = dict(config["model"].get("params") or {})

    print(
        f"model={config['model']['name']} backend=lookup_transformer "
        f"accelerator={accelerator} n_lookup={len(lookup_columns)} "
        f"n_numeric_tokens={len(numeric_columns)}",
        flush=True,
    )
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(train_arrays.numeric_values, y_np)
    ):
        fold_seed = seed + fold
        fold_ids[valid_index] = fold
        fold_train_arrays = lookup_backend.LookupBatchArrays(
            lookup_ids=train_arrays.lookup_ids[train_index],
            numeric_values=train_arrays.numeric_values[train_index],
            missing_mask=train_arrays.missing_mask[train_index],
        )
        fold_valid_arrays = lookup_backend.LookupBatchArrays(
            lookup_ids=train_arrays.lookup_ids[valid_index],
            numeric_values=train_arrays.numeric_values[valid_index],
            missing_mask=train_arrays.missing_mask[valid_index],
        )
        valid_pred, fold_test_pred, best_epoch, diagnostics = (
            lookup_backend.train_lookup_fold(
                train_arrays=fold_train_arrays,
                train_y=y_np[train_index],
                valid_arrays=fold_valid_arrays,
                valid_y=y_np[valid_index],
                test_arrays=test_arrays,
                lookup_cardinalities=preprocessor.lookup_cardinalities,
                params=dict(params),
                seed=fold_seed,
                device=device,
            )
        )
        valid_pred = np.asarray(valid_pred, dtype=dtype).reshape(-1)
        fold_test_pred = np.asarray(fold_test_pred, dtype=dtype).reshape(-1)
        if len(valid_pred) != len(valid_index):
            raise ValueError("Lookup validation predictions are misaligned")
        if len(fold_test_pred) != len(test_pred):
            raise ValueError("Lookup test predictions are misaligned")
        if not np.isfinite(valid_pred).all() or not np.isfinite(fold_test_pred).all():
            raise ValueError("Lookup fold predictions must be finite")
        oof[valid_index] = valid_pred
        test_pred += fold_test_pred / splitter.n_splits
        auc = float(roc_auc_score(y_np[valid_index], valid_pred))
        fold_scores.append(auc)
        best_iterations.append(int(best_epoch))
        fold_record = dict(diagnostics)
        fold_record.update(
            {
                "fold": int(fold),
                "seed": int(fold_seed),
                "auc": auc,
                "best_epoch": int(best_epoch),
            }
        )
        fold_diagnostics.append(fold_record)
        print(
            f"[fold {fold + 1}] AUC={auc:.6f} best_epoch={best_epoch} "
            f"seed={fold_seed}",
            flush=True,
        )

    if np.any(fold_ids < 0):
        raise RuntimeError("Every training row must receive exactly one validation fold")
    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    oof_auc = float(roc_auc_score(y_np, oof))
    print(f"[cv] mean={cv_mean:.6f} std={cv_std:.6f} oof={oof_auc:.6f}")
    return {
        "oof": oof,
        "test_pred": test_pred,
        "fold_ids": fold_ids,
        "feature_names": list(numeric_columns),
        "lookup_columns": list(lookup_columns),
        "cat_cols": [],
        "data_provenance": {
            "lookup_preprocessing": preprocessing_provenance,
        },
        "preprocessing_provenance": preprocessing_provenance,
        "fold_diagnostics": fold_diagnostics,
        "fold_scores": fold_scores,
        "best_iterations": best_iterations,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "oof_auc": oof_auc,
        "train_ids": train_df[config["competition"]["id_col"]].to_numpy(),
        "test_ids": test_df[config["competition"]["id_col"]].to_numpy(),
        "y": y_np,
        "backend": "lookup_transformer",
        "te_config": None,
        "te_fold_stats": [],
        "feature_importances": [],
    }


def _run_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
    fold_fn: FoldTrainer,
) -> dict[str, Any]:
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    X, X_test, y_np, cols, cat_cols, data_provenance = _prepare_xy(
        train_df, test_df, y, config
    )
    te_cfg = parse_exact_te_config(config)
    cv_cfg = config["cv"]
    splitter = StratifiedKFold(
        n_splits=int(cv_cfg["n_splits"]),
        shuffle=bool(cv_cfg.get("shuffle", True)),
        random_state=seed,
    )
    dtype = _float_dtype(config)
    oof = np.zeros(len(X), dtype=dtype)
    test_pred = np.zeros(len(X_test), dtype=dtype)
    fold_ids = np.full(len(X), -1, dtype=np.int16)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    te_fold_stats: list[dict[str, Any]] = []
    feature_importances: list[dict[str, dict[str, float]]] = []
    feature_names = list(cols)
    ctx = {
        "config": config,
        "cols": cols,
        "cat_cols": cat_cols,
        "seed": seed,
        "accelerator": get_accelerator(config),
        "feature_importances": feature_importances,
        "data_provenance": data_provenance,
    }
    te_note = ""
    if te_cfg is not None:
        te_note = f" exact_te_cols={te_cfg['columns']}"
    print(
        f"model={config['model']['name']} backend={resolve_backend(config)} "
        f"accelerator={ctx['accelerator']} n_raw_features={len(cols)}{te_note}"
    )

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y_np)):
        display_fold = fold + 1
        fold_ids[va_idx] = fold
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]
        X_te = X_test
        if te_cfg is not None:
            X_tr, X_va, X_te, te_stats = apply_fold_target_encoding(
                X_tr, y_tr, X_va, X_test, te_cfg
            )
            te_fold_stats.append(te_stats)
            feature_names = list(X_tr.columns)
            ctx["cols"] = feature_names
            n_unseen = sum(int(v["n_val_unseen"]) for v in te_stats["columns"].values())
            n_rare = sum(int(v["n_val_rare"]) for v in te_stats["columns"].values())
            print(
                f"[fold {display_fold}] n_features={len(feature_names)} "
                f"te_unseen={n_unseen} te_rare={n_rare}"
            )
        va_pred, te_pred, best_iter = fold_fn(X_tr, y_tr, X_va, y_va, X_te, ctx)
        oof[va_idx] = va_pred.astype(dtype, copy=False)
        test_pred += te_pred.astype(dtype, copy=False) / splitter.n_splits
        auc = float(roc_auc_score(y_va, va_pred))
        fold_scores.append(auc)
        best_iterations.append(int(best_iter))
        print(f"[fold {display_fold}] AUC={auc:.6f} best_iter={best_iter}")

    if np.any(fold_ids < 0):
        raise RuntimeError("Every training row must receive exactly one validation fold")

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    oof_auc = float(roc_auc_score(y_np, oof))
    print(f"[cv] mean={cv_mean:.6f} std={cv_std:.6f} oof={oof_auc:.6f}")
    return {
        "oof": oof,
        "test_pred": test_pred,
        "fold_ids": fold_ids,
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "data_provenance": data_provenance,
        "fold_scores": fold_scores,
        "best_iterations": best_iterations,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "oof_auc": oof_auc,
        "train_ids": train_df[config["competition"]["id_col"]].to_numpy(),
        "test_ids": test_df[config["competition"]["id_col"]].to_numpy(),
        "y": y_np,
        "backend": resolve_backend(config),
        "te_config": te_cfg,
        "te_fold_stats": te_fold_stats,
        "feature_importances": feature_importances,
    }


def train_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    backend = resolve_backend(config)
    if backend == "lookup_transformer":
        return _run_lookup_cv(train_df, test_df, y, config)
    trainers = {
        "lightgbm": _fold_lightgbm,
        "xgboost": _fold_xgboost,
        "catboost": _fold_catboost,
        "histgb": _fold_histgb,
        "logreg": _fold_logreg,
    }
    return _run_cv(train_df, test_df, y, config, trainers[backend])


def _fold_lightgbm(X_tr, y_tr, X_va, y_va, X_test, ctx):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("Install lightgbm to use model.name: lightgbm") from exc

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
    names = list(X_tr.columns)
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    ctx.setdefault("feature_importances", []).append(
        {
            "gain": {n: float(g) for n, g in zip(names, gain)},
            "split": {n: float(s) for n, s in zip(names, split)},
        }
    )
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
    importance = model.get_feature_importance()
    ctx.setdefault("feature_importances", []).append(
        {
            "gain": {
                name: float(value)
                for name, value in zip(X_tr.columns, importance)
            }
        }
    )
    return va_pred, te_pred, best_iter


def _callable_accepts(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _filter_init_kwargs(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return params
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return params
    allowed = set(sig.parameters) - {"self"}
    dropped = sorted(k for k in params if k not in allowed)
    if dropped:
        print(f"Dropping unsupported {cls.__name__} params: {dropped}", flush=True)
        return {k: v for k, v in params.items() if k in allowed}
    return params


def fit_histgb(clf: Any, X_tr, y_tr, X_va, y_va) -> Any:
    """Fit HistGB on sklearn 1.7+ (X_val) and older Kaggle images (no X_val)."""
    if _callable_accepts(clf.fit, "X_val"):
        clf.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
        return clf
    import sklearn

    print(
        f"sklearn {sklearn.__version__} HistGB.fit has no X_val; "
        "early-stopping uses validation_fraction on the fold train split.",
        flush=True,
    )
    clf.fit(X_tr, y_tr)
    return clf


def _fold_histgb(X_tr, y_tr, X_va, y_va, X_test, ctx):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier

    model_cfg = ctx["config"]["model"]
    params = dict(model_cfg.get("params") or {})
    params.setdefault("random_state", ctx["seed"])
    if "max_iter" not in params and "num_boost_round" in model_cfg:
        params["max_iter"] = int(model_cfg["num_boost_round"])
    if "n_iter_no_change" not in params and "early_stopping_rounds" in model_cfg:
        params["n_iter_no_change"] = int(model_cfg["early_stopping_rounds"])
        params.setdefault("early_stopping", True)
    if _callable_accepts(HistGradientBoostingClassifier.__init__, "categorical_features"):
        params.setdefault("categorical_features", "from_dtype")
    params = _filter_init_kwargs(HistGradientBoostingClassifier, params)
    print(f"histgb sklearn={sklearn.__version__}", flush=True)
    clf = HistGradientBoostingClassifier(**params)
    fit_histgb(clf, X_tr, y_tr, X_va, y_va)
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


def _mean_feature_importance(
    fold_importances: list[dict[str, dict[str, float]]],
) -> dict[str, float]:
    gains: dict[str, list[float]] = {}
    for fold in fold_importances:
        gain = fold.get("gain") or {}
        for name, value in gain.items():
            gains.setdefault(name, []).append(float(value))
    return {name: float(np.mean(vals)) for name, vals in gains.items()}


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
        oof_predictions_pq = oof_dir / "oof_predictions.parquet"
        oof_frame = pd.DataFrame(
            {
                id_col: artifacts["train_ids"],
                target: artifacts["y"],
                "pred": artifacts["oof"],
                "fold": artifacts["fold_ids"],
            }
        )
        oof_frame.to_parquet(oof_pq, index=False)
        oof_frame.to_parquet(oof_predictions_pq, index=False)
        written["oof_npy"] = _relpath(oof_npy)
        written["oof_parquet"] = _relpath(oof_pq)
        written["oof_predictions"] = _relpath(oof_predictions_pq)

    if output_cfg.get("save_test", True):
        test_npy = oof_dir / "test.npy"
        np.save(test_npy, artifacts["test_pred"])
        test_pq = oof_dir / "test.parquet"
        test_predictions_pq = oof_dir / "test_predictions.parquet"
        test_frame = pd.DataFrame(
            {id_col: artifacts["test_ids"], "pred": artifacts["test_pred"]}
        )
        test_frame.to_parquet(test_pq, index=False)
        test_frame.to_parquet(test_predictions_pq, index=False)
        written["test_npy"] = _relpath(test_npy)
        written["test_parquet"] = _relpath(test_pq)
        written["test_predictions"] = _relpath(test_predictions_pq)

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
    backend = artifacts.get("backend") or resolve_backend(config)
    versions = dependency_versions(backend)
    metrics = {
        **summary,
        "package_version": package_version,
        "lightgbm_version": versions["lightgbm"],
        "dependency_versions": versions,
        "metric": config["competition"]["metric"],
        "cv_mean": artifacts["cv_mean"],
        "cv_std": artifacts["cv_std"],
        "oof_auc": artifacts["oof_auc"],
        "fold_scores": artifacts["fold_scores"],
        "fold_auc_min": float(min(artifacts["fold_scores"])),
        "fold_auc_max": float(max(artifacts["fold_scores"])),
        "best_iterations": artifacts["best_iterations"],
        "n_features": len(artifacts["feature_names"]),
        "feature_names": artifacts["feature_names"],
        "n_categorical_features": len(artifacts.get("cat_cols") or []),
        "categorical_feature_names": artifacts.get("cat_cols") or [],
        "data_provenance": artifacts.get("data_provenance") or {},
        "n_train": int(len(artifacts["oof"])),
        "n_test": int(len(artifacts["test_pred"])),
        "n_splits": int(config["cv"]["n_splits"]),
        "model_name": config["model"]["name"],
        "config_path": config.get("_config_path"),
    }
    te_cfg = artifacts.get("te_config")
    if te_cfg:
        metrics["target_encoding"] = te_cfg
    te_fold_stats = artifacts.get("te_fold_stats")
    if te_fold_stats:
        metrics["te_fold_stats"] = te_fold_stats
    preprocessing_provenance = artifacts.get("preprocessing_provenance")
    if preprocessing_provenance is not None:
        _assert_target_free_provenance(preprocessing_provenance)
        metrics["preprocessing_provenance"] = preprocessing_provenance
    fold_diagnostics = artifacts.get("fold_diagnostics")
    if fold_diagnostics is not None:
        metrics["fold_diagnostics"] = fold_diagnostics
    importance_mean = _mean_feature_importance(artifacts.get("feature_importances") or [])
    if importance_mean:
        metrics["feature_importance_gain_mean"] = importance_mean
    metrics_path = oof_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    written["metrics"] = _relpath(metrics_path)

    experiment_path = oof_dir / "experiment.json"
    resolved_config = {k: v for k, v in config.items() if k != "_config_path"}
    experiment_payload = {
        **summary,
        "dependency_versions": versions,
        "resolved_config": resolved_config,
        "feature_names": artifacts["feature_names"],
        "categorical_feature_names": artifacts.get("cat_cols") or [],
        "data_provenance": artifacts.get("data_provenance") or {},
    }
    if preprocessing_provenance is not None:
        experiment_payload["preprocessing_provenance"] = preprocessing_provenance
    if fold_diagnostics is not None:
        experiment_payload["fold_diagnostics"] = fold_diagnostics
    experiment_path.write_text(
        json.dumps(experiment_payload, indent=2), encoding="utf-8"
    )
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
