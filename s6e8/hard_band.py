"""Leakage-safe hard-band training mask / sample weights.

The band is defined by a *frozen* base-model probability (default: saved
``lgbm_nocat`` OOF). That OOF is out-of-fold, so a row's score never used
that row's label. The specialist may train only on band rows or up-weight
them. The band is never built from the specialist's own predictions or from
the label.

If the frozen file is missing, ``fallback: inner_oof`` fits a small LightGBM
OOF on the *current* train frame only (still fold-safe).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from s6e8.data import PROJECT_ROOT, resolve_path
from s6e8.error_band import band_mask

VALID_MODES = {"train_only", "reweight"}
VALID_EVAL = {"band", "all"}
VALID_FALLBACK = {"none", "inner_oof"}


def parse_hard_band_config(config: dict[str, Any]) -> dict[str, Any] | None:
    raw = (config.get("features") or {}).get("hard_band")
    if not raw or not bool(raw.get("enabled", False)):
        return None
    mode = str(raw.get("mode", "train_only"))
    if mode not in VALID_MODES:
        raise ValueError(f"features.hard_band.mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    eval_on = str(raw.get("eval_on") or ("band" if mode == "train_only" else "all"))
    if eval_on not in VALID_EVAL:
        raise ValueError(f"features.hard_band.eval_on must be one of {sorted(VALID_EVAL)}, got {eval_on!r}")
    fallback = str(raw.get("fallback", "inner_oof"))
    if fallback not in VALID_FALLBACK:
        raise ValueError(
            f"features.hard_band.fallback must be one of {sorted(VALID_FALLBACK)}, got {fallback!r}"
        )
    lo = float(raw.get("lo", 0.3))
    hi = float(raw.get("hi", 0.7))
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(f"features.hard_band needs 0 <= lo < hi <= 1, got lo={lo} hi={hi}")
    return {
        "source_experiment": str(raw.get("source_experiment", "lgbm_nocat")),
        "source_path": raw.get("source_path"),
        "pred_col": str(raw.get("pred_col", "pred")),
        "id_col": str(raw.get("id_col") or config["competition"]["id_col"]),
        "lo": lo,
        "hi": hi,
        "mode": mode,
        "in_band_weight": float(raw.get("in_band_weight", 4.0)),
        "out_band_weight": float(raw.get("out_band_weight", 1.0)),
        "eval_on": eval_on,
        "fallback": fallback,
        "min_train_rows": int(raw.get("min_train_rows", 20)),
        "min_eval_rows": int(raw.get("min_eval_rows", 10)),
    }


def resolve_frozen_oof_path(hb: dict[str, Any], config: dict[str, Any]) -> Path:
    if hb.get("source_path"):
        path = Path(str(hb["source_path"]))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    oof_dir = resolve_path(config["paths"]["oof_dir"])
    return oof_dir / hb["source_experiment"] / "oof.parquet"


def load_frozen_oof(path: Path, *, id_col: str, pred_col: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Hard-band frozen OOF not found: {path}")
    frame = pd.read_parquet(path)
    missing = {id_col, pred_col} - set(frame.columns)
    if missing:
        raise KeyError(f"Frozen OOF {path} is missing columns: {sorted(missing)}")
    return frame[[id_col, pred_col]].copy()


def align_frozen_pred(
    train_ids: np.ndarray,
    frozen: pd.DataFrame,
    *,
    id_col: str,
    pred_col: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Left-align frozen predictions to ``train_ids``. Missing rows are NaN."""
    ids = pd.Series(train_ids)
    lookup = frozen.drop_duplicates(id_col, keep="last").set_index(id_col)[pred_col]
    pred = pd.to_numeric(ids.map(lookup), errors="coerce").to_numpy(dtype=float)
    n_missing = int(np.isnan(pred).sum())
    meta = {
        "n_train": int(len(pred)),
        "n_matched": int(len(pred) - n_missing),
        "n_missing": n_missing,
        "coverage": float((len(pred) - n_missing) / len(pred)) if len(pred) else 0.0,
    }
    return pred, meta


def compute_inner_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
    n_splits: int = 3,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 30,
) -> np.ndarray:
    """Fold-safe LightGBM OOF on the current train frame. Numeric columns only."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if not num_cols:
        raise ValueError("hard_band inner_oof needs at least one numeric column")
    Xn = X[num_cols]
    y = np.asarray(y)
    oof = np.full(len(X), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "verbosity": -1,
        "seed": int(seed),
    }
    for tr_idx, va_idx in splitter.split(Xn, y):
        dtrain = lgb.Dataset(Xn.iloc[tr_idx], label=y[tr_idx], free_raw_data=False)
        dvalid = lgb.Dataset(
            Xn.iloc[va_idx],
            label=y[va_idx],
            reference=dtrain,
            free_raw_data=False,
        )
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=int(num_boost_round),
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(int(early_stopping_rounds), verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof[va_idx] = booster.predict(Xn.iloc[va_idx], num_iteration=booster.best_iteration)
    return oof


def resolve_band_pred(
    train_ids: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    hb: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Frozen OOF if present and covering rows; otherwise optional inner OOF."""
    path = resolve_frozen_oof_path(hb, config)
    meta: dict[str, Any] = {
        "source_path": str(path),
        "source_experiment": hb["source_experiment"],
        "lo": hb["lo"],
        "hi": hb["hi"],
        "mode": hb["mode"],
        "eval_on": hb["eval_on"],
        "used_inner_oof": False,
    }
    pred: np.ndarray | None = None
    if path.is_file():
        frozen = load_frozen_oof(path, id_col=hb["id_col"], pred_col=hb["pred_col"])
        pred, align_meta = align_frozen_pred(
            train_ids, frozen, id_col=hb["id_col"], pred_col=hb["pred_col"]
        )
        meta.update(align_meta)
        meta["source"] = "frozen_oof"
        if align_meta["n_missing"] == 0:
            return pred, meta
        if hb["fallback"] != "inner_oof":
            raise ValueError(
                f"Hard-band frozen OOF {path} missed {align_meta['n_missing']} "
                f"of {align_meta['n_train']} train ids. Set fallback: inner_oof or fix the file."
            )
    elif hb["fallback"] != "inner_oof":
        raise FileNotFoundError(
            f"Hard-band frozen OOF not found: {path}. "
            "Download oof/lgbm_nocat/oof.parquet or set features.hard_band.fallback: inner_oof."
        )

    pred = compute_inner_oof(X, y, seed=int(config["experiment"]["seed"]))
    meta["used_inner_oof"] = True
    meta["source"] = "inner_oof"
    meta["n_train"] = int(len(pred))
    meta["n_matched"] = int(np.isfinite(pred).sum())
    meta["n_missing"] = int((~np.isfinite(pred)).sum())
    meta["coverage"] = float(meta["n_matched"] / meta["n_train"]) if meta["n_train"] else 0.0
    return pred, meta


def row_weights(
    pred: np.ndarray,
    *,
    lo: float,
    hi: float,
    in_band_weight: float,
    out_band_weight: float,
) -> np.ndarray:
    mask = band_mask(pred, lo, hi)
    weights = np.full(len(pred), float(out_band_weight), dtype=float)
    weights[mask] = float(in_band_weight)
    return weights


def apply_hard_band_fold(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    tr_pred: np.ndarray,
    va_pred: np.ndarray,
    hb: dict[str, Any],
) -> dict[str, Any]:
    """Filter / reweight the *training* fold. Validation predictions stay full-length."""
    tr_band = band_mask(tr_pred, hb["lo"], hb["hi"])
    va_band = band_mask(va_pred, hb["lo"], hb["hi"])
    n_tr_band = int(tr_band.sum())
    n_va_band = int(va_band.sum())
    used_train_only = False
    used_eval_band = False
    train_weight = None
    X_tr_out, y_tr_out = X_tr, y_tr
    X_va_eval, y_va_eval = X_va, y_va

    if hb["mode"] == "train_only" and n_tr_band >= hb["min_train_rows"]:
        keep = np.flatnonzero(tr_band)
        X_tr_out = X_tr.iloc[keep]
        y_tr_out = np.asarray(y_tr)[tr_band]
        used_train_only = True
    elif hb["mode"] == "reweight":
        train_weight = row_weights(
            tr_pred,
            lo=hb["lo"],
            hi=hb["hi"],
            in_band_weight=hb["in_band_weight"],
            out_band_weight=hb["out_band_weight"],
        )

    if hb["eval_on"] == "band" and n_va_band >= hb["min_eval_rows"]:
        y_band = np.asarray(y_va)[va_band]
        if len(np.unique(y_band)) >= 2:
            keep_va = np.flatnonzero(va_band)
            X_va_eval = X_va.iloc[keep_va]
            y_va_eval = y_band
            used_eval_band = True

    stats = {
        "n_train": int(len(y_tr)),
        "n_train_band": n_tr_band,
        "n_train_used": int(len(y_tr_out)),
        "n_valid": int(len(y_va)),
        "n_valid_band": n_va_band,
        "n_valid_eval": int(len(y_va_eval)),
        "used_train_only": used_train_only,
        "used_eval_band": used_eval_band,
        "mode": hb["mode"],
    }
    return {
        "X_tr": X_tr_out,
        "y_tr": y_tr_out,
        "X_va_eval": X_va_eval,
        "y_va_eval": y_va_eval,
        "train_weight": train_weight,
        "stats": stats,
    }


def gated_blend(
    base_pred: np.ndarray,
    specialist_pred: np.ndarray,
    *,
    lo: float = 0.3,
    hi: float = 0.7,
    mix: float = 1.0,
) -> np.ndarray:
    """Keep the base model outside the band; mix in the specialist inside it."""
    base = np.asarray(base_pred, dtype=float).copy()
    spec = np.asarray(specialist_pred, dtype=float)
    if base.shape != spec.shape:
        raise ValueError(f"gated_blend shape mismatch: {base.shape} vs {spec.shape}")
    mix = float(mix)
    if not (0.0 <= mix <= 1.0):
        raise ValueError(f"mix must be in [0, 1], got {mix}")
    mask = band_mask(base, lo, hi)
    base[mask] = (1.0 - mix) * base[mask] + mix * spec[mask]
    return base
