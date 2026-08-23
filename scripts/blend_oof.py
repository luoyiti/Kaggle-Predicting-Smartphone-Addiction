#!/usr/bin/env python3
"""Blend saved OOF predictions by maximizing OOF ROC-AUC.

Example:
  python scripts/blend_oof.py --experiments lgbm_nocat histgb_nocat --method grid
  python scripts/blend_oof.py --experiments lgbm_nocat catboost_exactcat_v1 --method stack_logistic
  python scripts/blend_oof.py --experiments catboost_exactcat_budget_v1 catboost_exactcat_budget_seed7 catboost_exactcat_budget_seed2026 --method mean --name catboost_exactcat_budget_seedavg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8.blending import (
    BLEND_METHODS,
    blend_grid,
    blend_logit,
    blend_mean,
    blend_rank,
    stack_logistic_cv,
    stack_ridge_cv,
)
from s6e8.data import PROJECT_ROOT
from s6e8.oof_guard import (
    REQUIRED_OOF_FILES,
    check_npy_shape,
    validate_blend_protocol,
    validate_prediction_frames,
    warn_if_budget_v1_not_cpu_dropcats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend experiment OOF / test predictions",
        epilog=(
            "Seed-average CatBoost after Kaggle 5-folds: python scripts/blend_oof.py "
            "--experiments catboost_exactcat_budget_v1 catboost_exactcat_budget_seed7 "
            "catboost_exactcat_budget_seed2026 --method mean "
            "--name catboost_exactcat_budget_seedavg"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--oof-dir", default="oof")
    parser.add_argument("--method", choices=BLEND_METHODS, default="grid")
    parser.add_argument("--name", default=None, help="Output experiment name")
    parser.add_argument("--submission-dir", default="submissions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5, help="Inner CV splits for stack_* methods")
    parser.add_argument("--C", type=float, default=1.0, help="Logistic C for stack_logistic")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha for stack_ridge")
    parser.add_argument("--grid-step", type=int, default=5)
    return parser.parse_args()


def _load(exp: str, oof_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    folder = oof_root / exp
    missing = [name for name in REQUIRED_OOF_FILES if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} for experiment {exp!r} under {folder}. "
            "Train each listed experiment (Kaggle 5-fold) before blending. "
            "Seed-average CatBoost after the three budget seed runs exist:\n"
            "  python scripts/blend_oof.py --experiments "
            "catboost_exactcat_budget_v1 catboost_exactcat_budget_seed7 "
            "catboost_exactcat_budget_seed2026 --method mean "
            "--name catboost_exactcat_budget_seedavg"
        )
    oof = pd.read_parquet(folder / "oof.parquet")
    test = pd.read_parquet(folder / "test.parquet")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    oof_pred, test_pred = validate_prediction_frames(exp, oof, test, metrics)
    check_npy_shape(folder, "oof", oof_pred)
    check_npy_shape(folder, "test", test_pred)
    warning = warn_if_budget_v1_not_cpu_dropcats(exp, metrics)
    if warning:
        print(f"WARNING: {warning}", flush=True)
    return oof, test, metrics


def main() -> None:
    args = parse_args()
    oof_root = Path(args.oof_dir)
    if not oof_root.is_absolute():
        oof_root = PROJECT_ROOT / oof_root

    oofs = []
    tests = []
    y = None
    ids = None
    test_ids = None
    rows = []
    loaded_metrics: list[tuple[str, dict]] = []
    for exp in args.experiments:
        oof, test, metrics = _load(exp, oof_root)
        loaded_metrics.append((exp, metrics))
        oof = oof.sort_values("id").reset_index(drop=True)
        test = test.sort_values("id").reset_index(drop=True)
        oof_pred, test_pred = validate_prediction_frames(exp, oof, test, metrics)
        if y is None:
            if "addicted_label" not in oof.columns:
                raise ValueError(f"{exp} OOF missing addicted_label")
            y = oof["addicted_label"].to_numpy()
            ids = oof["id"].to_numpy()
            test_ids = test["id"].to_numpy()
        else:
            if not np.array_equal(ids, oof["id"].to_numpy()):
                raise ValueError(
                    f"{exp} OOF ids do not align "
                    f"(n={len(oof)} vs first n={len(ids)})"
                )
            if not np.array_equal(test_ids, test["id"].to_numpy()):
                raise ValueError(
                    f"{exp} test ids do not align "
                    f"(n={len(test)} vs first n={len(test_ids)})"
                )
            if oof_pred.shape != oofs[0].shape:
                raise ValueError(
                    f"{exp} OOF pred shape {oof_pred.shape} != {oofs[0].shape}"
                )
            if test_pred.shape != tests[0].shape:
                raise ValueError(
                    f"{exp} test pred shape {test_pred.shape} != {tests[0].shape}"
                )
        if y is not None and oof_pred.shape != np.asarray(y).shape:
            raise ValueError(
                f"{exp} OOF pred shape {oof_pred.shape} != labels {np.asarray(y).shape}"
            )
        oofs.append(oof_pred)
        tests.append(test_pred)
        rows.append((exp, float(metrics.get("oof_auc", np.nan))))
        print(f"{exp}: oof_auc={metrics.get('oof_auc')} n_train={len(oof)}")
    validate_blend_protocol(loaded_metrics)

    stacked = np.vstack(oofs)
    corr = np.corrcoef(stacked)
    print("OOF Pearson correlation:")
    print(pd.DataFrame(corr, index=args.experiments, columns=args.experiments).round(4))

    extra: dict = {}
    method = args.method
    if method == "mean":
        blend_oof, blend_test, weights = blend_mean(oofs, tests)
    elif method == "rank":
        blend_oof, blend_test, weights = blend_rank(oofs, tests)
    elif method == "logit":
        blend_oof, blend_test, weights = blend_logit(oofs, tests)
    elif method == "grid":
        blend_oof, blend_test, weights = blend_grid(oofs, tests, y, step=args.grid_step)
    elif method == "stack_logistic":
        blend_oof, blend_test, weights, extra = stack_logistic_cv(
            oofs, tests, y, seed=args.seed, n_splits=args.n_splits, C=args.C
        )
    elif method == "stack_ridge":
        blend_oof, blend_test, weights, extra = stack_ridge_cv(
            oofs, tests, y, seed=args.seed, n_splits=args.n_splits, alpha=args.alpha
        )
    else:
        raise ValueError(f"Unknown blend method {method!r}")

    auc = float(roc_auc_score(y, blend_oof))
    weight_map = {k: float(v) for k, v in zip(args.experiments, np.asarray(weights).ravel())}
    print(f"blend method={method} weights={weight_map} oof_auc={auc:.6f}")
    singles = ", ".join(f"{n}={a:.6f}" for n, a in rows)
    print(f"components: {singles}")

    name = args.name or ("blend_" + "_".join(args.experiments))
    out_dir = oof_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", blend_oof)
    np.save(out_dir / "test.npy", blend_test)
    pd.DataFrame({"id": ids, "addicted_label": y, "pred": blend_oof}).to_parquet(
        out_dir / "oof.parquet", index=False
    )
    pd.DataFrame({"id": test_ids, "pred": blend_test}).to_parquet(
        out_dir / "test.parquet", index=False
    )
    metrics = {
        "experiment": name,
        "method": method,
        "components": args.experiments,
        "weights": weight_map,
        "oof_auc": auc,
        "component_auc": {k: v for k, v in rows},
        "oof_corr": corr.tolist(),
        **extra,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    records_dir = PROJECT_ROOT / "experiments"
    records_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": name,
        "cv_auc": auc,
        "method": method,
        "components": args.experiments,
        "weights": weight_map,
        "component_auc": {k: v for k, v in rows},
        "n_train": int(len(blend_oof)),
        "n_test": int(len(blend_test)),
        "diagnostic": False,
        "change": f"{method} blend of {', '.join(args.experiments)}",
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
