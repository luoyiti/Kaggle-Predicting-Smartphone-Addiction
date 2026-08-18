#!/usr/bin/env python3
"""Blend saved OOF predictions by maximizing OOF ROC-AUC.

Example:
  python scripts/blend_oof.py --experiments lgbm_raw histgb_raw --method rank
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8.data import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend experiment OOF / test predictions")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--oof-dir", default="oof")
    parser.add_argument("--method", choices=["mean", "rank", "logit", "grid"], default="grid")
    parser.add_argument("--name", default=None, help="Output experiment name")
    parser.add_argument("--submission-dir", default="submissions")
    return parser.parse_args()


def _load(exp: str, oof_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    folder = oof_root / exp
    oof = pd.read_parquet(folder / "oof.parquet")
    test = pd.read_parquet(folder / "test.parquet")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    return oof, test, metrics


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy() / (len(x) + 1.0)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _grid_weights(n: int, step: int = 5) -> list[tuple[float, ...]]:
    ticks = list(range(0, 101, step))
    out = []
    for combo in product(ticks, repeat=n):
        if sum(combo) != 100:
            continue
        if all(v == 0 for v in combo):
            continue
        out.append(tuple(v / 100.0 for v in combo))
    return out


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
    for exp in args.experiments:
        oof, test, metrics = _load(exp, oof_root)
        oof = oof.sort_values("id").reset_index(drop=True)
        test = test.sort_values("id").reset_index(drop=True)
        if y is None:
            y = oof["addicted_label"].to_numpy()
            ids = oof["id"].to_numpy()
            test_ids = test["id"].to_numpy()
        else:
            if not np.array_equal(ids, oof["id"].to_numpy()):
                raise ValueError(f"{exp} OOF ids do not align")
            if not np.array_equal(test_ids, test["id"].to_numpy()):
                raise ValueError(f"{exp} test ids do not align")
        oofs.append(oof["pred"].to_numpy())
        tests.append(test["pred"].to_numpy())
        rows.append((exp, float(metrics.get("oof_auc", np.nan))))
        print(f"{exp}: oof_auc={metrics.get('oof_auc')} n_train={len(oof)}")

    stacked = np.vstack(oofs)
    corr = np.corrcoef(stacked)
    print("OOF Pearson correlation:")
    print(pd.DataFrame(corr, index=args.experiments, columns=args.experiments).round(4))

    method = args.method
    if method == "mean":
        weights = np.ones(len(args.experiments)) / len(args.experiments)
        blend_oof = stacked.mean(axis=0)
        blend_test = np.vstack(tests).mean(axis=0)
    elif method == "rank":
        weights = np.ones(len(args.experiments)) / len(args.experiments)
        blend_oof = np.mean([_rank(x) for x in oofs], axis=0)
        blend_test = np.mean([_rank(x) for x in tests], axis=0)
    elif method == "logit":
        weights = np.ones(len(args.experiments)) / len(args.experiments)
        blend_oof = _sigmoid(np.mean([_logit(x) for x in oofs], axis=0))
        blend_test = _sigmoid(np.mean([_logit(x) for x in tests], axis=0))
    else:
        best_auc = -1.0
        weights = np.ones(len(args.experiments)) / len(args.experiments)
        blend_oof = stacked.mean(axis=0)
        for w in _grid_weights(len(args.experiments)):
            pred = np.tensordot(w, stacked, axes=(0, 0))
            auc = float(roc_auc_score(y, pred))
            if auc > best_auc:
                best_auc = auc
                weights = np.array(w, dtype=float)
                blend_oof = pred
        blend_test = np.tensordot(weights, np.vstack(tests), axes=(0, 0))

    auc = float(roc_auc_score(y, blend_oof))
    weight_map = {k: float(v) for k, v in zip(args.experiments, weights)}
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
    pd.DataFrame({"id": test_ids, "pred": blend_test}).to_parquet(out_dir / "test.parquet", index=False)
    metrics = {
        "experiment": name,
        "method": method,
        "components": args.experiments,
        "weights": {k: float(v) for k, v in zip(args.experiments, weights)},
        "oof_auc": auc,
        "component_auc": {k: v for k, v in rows},
        "oof_corr": corr.tolist(),
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
        "change": f"grid/mean blend of {', '.join(args.experiments)}",
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
