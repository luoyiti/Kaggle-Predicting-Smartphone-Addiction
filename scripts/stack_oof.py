#!/usr/bin/env python3
"""Fold-safe logistic stacker on saved OOF columns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from s6e8.ensemble import logistic_stack
from s6e8.oof_io import load_experiment_oof, write_prediction_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logistic stack of saved OOF predictions")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--oof-dir", default="oof")
    parser.add_argument("--name", default=None)
    parser.add_argument("--submission-dir", default="submissions")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--write-experiment-record", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = [load_experiment_oof(exp, args.oof_dir) for exp in args.experiments]
    y = loaded[0]["y"]
    ids = loaded[0]["train_ids"]
    test_ids = loaded[0]["test_ids"]
    oofs = []
    tests = []
    component_auc = {}
    for item, exp in zip(loaded, args.experiments):
        if not np.array_equal(ids, item["train_ids"]):
            raise ValueError(f"{exp} OOF ids do not align")
        oofs.append(item["oof_pred"])
        tests.append(item["test_pred"])
        component_auc[exp] = (item["metrics"] or {}).get("oof_auc")
        print(f"{exp}: oof_auc={component_auc[exp]}")

    result = logistic_stack(
        oofs, tests, y, n_splits=args.n_splits, seed=args.seed, C=args.C
    )
    print(
        f"stack logistic C={args.C} oof_auc={result['oof_auc']:.6f} "
        f"mean_coef={result['mean_coef'].tolist()}"
    )
    name = args.name or ("stack_" + "_".join(args.experiments))
    metrics = {
        "experiment": name,
        "method": "logistic_stack",
        "components": args.experiments,
        "mean_coef": {k: float(v) for k, v in zip(args.experiments, result["mean_coef"])},
        "mean_intercept": result["mean_intercept"],
        "C": args.C,
        "oof_auc": result["oof_auc"],
        "cv_auc": result["oof_auc"],
        "component_auc": component_auc,
        "n_train": int(len(result["oof"])),
        "n_test": int(len(result["test"])),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "diagnostic": False,
        "change": f"logistic stack of {', '.join(args.experiments)}",
    }
    written = write_prediction_bundle(
        name=name,
        train_ids=ids,
        y=y,
        oof_pred=result["oof"],
        test_ids=test_ids,
        test_pred=result["test"],
        metrics=metrics,
        oof_root=args.oof_dir,
        submission_dir=args.submission_dir,
        write_experiment_record=args.write_experiment_record,
    )
    print("wrote:")
    for key, path in written.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
