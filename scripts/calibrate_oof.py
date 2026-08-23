#!/usr/bin/env python3
"""Fit isotonic or Platt calibration on OOF; apply to test predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.calibration import calibrate_with_oof_cv
from s6e8.oof_io import load_experiment_oof, write_prediction_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate saved OOF / test probabilities")
    parser.add_argument("--experiment", required=True, help="Source experiment with oof.parquet")
    parser.add_argument("--method", choices=["isotonic", "platt"], default="isotonic")
    parser.add_argument("--oof-dir", default="oof")
    parser.add_argument("--name", default=None)
    parser.add_argument("--submission-dir", default="submissions")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-experiment-record", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = load_experiment_oof(args.experiment, args.oof_dir)
    result = calibrate_with_oof_cv(
        src["oof_pred"],
        src["y"],
        src["test_pred"],
        method=args.method,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    print(
        f"source={args.experiment} method={args.method} "
        f"raw_auc={result['raw']['auc']:.6f} cal_auc={result['calibrated']['auc']:.6f} "
        f"raw_ece={result['raw']['ece']:.4f} cal_ece={result['calibrated']['ece']:.4f}"
    )
    name = args.name or f"{args.experiment}_cal_{args.method}"
    metrics = {
        "experiment": name,
        "source_experiment": args.experiment,
        "method": args.method,
        "oof_auc": result["oof_auc"],
        "cv_auc": result["oof_auc"],
        "auc_delta": result["auc_delta"],
        "ece_delta": result["ece_delta"],
        "raw": result["raw"],
        "calibrated": result["calibrated"],
        "calibration": result["calibrated"],
        "n_train": int(len(result["oof"])),
        "n_test": int(len(result["test"])),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "diagnostic": False,
        "change": f"{args.method} calibration of {args.experiment}",
    }
    written = write_prediction_bundle(
        name=name,
        train_ids=src["train_ids"],
        y=src["y"],
        oof_pred=result["oof"],
        test_ids=src["test_ids"],
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
