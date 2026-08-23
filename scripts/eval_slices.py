#!/usr/bin/env python3
"""Slice AUC for a saved OOF against train.csv columns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.data import load_config, load_train
from s6e8.metrics import calibration_summary, slice_metrics
from s6e8.oof_io import join_oof_with_train, load_experiment_oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute slice AUC for a saved OOF")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", default="configs/lgbm_nocat.yaml")
    parser.add_argument("--oof-dir", default="oof")
    parser.add_argument("--output", default=None, help="Optional JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_df = load_train(config)
    oof = load_experiment_oof(args.experiment, args.oof_dir)
    id_col = config["competition"]["id_col"]
    target = config["competition"]["target"]
    merged = join_oof_with_train(train_df, oof, id_col=id_col, target=target)
    y = merged[target].to_numpy()
    pred = merged["pred"].to_numpy()
    slices = slice_metrics(y, pred, merged)
    cal = calibration_summary(y, pred)
    payload = {
        "experiment": args.experiment,
        "slices": slices,
        "calibration": cal,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
