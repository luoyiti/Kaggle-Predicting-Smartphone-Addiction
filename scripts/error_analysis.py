#!/usr/bin/env python3
"""Cluster OOF mistakes and optionally write a next-experiment YAML stub."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.data import load_config, load_train
from s6e8.error_analysis import analyze_oof_errors, render_next_experiment_yaml
from s6e8.oof_io import join_oof_with_train, load_experiment_oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOF error analysis → next experiment")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", default="configs/lgbm_nocat.yaml")
    parser.add_argument("--oof-dir", default="oof")
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--write-next-config",
        default=None,
        help="If the analyzer suggests a new experiment, write this YAML path",
    )
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
    analysis = analyze_oof_errors(merged, y, pred, experiment=args.experiment)
    suggestion = analysis["suggestion"]
    print(json.dumps(
        {
            "oof_auc": analysis["oof_auc"],
            "hard_band_n": analysis["hard_band_n"],
            "hard_band_auc": analysis["hard_band_auc"],
            "max_residual_column_auc": analysis["max_residual_column_auc"],
            "suggestion": suggestion,
        },
        indent=2,
    ))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json_out}")
    stub = render_next_experiment_yaml(analysis)
    if args.write_next_config and stub:
        path = Path(args.write_next_config)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing config {path}")
        path.write_text(stub, encoding="utf-8")
        print(f"wrote next experiment stub {path}")
    elif args.write_next_config and not stub:
        print("no new experiment YAML: analyzer says stop or ensemble")


if __name__ == "__main__":
    main()
