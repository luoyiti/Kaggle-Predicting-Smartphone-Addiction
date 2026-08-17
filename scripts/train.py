#!/usr/bin/env python3
"""Train a config-driven experiment and persist OOF / test predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.data import load_config, load_test, load_train, split_xy
from s6e8.models.train import save_artifacts, train_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train S6E8 experiment")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to experiment YAML (relative to repo root or absolute)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    print(f"experiment={config['experiment']['name']}")
    print(f"config={config['_config_path']}")
    print(
        "versions "
        f"data={config['experiment']['data_version']} "
        f"features={config['experiment']['feature_version']} "
        f"model={config['experiment']['model_version']}"
    )

    train_df = load_train(config)
    test_df = load_test(config)
    X_train, y = split_xy(train_df, config)

    print(f"train={len(X_train):,} test={len(test_df):,} target={y.name}")
    artifacts = train_cv(X_train, test_df, y, config)
    written = save_artifacts(artifacts, config)
    print("wrote:")
    for key, path in written.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
