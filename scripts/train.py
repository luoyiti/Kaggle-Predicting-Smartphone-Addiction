#!/usr/bin/env python3
"""Train a config-driven experiment and persist OOF / test predictions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.data import load_config, load_test, load_train, split_xy
from s6e8.models.train import (
    apply_diagnostic_overrides,
    assert_oof_available,
    save_artifacts,
    stratified_subsample,
    train_cv,
)
from s6e8.runtime import apply_runtime_override, detect_environment, get_git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train S6E8 experiment")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to experiment YAML (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--accelerator",
        choices=["cpu", "gpu"],
        default=None,
        help="Override YAML runtime.accelerator (cpu|gpu)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing experiment OOF directory",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Stratified subsample of train for ranking only. Renames experiment to *_diagN. Not a competition score.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Override cv.n_splits (use with --max-train-rows for cheap ranking).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_runtime_override(config, accelerator=args.accelerator)
    apply_diagnostic_overrides(
        config,
        max_train_rows=args.max_train_rows,
        n_splits=args.n_splits,
    )

    print(f"experiment={config['experiment']['name']}")
    print(f"config={config['_config_path']}")
    print(f"environment={detect_environment()}")
    print(f"accelerator={config['runtime']['accelerator']}")
    print(
        "versions "
        f"data={config['experiment']['data_version']} "
        f"features={config['experiment']['feature_version']} "
        f"model={config['experiment']['model_version']}"
    )
    git_commit = get_git_commit()
    if git_commit:
        print(f"git_commit={git_commit}")

    assert_oof_available(config, overwrite=args.overwrite)

    train_df = load_train(config)
    test_df = load_test(config)
    max_rows = (config.get("runtime") or {}).get("max_train_rows")
    if max_rows:
        train_df = stratified_subsample(
            train_df,
            config["competition"]["target"],
            int(max_rows),
            int(config["experiment"]["seed"]),
        )
        print(f"subsampled_train={len(train_df):,} (diagnostic)")
    X_train, y = split_xy(train_df, config)

    print(f"train={len(X_train):,} test={len(test_df):,} target={y.name}")
    started = time.perf_counter()
    artifacts = train_cv(X_train, test_df, y, config)
    artifacts["runtime_seconds"] = round(time.perf_counter() - started, 3)
    artifacts["git_commit"] = git_commit
    written = save_artifacts(artifacts, config)
    print("wrote:")
    for key, path in written.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
