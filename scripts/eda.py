#!/usr/bin/env python3
"""Run config-driven EDA and write a self-contained HTML report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.data import load_config, load_test, load_train
from s6e8.eda import SCATTER_DEFAULT, run_eda
from s6e8.eda_report import write_html_report
from s6e8.runtime import detect_environment, get_git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S6E8 exploratory data analysis → HTML report")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to experiment YAML (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--output",
        default="reports/eda_report.html",
        help="HTML output path (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SCATTER_DEFAULT,
        help="Max rows for scatter plots (statistics still use the full tables)",
    )
    return parser.parse_args()


def _resolve_output(path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    return path


def main() -> None:
    args = parse_args()
    if args.sample_size < 100:
        raise SystemExit("--sample-size must be >= 100")

    config = load_config(args.config)
    print(f"experiment={config['experiment']['name']}")
    print(f"config={config['_config_path']}")
    print(f"environment={detect_environment()}")
    git_commit = get_git_commit()
    if git_commit:
        print(f"git_commit={git_commit}")

    train_df = load_train(config)
    test_df = load_test(config)
    print(f"train={len(train_df):,} test={len(test_df):,} target={config['competition']['target']}")
    print("running EDA on full tables …")

    result = run_eda(train_df, test_df, config, sample_size=args.sample_size)
    output = _resolve_output(args.output)
    written = write_html_report(result, output)
    size_mb = written.stat().st_size / (1024 * 1024)
    print(f"wrote: {written} ({size_mb:.2f} MiB)")
    print(f"findings={len(result.findings)}")
    for finding in result.findings[:8]:
        print(f"  [{finding.severity}] {finding.title}")


if __name__ == "__main__":
    main()
