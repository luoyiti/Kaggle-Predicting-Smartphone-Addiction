#!/usr/bin/env python3
"""Audit train/test against the S6E8 data contract (schema, missingness, shift, leakage)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.data import load_config, load_test, load_train
from s6e8.data_audit import audit_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S6E8 data-contract audit")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit 1 if the audit records severity=error findings",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_df = load_train(config)
    test_df = load_test(config)
    report = audit_tables(train_df, test_df)
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, indent=2, default=str))
    print("findings:")
    for finding in report["findings"]:
        print(f"  [{finding['severity']}] {finding['title']}: {finding.get('detail')}")
    if not report["findings"]:
        print("  none")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json_out}")
    if args.fail_on_error and not report["ok"]:
        raise SystemExit("data contract audit failed")


if __name__ == "__main__":
    main()
