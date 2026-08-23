#!/usr/bin/env python3
"""Fail-closed promotion check: candidate metrics vs baseline OOF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.oof_io import load_experiment_oof, resolve_oof_root
from s6e8.promotion import evaluate_promotion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a candidate experiment vs baseline")
    parser.add_argument("--candidate", required=True, help="Candidate experiment name")
    parser.add_argument("--baseline", default="lgbm_nocat")
    parser.add_argument("--oof-dir", default="oof")
    parser.add_argument(
        "--gates",
        default="configs/ensemble/promotion_gates.yaml",
        help="YAML with a promotion: mapping (optional file)",
    )
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="If baseline OOF is absent, skip delta check (still fail closed on candidate)",
    )
    return parser.parse_args()


def _load_gates(path_like: str) -> dict:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict) and "promotion" in raw:
        return dict(raw["promotion"])
    return dict(raw) if isinstance(raw, dict) else {}


def _metrics_from_exp(exp: str, oof_dir: str) -> dict:
    folder = resolve_oof_root(oof_dir) / exp
    metrics_path = folder / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    loaded = load_experiment_oof(exp, oof_dir)
    metrics = dict(loaded["metrics"] or {})
    metrics.setdefault("oof_auc", None)
    metrics.setdefault("n_train", len(loaded["oof"]))
    metrics.setdefault("n_test", len(loaded["test"]))
    return metrics


def main() -> None:
    args = parse_args()
    gates = _load_gates(args.gates)
    try:
        candidate = _metrics_from_exp(args.candidate, args.oof_dir)
    except FileNotFoundError as exc:
        raise SystemExit(f"FAIL closed: {exc}") from exc

    baseline = None
    try:
        baseline = _metrics_from_exp(args.baseline, args.oof_dir)
    except FileNotFoundError:
        if not args.allow_missing_baseline:
            raise SystemExit(
                f"FAIL closed: baseline OOF missing for {args.baseline!r}. "
                "Pass --allow-missing-baseline only for smoke tests."
            )

    result = evaluate_promotion(candidate, baseline, gates)
    print(json.dumps({"passed": result.passed, "failures": result.failures, "checks": result.checks}, indent=2))
    if not result.passed:
        raise SystemExit("promotion gates failed")
    print("PROMOTION OK")


if __name__ == "__main__":
    main()
