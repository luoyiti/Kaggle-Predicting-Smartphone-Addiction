#!/usr/bin/env python3
"""Summarize the uncertain OOF band (default pred in (0.3, 0.7)).

Does not train. Reads ``oof/<experiment>/oof.parquet`` written by a prior
Kaggle 5-fold (or diagnostic) run.

Example:
  python scripts/analyze_error_band.py --experiment lgbm_nocat \\
    --compare catboost_exactcat_budget_v1 --train data/raw/train.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from s6e8.data import PROJECT_ROOT
from s6e8.error_band import DEFAULT_NUMERIC, analyze_error_band


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze OOF error band")
    parser.add_argument("--experiment", required=True, help="OOF folder name under --oof-dir")
    parser.add_argument("--compare", default=None, help="Optional second OOF experiment")
    parser.add_argument("--oof-dir", default="oof")
    parser.add_argument("--train", default=None, help="Optional train CSV to join raw features")
    parser.add_argument("--pred-col", default="pred")
    parser.add_argument("--y-col", default="addicted_label")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--lo", type=float, default=0.3)
    parser.add_argument("--hi", type=float, default=0.7)
    parser.add_argument(
        "--out",
        default=None,
        help="JSON path (default experiments/error_band_<experiment>.json)",
    )
    return parser.parse_args()


def _load_oof(root: Path, name: str) -> pd.DataFrame:
    path = root / name / "oof.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Download a kernel OOF before analyzing.")
    return pd.read_parquet(path)


def main() -> None:
    args = parse_args()
    oof_root = Path(args.oof_dir)
    if not oof_root.is_absolute():
        oof_root = PROJECT_ROOT / oof_root
    oof = _load_oof(oof_root, args.experiment)
    compare = _load_oof(oof_root, args.compare) if args.compare else None
    features = None
    if args.train:
        train_path = Path(args.train)
        if not train_path.is_absolute():
            train_path = PROJECT_ROOT / train_path
        features = pd.read_csv(train_path)
    report = analyze_error_band(
        oof,
        pred_col=args.pred_col,
        y_col=args.y_col,
        id_col=args.id_col,
        lo=args.lo,
        hi=args.hi,
        features=features,
        feature_columns=list(DEFAULT_NUMERIC),
        compare=compare,
        compare_name=args.compare,
    )
    report["experiment"] = args.experiment
    out = Path(args.out) if args.out else PROJECT_ROOT / "experiments" / f"error_band_{args.experiment}.json"
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    band = report["band"]
    print(
        f"experiment={args.experiment} n={report['n']} "
        f"auc_all={report['auc_all']:.6f} "
        f"band_n={band['n']} band_pos={band['pos_rate']} band_auc={band['auc']}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
