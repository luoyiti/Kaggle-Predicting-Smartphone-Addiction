#!/usr/bin/env python3
"""Blend saved OOF predictions by maximizing OOF ROC-AUC.

Example:
  python scripts/blend_oof.py --experiments lgbm_nocat histgb_nocat --method grid
  python scripts/blend_oof.py --experiments lgbm_nocat histgb_nocat --method auc_weighted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from s6e8.ensemble import blend_predictions
from s6e8.oof_io import load_experiment_oof, write_prediction_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend experiment OOF / test predictions")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--oof-dir", default="oof")
    parser.add_argument(
        "--method",
        choices=["mean", "rank", "logit", "grid", "auc_weighted"],
        default="grid",
    )
    parser.add_argument("--name", default=None, help="Output experiment name")
    parser.add_argument("--submission-dir", default="submissions")
    parser.add_argument(
        "--write-experiment-record",
        action="store_true",
        help="Write experiments/<name>.json (do not use for synthetic smoke)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = [load_experiment_oof(exp, args.oof_dir) for exp in args.experiments]
    y = loaded[0]["y"]
    ids = loaded[0]["train_ids"]
    test_ids = loaded[0]["test_ids"]
    oofs = []
    tests = []
    rows = []
    for item, exp in zip(loaded, args.experiments):
        if not np.array_equal(ids, item["train_ids"]):
            raise ValueError(f"{exp} OOF ids do not align")
        if not np.array_equal(test_ids, item["test_ids"]):
            raise ValueError(f"{exp} test ids do not align")
        oofs.append(item["oof_pred"])
        tests.append(item["test_pred"])
        auc = float((item["metrics"] or {}).get("oof_auc", np.nan))
        rows.append((exp, auc))
        print(f"{exp}: oof_auc={auc} n_train={len(item['oof'])}")

    stacked = np.vstack(oofs)
    corr = np.corrcoef(stacked)
    print("OOF Pearson correlation:")
    print(pd.DataFrame(corr, index=args.experiments, columns=args.experiments).round(4))

    result = blend_predictions(
        oofs,
        tests,
        y,
        method=args.method,
        component_aucs=[a for _, a in rows],
    )
    weight_map = {k: float(v) for k, v in zip(args.experiments, result["weights"])}
    print(f"blend method={args.method} weights={weight_map} oof_auc={result['oof_auc']:.6f}")
    singles = ", ".join(f"{n}={a:.6f}" for n, a in rows)
    print(f"components: {singles}")

    name = args.name or ("blend_" + "_".join(args.experiments))
    metrics = {
        "experiment": name,
        "method": args.method,
        "components": args.experiments,
        "weights": weight_map,
        "oof_auc": result["oof_auc"],
        "cv_auc": result["oof_auc"],
        "component_auc": {k: v for k, v in rows},
        "oof_corr": result["oof_corr"].tolist(),
        "n_train": int(len(result["oof"])),
        "n_test": int(len(result["test"])),
        "diagnostic": False,
        "change": f"{args.method} blend of {', '.join(args.experiments)}",
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
    if not args.write_experiment_record:
        print("note: skipped experiments/*.json (pass --write-experiment-record for Kernel blends)")


if __name__ == "__main__":
    main()
