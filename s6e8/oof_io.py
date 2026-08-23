"""Load saved OOF / test prediction folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from s6e8.data import PROJECT_ROOT


def resolve_oof_root(oof_dir: str | Path = "oof") -> Path:
    path = Path(oof_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def join_oof_with_train(
    train_df: pd.DataFrame,
    oof_payload: dict[str, Any],
    *,
    id_col: str = "id",
    target: str = "addicted_label",
) -> pd.DataFrame:
    """Align train features with OOF labels/preds without colliding on target name."""
    feat = train_df.drop(columns=[target], errors="ignore")
    oof_df = oof_payload["oof"][[id_col, "pred"]].copy()
    oof_df[target] = oof_payload["y"]
    merged = feat.merge(oof_df, on=id_col, how="inner")
    if len(merged) != len(oof_payload["oof"]):
        raise ValueError(
            f"id mismatch: train features {len(feat)} vs OOF {len(oof_payload['oof'])} "
            f"merged {len(merged)}"
        )
    return merged


def load_experiment_oof(exp: str, oof_root: str | Path = "oof") -> dict[str, Any]:
    folder = resolve_oof_root(oof_root) / exp
    oof_path = folder / "oof.parquet"
    test_path = folder / "test.parquet"
    metrics_path = folder / "metrics.json"
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"OOF artifacts missing for {exp!r} under {folder}. "
            "Run scripts/train.py first; diagnostic/Kernel outputs are gitignored."
        )
    oof = pd.read_parquet(oof_path).sort_values("id").reset_index(drop=True)
    test = pd.read_parquet(test_path).sort_values("id").reset_index(drop=True)
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "experiment": exp,
        "folder": folder,
        "oof": oof,
        "test": test,
        "metrics": metrics,
        "y": oof["addicted_label"].to_numpy() if "addicted_label" in oof.columns else None,
        "oof_pred": oof["pred"].to_numpy(),
        "test_pred": test["pred"].to_numpy(),
        "train_ids": oof["id"].to_numpy(),
        "test_ids": test["id"].to_numpy(),
    }


def write_prediction_bundle(
    *,
    name: str,
    train_ids: np.ndarray,
    y: np.ndarray,
    oof_pred: np.ndarray,
    test_ids: np.ndarray,
    test_pred: np.ndarray,
    metrics: dict[str, Any],
    oof_root: str | Path = "oof",
    submission_dir: str | Path = "submissions",
    write_experiment_record: bool = False,
) -> dict[str, str]:
    oof_dir = resolve_oof_root(oof_root) / name
    oof_dir.mkdir(parents=True, exist_ok=True)
    np.save(oof_dir / "oof.npy", oof_pred)
    np.save(oof_dir / "test.npy", test_pred)
    pd.DataFrame({"id": train_ids, "addicted_label": y, "pred": oof_pred}).to_parquet(
        oof_dir / "oof.parquet", index=False
    )
    pd.DataFrame({"id": test_ids, "pred": test_pred}).to_parquet(
        oof_dir / "test.parquet", index=False
    )
    (oof_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    written = {
        "oof_dir": str(oof_dir),
        "metrics": str(oof_dir / "metrics.json"),
    }
    if write_experiment_record:
        records_dir = PROJECT_ROOT / "experiments"
        records_dir.mkdir(parents=True, exist_ok=True)
        record_path = records_dir / f"{name}.json"
        record_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        written["experiment_record"] = str(record_path)
    sub_dir = Path(submission_dir)
    if not sub_dir.is_absolute():
        sub_dir = PROJECT_ROOT / sub_dir
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_path = sub_dir / f"{name}.csv"
    pd.DataFrame({"id": test_ids, "addicted_label": test_pred}).to_csv(sub_path, index=False)
    written["submission"] = str(sub_path)
    return written
