#!/usr/bin/env python3
"""Copy a downloaded Kaggle kernel OOF dump into ``oof/<experiment>/``.

The CPU drop-cats harvest writes to ``oof/catboost_exactcat_budget_v1/``.
That folder currently holds the PR #11 GPU orig-cats dump. This script
refuses to replace it unless incoming ``metrics.json`` says
``accelerator=cpu`` and ``n_categorical_features=9``.

Example:
  python scripts/install_kernel_oof.py \\
    --src artifacts/kaggle-output \\
    --experiment catboost_exactcat_budget_v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from s6e8.data import PROJECT_ROOT
from s6e8.oof_guard import (
    CPU_DROPCATS_EXPERIMENT,
    PR11_ORIGCATS_ALIAS,
    REQUIRED_OOF_FILES,
    OofProtocolError,
    check_npy_shape,
    is_cpu_dropcats_budget,
    looks_like_pr11_origcats,
    validate_cpu_dropcats_budget_metrics,
    validate_prediction_frames,
)


COPY_NAMES = (
    "oof.parquet",
    "test.parquet",
    "oof.npy",
    "test.npy",
    "metrics.json",
    "experiment.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install harvested kernel OOF into oof/")
    parser.add_argument("--src", required=True, help="Kernel output dir or oof/<exp> dir")
    parser.add_argument("--experiment", required=True, help="Destination experiment folder name")
    parser.add_argument("--oof-dir", default="oof")
    parser.add_argument(
        "--preserve-existing-as",
        default=PR11_ORIGCATS_ALIAS,
        help=(
            "If dest is the PR #11 GPU orig-cats dump, copy it here before replace "
            f"(default {PR11_ORIGCATS_ALIAS})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing a dest that already looks like CPU drop-cats",
    )
    return parser.parse_args()


def _load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def find_experiment_dir(src: Path, experiment: str) -> Path:
    candidates = [src / "oof" / experiment, src / experiment, src]
    hits: list[Path] = []
    seen: set[Path] = set()
    for cand in candidates:
        if not cand.is_dir():
            continue
        metrics_path = cand / "metrics.json"
        if not metrics_path.is_file():
            continue
        resolved = cand.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        metrics = _load_metrics(metrics_path)
        name = str(metrics.get("experiment") or cand.name)
        if name == experiment or cand.name == experiment:
            hits.append(cand)
    if not hits:
        for metrics_path in sorted(src.rglob("metrics.json")):
            metrics = _load_metrics(metrics_path)
            name = str(metrics.get("experiment") or metrics_path.parent.name)
            if name == experiment:
                hits.append(metrics_path.parent)
    if not hits:
        raise FileNotFoundError(
            f"No metrics.json for experiment {experiment!r} under {src}"
        )
    return hits[0]


def _copy_tree(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in COPY_NAMES:
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, dest / name)


def install_experiment(
    src: Path,
    experiment: str,
    oof_root: Path,
    *,
    preserve_existing_as: str | None,
    overwrite: bool,
) -> Path:
    incoming = find_experiment_dir(src, experiment)
    missing = [name for name in REQUIRED_OOF_FILES if not (incoming / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incoming {incoming} missing {missing}")
    metrics = _load_metrics(incoming / "metrics.json")
    oof = pd.read_parquet(incoming / "oof.parquet")
    test = pd.read_parquet(incoming / "test.parquet")
    oof_pred, test_pred = validate_prediction_frames(experiment, oof, test, metrics)
    check_npy_shape(incoming, "oof", oof_pred)
    check_npy_shape(incoming, "test", test_pred)

    dest = oof_root / experiment
    if experiment == CPU_DROPCATS_EXPERIMENT:
        validate_cpu_dropcats_budget_metrics(metrics, experiment=experiment)
        if dest.is_dir() and (dest / "metrics.json").is_file():
            existing = _load_metrics(dest / "metrics.json")
            if is_cpu_dropcats_budget(existing) and not overwrite:
                raise OofProtocolError(
                    f"Refuse to overwrite oof/{experiment}/: dest already looks like "
                    "CPU drop-cats (accelerator=cpu, n_cat=9). Pass --overwrite to replace."
                )
            if looks_like_pr11_origcats(existing) and preserve_existing_as:
                alias = oof_root / preserve_existing_as
                if alias.resolve() == dest.resolve():
                    raise OofProtocolError(
                        "--preserve-existing-as cannot be the same as --experiment"
                    )
                if not (alias / "metrics.json").is_file():
                    print(
                        f"Preserving PR #11 GPU orig-cats dump as oof/{preserve_existing_as}/",
                        flush=True,
                    )
                    _copy_tree(dest, alias)
                else:
                    print(
                        f"oof/{preserve_existing_as}/ already exists; leaving it untouched",
                        flush=True,
                    )
    elif dest.is_dir() and (dest / "metrics.json").is_file() and not overwrite:
        raise FileExistsError(
            f"oof/{experiment}/ already exists. Pass --overwrite to replace."
        )

    if dest.exists():
        shutil.rmtree(dest)
    _copy_tree(incoming, dest)
    print(f"installed {incoming} -> {dest}", flush=True)
    print(
        f"accelerator={metrics.get('accelerator')} "
        f"n_categorical_features={metrics.get('n_categorical_features')} "
        f"n_splits={metrics.get('n_splits')} "
        f"oof_auc={metrics.get('oof_auc')}",
        flush=True,
    )
    return dest


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    oof_root = Path(args.oof_dir)
    if not oof_root.is_absolute():
        oof_root = PROJECT_ROOT / oof_root
    install_experiment(
        src,
        args.experiment,
        oof_root,
        preserve_existing_as=args.preserve_existing_as,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
