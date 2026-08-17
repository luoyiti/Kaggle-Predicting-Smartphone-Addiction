"""Load competition tables and YAML experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    config["_config_path"] = str(path)
    from s6e8.runtime import apply_runtime_override, validate_config

    apply_runtime_override(config)
    validate_config(config)
    return config


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_train(config: dict[str, Any]) -> pd.DataFrame:
    from s6e8.runtime import resolve_input_path

    path = resolve_input_path(config["paths"]["train"], config)
    if not path.exists():
        raise FileNotFoundError(
            f"Train file not found: {path}\n"
            "Local download:\n"
            "  kaggle competitions download -c playground-series-s6e8 -p data/raw\n"
            "  unzip -o data/raw/playground-series-s6e8.zip -d data/raw\n"
            "On Kaggle Kernels the competition dataset is usually mounted under "
            "/kaggle/input/<competition-slug>/ or /kaggle/input/competitions/<competition-slug>/."
        )
    return pd.read_csv(path)


def load_test(config: dict[str, Any]) -> pd.DataFrame:
    from s6e8.runtime import resolve_input_path

    path = resolve_input_path(config["paths"]["test"], config)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    return pd.read_csv(path)


def load_sample_submission(config: dict[str, Any]) -> pd.DataFrame | None:
    from s6e8.runtime import resolve_input_path

    sample_path = config["paths"].get("sample_submission")
    if not sample_path:
        return None
    path = resolve_input_path(sample_path, config)
    if not path.exists():
        return None
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    target = config["competition"]["target"]
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' missing from train frame")
    y = df[target].astype("int64")
    X = df.drop(columns=[target])
    return X, y
