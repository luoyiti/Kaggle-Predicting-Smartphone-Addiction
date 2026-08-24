#!/usr/bin/env python3
"""Validate every experiment YAML under configs/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.data import load_config


def _validate_train_config(path: Path) -> None:
    config = load_config(path)
    print(
        f"ok {path.relative_to(ROOT)} "
        f"experiment={config['experiment']['name']} "
        f"accelerator={config['runtime']['accelerator']}"
    )


def _validate_postprocess_config(path: Path) -> None:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping")
    kind = raw.get("kind")
    name = (raw.get("experiment") or {}).get("name")
    if not kind:
        raise ValueError(f"{path} missing kind (ensemble|calibration|promotion)")
    if not name:
        raise ValueError(f"{path} missing experiment.name")
    print(f"ok {path.relative_to(ROOT)} kind={kind} experiment={name}")


def main() -> None:
    configs = sorted((ROOT / "configs").glob("*.yaml"))
    if not configs:
        raise SystemExit("No YAML files found in configs/")
    for path in configs:
        _validate_train_config(path)
    post = sorted((ROOT / "configs" / "ensemble").glob("*.yaml"))
    for path in post:
        _validate_postprocess_config(path)


if __name__ == "__main__":
    main()
