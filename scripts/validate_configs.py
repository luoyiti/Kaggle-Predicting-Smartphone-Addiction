#!/usr/bin/env python3
"""Validate every experiment YAML under configs/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.data import load_config


def main() -> None:
    configs = sorted((ROOT / "configs").glob("*.yaml"))
    if not configs:
        raise SystemExit("No YAML files found in configs/")
    for path in configs:
        config = load_config(path)
        print(
            f"ok {path.relative_to(ROOT)} "
            f"experiment={config['experiment']['name']} "
            f"accelerator={config['runtime']['accelerator']}"
        )


if __name__ == "__main__":
    main()
