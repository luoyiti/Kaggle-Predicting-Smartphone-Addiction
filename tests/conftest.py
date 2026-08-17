"""Shared test fixtures. Synthetic frames are test data, not competition data."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def baseline_config_path() -> Path:
    return ROOT / "configs" / "baseline.yaml"
