"""Environment detection, data paths, accelerator, and experiment metadata."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from s6e8 import __version__ as package_version

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

VALID_ACCELERATORS = ("cpu", "gpu")
FORMAL_PROTOCOL = "fixed5_seed42_v1"

_CORE_DISTRIBUTIONS = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "scikit-learn"),
    ("pyarrow", "pyarrow"),
    ("lightgbm", "lightgbm"),
)
_BACKEND_DISTRIBUTIONS = {
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "lgb": "lightgbm",
    "catboost": "catboost",
    "cat": "catboost",
    "cb": "catboost",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "histgb": "scikit-learn",
    "histgradientboosting": "scikit-learn",
    "sklearn_histgb": "scikit-learn",
    "logreg": "scikit-learn",
    "logistic": "scikit-learn",
    "logisticregression": "scikit-learn",
}


def _installed_distribution_version(distribution: str) -> str | None:
    """Read installed package metadata without importing optional backends."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_versions(model_name: str) -> dict[str, str | None]:
    """Return deterministic core and selected-backend dependency provenance."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for key, distribution in _CORE_DISTRIBUTIONS:
        versions[key] = _installed_distribution_version(distribution)
    versions["s6e8"] = _installed_distribution_version("s6e8") or package_version

    backend = str(model_name).strip().lower()
    distribution = _BACKEND_DISTRIBUTIONS.get(backend, backend)
    if distribution and distribution not in versions:
        versions[distribution] = _installed_distribution_version(distribution)
    return versions

REQUIRED_CONFIG_KEYS = (
    ("experiment", "name"),
    ("experiment", "seed"),
    ("experiment", "data_version"),
    ("experiment", "feature_version"),
    ("experiment", "model_version"),
    ("competition", "slug"),
    ("competition", "target"),
    ("competition", "id_col"),
    ("paths", "train"),
    ("paths", "test"),
    ("paths", "oof_dir"),
    ("paths", "submission_dir"),
    ("cv", "n_splits"),
    ("features", "numeric"),
    ("features", "categorical"),
    ("model", "name"),
)


def _nested_get(config: dict[str, Any], *keys: str) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(".".join(keys))
        cur = cur[key]
    return cur


def _validate_formal_protocol(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    if not bool(experiment.get("formal", False)):
        return
    protocol = experiment.get("validation_protocol")
    expected = {
        "protocol": FORMAL_PROTOCOL,
        "seed": 42,
        "type": "stratified",
        "n_splits": 5,
        "shuffle": True,
    }
    actual = {
        "protocol": protocol,
        "seed": int(experiment["seed"]),
        "type": str(config["cv"].get("type", "")),
        "n_splits": int(config["cv"]["n_splits"]),
        "shuffle": bool(config["cv"].get("shuffle", False)),
    }
    if actual != expected:
        raise ValueError(f"Formal protocol {FORMAL_PROTOCOL} required; got {actual}")

    output = config.get("output") or {}
    if output.get("save_oof") is not True or output.get("save_test") is not True:
        raise ValueError(
            "Formal output requirements: output.save_oof and output.save_test "
            "must both be true"
        )


def validate_config(config: dict[str, Any]) -> None:
    missing: list[str] = []
    for keys in REQUIRED_CONFIG_KEYS:
        try:
            _nested_get(config, *keys)
        except KeyError:
            missing.append(".".join(keys))
    if missing:
        raise ValueError("Config missing required keys: " + ", ".join(missing))

    name = str(config["experiment"]["name"]).strip()
    if not name:
        raise ValueError("experiment.name must be a non-empty string")
    if "/" in name or name in {".", ".."}:
        raise ValueError(f"Unsafe experiment.name: {name!r}")

    accelerator = get_accelerator(config)
    if accelerator not in VALID_ACCELERATORS:
        raise ValueError(
            f"runtime.accelerator must be one of {VALID_ACCELERATORS}, got {accelerator!r}"
        )

    _validate_formal_protocol(config)


def detect_environment() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").exists():
        return "kaggle"
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "github_actions"
    return "local"


def is_kaggle() -> bool:
    return detect_environment() == "kaggle"


def normalize_accelerator(value: str) -> str:
    accel = str(value).strip().lower()
    if accel not in VALID_ACCELERATORS:
        raise ValueError(
            f"accelerator must be one of {VALID_ACCELERATORS}, got {value!r}"
        )
    return accel


def get_accelerator(config: dict[str, Any], override: str | None = None) -> str:
    if override:
        return normalize_accelerator(override)
    env = os.environ.get("S6E8_ACCELERATOR", "").strip()
    if env:
        return normalize_accelerator(env)
    runtime = config.get("runtime") or {}
    return normalize_accelerator(str(runtime.get("accelerator", "cpu")))


def apply_runtime_override(config: dict[str, Any], accelerator: str | None = None) -> dict[str, Any]:
    """Return config with resolved runtime.accelerator (env / CLI wins over YAML)."""
    runtime = dict(config.get("runtime") or {})
    runtime["accelerator"] = get_accelerator(config, override=accelerator)
    config["runtime"] = runtime
    return config


def discover_kaggle_input_root(input_root: Path, slug: str) -> Path:
    """Find the mounted competition folder. Prefer the official slug, else train.csv.

    Current Kaggle Batch kernels mount competitions at
    ``/kaggle/input/competitions/<slug>``, not ``/kaggle/input/<slug>``.
    """
    candidates = (
        input_root / slug,
        input_root / "competitions" / slug,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if input_root.is_dir():
        hits = [p.parent for p in input_root.rglob("train.csv") if p.is_file()]
        hits.sort(key=lambda path: (0 if path.name == slug else 1, str(path)))
        if hits:
            return hits[0]
    return candidates[0]


def kaggle_input_root(config: dict[str, Any]) -> Path:
    explicit = (config.get("paths") or {}).get("kaggle_input")
    if explicit:
        return Path(explicit)
    slug = config["competition"]["slug"]
    return discover_kaggle_input_root(Path("/kaggle/input"), slug)


def resolve_input_path(path_like: str | Path, config: dict[str, Any]) -> Path:
    """Resolve train/test/sample paths for local disk or mounted Kaggle data."""
    configured = Path(path_like)
    local = configured if configured.is_absolute() else _resolve_path(configured)

    if is_kaggle():
        candidate = kaggle_input_root(config) / configured.name
        if candidate.exists():
            return candidate
        if local.exists():
            return local
        return candidate

    return local


def get_git_commit() -> str | None:
    for key in ("S6E8_GIT_COMMIT", "GITHUB_SHA"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return output or None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def apply_model_device(
    params: dict[str, Any],
    model_name: str,
    accelerator: str,
) -> dict[str, Any]:
    """Copy model params and set backend device keys when the YAML did not.

    Unknown backends are left unchanged so future trainers can opt in.
    """
    out = dict(params)
    name = model_name.lower()
    accel = normalize_accelerator(accelerator)

    if name in {"lightgbm", "lgbm", "lgb"}:
        if accel == "gpu":
            if "device_type" not in out and "device" not in out:
                out["device_type"] = "gpu"
        return out

    if name in {"xgboost", "xgb"}:
        if accel == "gpu":
            out.setdefault("tree_method", "hist")
            out.setdefault("device", "cuda")
        else:
            out.setdefault("tree_method", "hist")
            out.setdefault("device", "cpu")
        return out

    if name in {"catboost", "cb"}:
        out.setdefault("task_type", "GPU" if accel == "gpu" else "CPU")
        return out

    return out


def experiment_summary(
    config: dict[str, Any],
    *,
    cv_auc: float | None = None,
    runtime_seconds: float | None = None,
    git_commit: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "experiment": config["experiment"]["name"],
        "git_commit": git_commit,
        "config": config.get("_config_path"),
        "seed": config["experiment"]["seed"],
        "data_version": config["experiment"]["data_version"],
        "feature_version": config["experiment"]["feature_version"],
        "model_version": config["experiment"]["model_version"],
        "accelerator": get_accelerator(config),
        "cv_auc": cv_auc,
        "runtime_seconds": runtime_seconds,
        "timestamp": utc_timestamp(),
        "environment": detect_environment(),
    }
    exp = config.get("experiment") or {}
    for key in ("hypothesis", "change", "diagnostic", "source_formal"):
        if key in exp and exp[key] is not None:
            payload[key] = exp[key]
    if extra:
        payload.update(extra)
    return payload
