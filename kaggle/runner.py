#!/usr/bin/env python3
"""Kaggle Kernel entrypoint. Does not reimplement training.

Resolves the experiment config, materializes this repo if needed, then runs:

    python scripts/train.py --config <experiment.yaml>

Source resolution order:
1. Already present (full GitHub checkout, or a staged kernel directory)
2. SOURCE_ARCHIVE_B64 injected by scripts/prepare_kaggle_kernel.py
3. git clone of S6E8_GIT_REPO at S6E8_GIT_COMMIT (requires internet)
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

# Populated by scripts/prepare_kaggle_kernel.py when bundling a kernel push.
RUN_CONTEXT: dict[str, Any] | None = None
SOURCE_ARCHIVE_B64: str | None = None

KAGGLE_WORKING = Path("/kaggle/working")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle runner for S6E8 training")
    parser.add_argument("--config", default=None, help="Experiment YAML path")
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Kaggle working dirs are ephemeral; overwrite is safe and default",
    )
    # Kaggle often wraps a script as a notebook and appends Jupyter args such as
    # `-f /root/.local/share/jupyter/runtime/kernel-*.json`.
    args, _unknown = parser.parse_known_args(argv)
    return args


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def resolve_context(args: argparse.Namespace) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    if isinstance(RUN_CONTEXT, dict):
        ctx.update(RUN_CONTEXT)
    ctx.update(_load_json(Path("run_context.json")))
    ctx.update(_load_json(Path("/kaggle/working/run_context.json")))

    if args.config:
        ctx["config"] = args.config
    elif os.environ.get("S6E8_CONFIG"):
        ctx["config"] = os.environ["S6E8_CONFIG"]
    ctx.setdefault("config", "configs/baseline.yaml")

    if args.accelerator:
        ctx["accelerator"] = args.accelerator
    elif os.environ.get("S6E8_ACCELERATOR"):
        ctx["accelerator"] = os.environ["S6E8_ACCELERATOR"]
    ctx.setdefault("accelerator", "cpu")

    if os.environ.get("S6E8_GIT_COMMIT"):
        ctx["git_commit"] = os.environ["S6E8_GIT_COMMIT"]
    if os.environ.get("S6E8_GIT_REPO"):
        ctx["git_repo"] = os.environ["S6E8_GIT_REPO"]
    return ctx


def looks_like_repo(root: Path) -> bool:
    return (root / "s6e8" / "__init__.py").is_file() and (
        root / "scripts" / "train.py"
    ).is_file()


def _script_dir() -> Path | None:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return None


def find_existing_repo() -> Path | None:
    cwd = Path.cwd()
    if looks_like_repo(cwd):
        return cwd.resolve()
    here = _script_dir()
    if here is not None:
        if looks_like_repo(here):
            return here
        if here.name == "kaggle" and looks_like_repo(here.parent):
            return here.parent
    if KAGGLE_WORKING.exists() and looks_like_repo(KAGGLE_WORKING):
        return KAGGLE_WORKING
    return None


def _extract_tar(source: Path | io.BytesIO, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=source if not isinstance(source, Path) else None, name=str(source) if isinstance(source, Path) else None, mode="r:gz") as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:
            tar.extractall(dest)


def extract_archive(dest: Path, blob_b64: str) -> Path:
    _extract_tar(io.BytesIO(base64.b64decode(blob_b64)), dest)
    if not looks_like_repo(dest):
        raise RuntimeError(f"Extracted archive is not a valid repo at {dest}")
    return dest


def clone_repo(dest: Path, repo: str, sha: str | None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repo, str(dest)], check=True)
    if sha:
        subprocess.run(["git", "fetch", "--depth", "1", "origin", sha], cwd=dest)
        checkout = subprocess.run(["git", "checkout", sha], cwd=dest)
        if checkout.returncode != 0:
            raise RuntimeError(
                f"Failed to checkout {sha} from {repo}. "
                "Enable internet on the kernel, or rely on the bundled source archive."
            )
    if not looks_like_repo(dest):
        raise RuntimeError(f"Cloned repo is missing training code: {dest}")
    return dest


def ensure_repo(ctx: dict[str, Any]) -> Path:
    existing = find_existing_repo()
    if existing is not None:
        return existing

    dest = KAGGLE_WORKING / "src" if KAGGLE_WORKING.exists() else Path.cwd() / "src"
    archive = SOURCE_ARCHIVE_B64 or os.environ.get("S6E8_SOURCE_ARCHIVE_B64")
    tar_path = Path("source.tar.gz")
    if tar_path.is_file() and not archive:
        _extract_tar(tar_path, dest)
        if looks_like_repo(dest):
            return dest

    if archive:
        return extract_archive(dest, archive)

    repo = ctx.get("git_repo")
    if not repo:
        raise FileNotFoundError(
            "Training sources not found. The Kaggle API only uploads code_file; "
            "prepare_kaggle_kernel.py must bundle source.tar.gz / SOURCE_ARCHIVE_B64, "
            "or set git_repo so the runner can clone GitHub."
        )
    return clone_repo(dest, str(repo), ctx.get("git_commit"))


CORE_MODULES = ("lightgbm", "pandas", "pyarrow", "sklearn", "yaml")


def _pip_install(arguments: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *arguments]
    print("Installing requirements:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_core_requirements(root: Path) -> None:
    """Install the repository requirements only when a core dependency is absent."""
    req = root / "requirements.txt"
    if not req.is_file():
        return
    if all(importlib.util.find_spec(module) is not None for module in CORE_MODULES):
        return
    _pip_install(["-r", str(req)])


def optional_requirements(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Return backend-only packages absent from the baseline requirements file."""
    backend = str(config["model"]["name"]).lower()
    if backend in {"catboost", "cat", "cb"}:
        return [("catboost", "catboost>=1.2.8,<2")]
    return []


def load_selected_config(root: Path, ctx: dict[str, Any]) -> dict[str, Any]:
    """Load the selected config only after PyYAML is guaranteed available."""
    import yaml

    config_path = Path(str(ctx["config"]))
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must contain a mapping: {config_path}")
    return config


def maybe_install_requirements(root: Path, ctx: dict[str, Any]) -> None:
    """Ensure core packages, then add only the selected model backend package."""
    ensure_core_requirements(root)
    config = load_selected_config(root, ctx)
    for module, requirement in optional_requirements(config):
        if importlib.util.find_spec(module) is None:
            _pip_install([requirement])


OUTPUT_DIR_NAMES = ("oof", "submissions", "experiments")


def copy_outputs_to_kaggle_working(root: Path) -> None:
    """Kaggle only publishes /kaggle/working. Training may have written under src/."""
    if not KAGGLE_WORKING.exists():
        return
    try:
        root_resolved = root.resolve()
        working_resolved = KAGGLE_WORKING.resolve()
    except OSError:
        root_resolved = root
        working_resolved = KAGGLE_WORKING
    for name in OUTPUT_DIR_NAMES:
        src = root_resolved / name
        if not src.is_dir():
            continue
        dest = working_resolved / name
        if src == dest:
            continue
        print(f"Copying {src} -> {dest}", flush=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)


def run_train(root: Path, ctx: dict[str, Any], overwrite: bool) -> None:
    env = os.environ.copy()
    if ctx.get("git_commit"):
        env["S6E8_GIT_COMMIT"] = str(ctx["git_commit"])
    env["S6E8_ACCELERATOR"] = str(ctx["accelerator"])
    env["S6E8_CONFIG"] = str(ctx["config"])
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(root / "scripts" / "train.py"),
        "--config",
        str(ctx["config"]),
        "--accelerator",
        str(ctx["accelerator"]),
    ]
    if overwrite:
        cmd.append("--overwrite")
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=root, env=env, check=True)


def debug_kaggle_fs() -> None:
    try:
        print(f"cwd={Path.cwd()}", flush=True)
        print(f"KAGGLE_KERNEL_RUN_TYPE={os.environ.get('KAGGLE_KERNEL_RUN_TYPE')}", flush=True)
        inp = Path("/kaggle/input")
        print(f"/kaggle/input exists={inp.exists()}", flush=True)
        if inp.exists():
            for child in sorted(inp.iterdir()):
                names = sorted(p.name for p in child.iterdir()) if child.is_dir() else []
                print(f"  {child}: {names[:30]}", flush=True)
            trains = sorted(str(p) for p in inp.rglob("train.csv") if p.is_file())
            print(f"train.csv hits={trains[:20]}", flush=True)
    except Exception as exc:
        print(f"debug_kaggle_fs failed: {exc!r}", flush=True)


def main() -> None:
    args = parse_args()
    debug_kaggle_fs()
    ctx = resolve_context(args)
    print(f"kaggle_runner config={ctx['config']} accelerator={ctx['accelerator']}")
    if ctx.get("git_commit"):
        print(f"kaggle_runner git_commit={ctx['git_commit']}")
    root = ensure_repo(ctx)
    print(f"kaggle_runner repo_root={root}")
    maybe_install_requirements(root, ctx)
    run_train(root, ctx, overwrite=args.overwrite)
    copy_outputs_to_kaggle_working(root)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise
