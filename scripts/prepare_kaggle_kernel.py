#!/usr/bin/env python3
"""Assemble a Kaggle Kernel directory for `kaggle kernels push`.

The Kaggle API uploads only `code_file`, so this script injects a gzip+base64
source archive into runner.py along with the experiment run context.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import tarfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.data import load_config
from s6e8.runtime import apply_runtime_override, get_git_commit, normalize_accelerator

DEFAULT_SLUG = "s6e8-cloud-train"
DEFAULT_TITLE = "S6E8 Cloud Train"
GPU_MACHINE_SHAPE = "NvidiaTeslaT4"
BUNDLE_PATHS = (
    "s6e8",
    "scripts/train.py",
    "configs",
    "requirements.txt",
)
SKIP_PARTS = {"__pycache__", ".git", ".venv", "venv"}


def _exclude_junk(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(tarinfo.name).parts
    if any(part in SKIP_PARTS for part in parts):
        return None
    if tarinfo.name.endswith(".pyc"):
        return None
    return tarinfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Kaggle kernel staging directory")
    parser.add_argument("--config", required=True, help="Experiment YAML, e.g. configs/baseline.yaml")
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--username", default=os.environ.get("KAGGLE_USERNAME", ""))
    parser.add_argument("--slug", default=os.environ.get("KAGGLE_KERNEL_SLUG", DEFAULT_SLUG))
    parser.add_argument(
        "--title",
        default=None,
        help="Kernel title; defaults to a title whose slug matches --slug",
    )
    parser.add_argument("--git-repo", default=os.environ.get("S6E8_GIT_REPO", ""))
    parser.add_argument("--git-commit", default=os.environ.get("S6E8_GIT_COMMIT") or get_git_commit())
    parser.add_argument("--out", default=".kernel-staging")
    parser.add_argument(
        "--gpu-machine-shape",
        default=GPU_MACHINE_SHAPE,
        help="Kaggle machine_shape when accelerator=gpu",
    )
    return parser.parse_args()


def _git_repo(explicit: str) -> str:
    if explicit:
        return explicit
    gh = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if gh:
        return f"https://github.com/{gh}.git"
    return "https://github.com/luoyiti/Kaggle-Predicting-Smartphone-Addiction.git"


def default_title_for_slug(slug: str) -> str:
    """Return a readable title that Kaggle resolves back to the same slug."""
    return " ".join(part.upper() if part == "s6e8" else part.capitalize() for part in slug.split("-"))


def build_source_archive(root: Path) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for rel in BUNDLE_PATHS:
            path = root / rel
            if not path.exists():
                raise FileNotFoundError(f"Cannot bundle missing path: {path}")
            tar.add(path, arcname=rel, filter=_exclude_junk)
    return buffer.getvalue()


def format_b64_literal(blob: str, width: int = 76) -> str:
    """Emit a concatenated string literal so Kaggle notebook wrapping does not get a megabyte line."""
    chunks = [blob[i : i + width] for i in range(0, len(blob), width)] or [""]
    inner = "\n    ".join(json.dumps(chunk) for chunk in chunks)
    return f"(\n    {inner}\n)"


def write_runner(staging: Path, context: dict, archive: bytes) -> None:
    template = (ROOT / "kaggle" / "runner.py").read_text(encoding="utf-8")
    marker = (
        "RUN_CONTEXT: dict[str, Any] | None = None\n"
        "SOURCE_ARCHIVE_B64: str | None = None\n"
    )
    if marker not in template:
        raise RuntimeError("kaggle/runner.py is missing RUN_CONTEXT markers")
    generated = (
        f"RUN_CONTEXT: dict[str, Any] | None = {json.dumps(context, indent=2)}\n"
        f"SOURCE_ARCHIVE_B64: str | None = {format_b64_literal(base64.b64encode(archive).decode('ascii'))}\n"
    )
    (staging / "runner.py").write_text(template.replace(marker, generated, 1), encoding="utf-8")


def write_metadata(
    staging: Path,
    *,
    username: str,
    slug: str,
    title: str,
    accelerator: str,
    enable_internet: bool,
    competition_slug: str,
    gpu_machine_shape: str,
) -> dict:
    if not username:
        raise ValueError(
            "Kaggle username is required to set kernel id. "
            "Pass --username or set KAGGLE_USERNAME."
        )
    enable_gpu = accelerator == "gpu"
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": bool(enable_internet),
        "machine_shape": gpu_machine_shape if enable_gpu else "",
        "dataset_sources": [],
        "competition_sources": [competition_slug],
        "kernel_sources": [],
        "model_sources": [],
    }
    (staging / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_runtime_override(config, accelerator=args.accelerator)
    accelerator = normalize_accelerator(config["runtime"]["accelerator"])
    enable_internet = bool((config.get("runtime") or {}).get("enable_internet", True))
    competition_slug = config["competition"]["slug"]

    rel_config = args.config
    config_path = Path(args.config)
    try:
        rel_config = str(config_path.resolve().relative_to(ROOT))
    except ValueError:
        pass

    context = {
        "config": rel_config,
        "accelerator": accelerator,
        "git_commit": args.git_commit,
        "git_repo": _git_repo(args.git_repo),
        "experiment": config["experiment"]["name"],
        "competition": competition_slug,
    }

    staging = Path(args.out)
    if not staging.is_absolute():
        staging = ROOT / staging
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    archive = build_source_archive(ROOT)
    (staging / "source.tar.gz").write_bytes(archive)
    (staging / "run_context.json").write_text(
        json.dumps(context, indent=2) + "\n", encoding="utf-8"
    )
    write_runner(staging, context, archive)
    metadata = write_metadata(
        staging,
        username=args.username.strip(),
        slug=args.slug.strip(),
        title=args.title or default_title_for_slug(args.slug.strip()),
        accelerator=accelerator,
        enable_internet=enable_internet,
        competition_slug=competition_slug,
        gpu_machine_shape=args.gpu_machine_shape,
    )

    print(f"staging={staging}")
    print(f"kernel_id={metadata['id']}")
    print(f"accelerator={accelerator}")
    print(f"enable_gpu={metadata['enable_gpu']}")
    print(f"config={rel_config}")
    print(f"archive_bytes={len(archive)}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as fh:
            fh.write(f"kernel_id={metadata['id']}\n")
            fh.write(f"accelerator={accelerator}\n")
            fh.write(f"experiment={context['experiment']}\n")
            fh.write(f"config={rel_config}\n")
            fh.write(f"staging={staging}\n")


if __name__ == "__main__":
    main()
