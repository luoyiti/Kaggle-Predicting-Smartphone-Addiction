from __future__ import annotations

import json
import subprocess
import sys


def test_prepare_kernel_staging(tmp_path, repo_root):
    staging = tmp_path / "kernel"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_kaggle_kernel.py"),
            "--config",
            "configs/baseline.yaml",
            "--accelerator",
            "cpu",
            "--username",
            "testuser",
            "--slug",
            "s6e8-cloud-train",
            "--out",
            str(staging),
            "--git-commit",
            "deadbeef",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert "kernel_id=testuser/s6e8-cloud-train" in proc.stdout
    meta = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["enable_gpu"] is False
    assert meta["competition_sources"] == ["playground-series-s6e8"]
    assert meta["code_file"] == "runner.py"
    runner = (staging / "runner.py").read_text(encoding="utf-8")
    assert "configs/baseline.yaml" in runner
    assert "SOURCE_ARCHIVE_B64" in runner
    assert (staging / "source.tar.gz").stat().st_size > 0
    assert max(len(line) for line in runner.splitlines()) < 200
    compile(runner, str(staging / "runner.py"), "exec")


def test_prepare_kernel_gpu_metadata(tmp_path, repo_root):
    staging = tmp_path / "kernel-gpu"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_kaggle_kernel.py"),
            "--config",
            "configs/baseline.yaml",
            "--accelerator",
            "gpu",
            "--username",
            "testuser",
            "--out",
            str(staging),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    meta = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["enable_gpu"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"


def test_kernel_metadata_template_is_safe(repo_root):
    text = (repo_root / "kaggle" / "kernel-metadata.json").read_text(encoding="utf-8")
    assert "YOUR_KAGGLE_USERNAME" in text
    assert "playground-series-s6e8" in text


def test_runner_ignores_jupyter_kernel_args(repo_root):
    import importlib.util

    path = repo_root / "kaggle" / "runner.py"
    spec = importlib.util.spec_from_file_location("kaggle_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(
        [
            "--config",
            "configs/baseline.yaml",
            "-f",
            "/root/.local/share/jupyter/runtime/kernel-123.json",
        ]
    )
    assert args.config == "configs/baseline.yaml"
    assert args.accelerator is None
