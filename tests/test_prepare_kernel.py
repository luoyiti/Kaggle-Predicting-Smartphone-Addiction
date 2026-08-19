from __future__ import annotations

import json
import importlib.util
import re
import subprocess
import sys

import pytest


def load_prepare_kernel(repo_root):
    path = repo_root / "scripts" / "prepare_kaggle_kernel.py"
    spec = importlib.util.spec_from_file_location("prepare_kaggle_kernel_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_prepare_kernel_attaches_enabled_external_reference_dataset(tmp_path, repo_root):
    staging = tmp_path / "kernel-reference"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_kaggle_kernel.py"),
            "--config",
            "configs/catboost_exactcat_budget_refdist_v1.yaml",
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
    assert proc.returncode == 0
    meta = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == [
        "jayjoshi37/smartphone-usage-and-addiction-prediction"
    ]
    assert meta["competition_sources"] == ["playground-series-s6e8"]


def test_configured_dataset_sources_accepts_only_enabled_owner_dataset_slugs(repo_root):
    prepare = load_prepare_kernel(repo_root)
    assert prepare.configured_dataset_sources({}) == []
    assert prepare.configured_dataset_sources(
        {"external_reference": {"enabled": False, "dataset_source": "owner/dataset"}}
    ) == []
    with pytest.raises(ValueError, match="owner/dataset"):
        prepare.configured_dataset_sources(
            {"external_reference": {"enabled": True, "dataset_source": "not-a-slug"}}
        )


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


def test_custom_kernel_slug_gets_a_matching_default_title(tmp_path, repo_root):
    """Catch Kaggle 409s caused by a title that resolves to a different id."""
    staging = tmp_path / "kernel-custom-slug"
    slug = "s6e8-histgb-nocat-long-v1"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_kaggle_kernel.py"),
            "--config",
            "configs/histgb_nocat_long_v1.yaml",
            "--accelerator",
            "cpu",
            "--username",
            "testuser",
            "--slug",
            slug,
            "--out",
            str(staging),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    meta = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    title_slug = re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-")
    assert meta["id"] == f"testuser/{slug}"
    assert title_slug == slug


def test_explicit_kernel_title_is_preserved(tmp_path, repo_root):
    staging = tmp_path / "kernel-explicit-title"
    title = "Explicit Human Title"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_kaggle_kernel.py"),
            "--config",
            "configs/baseline.yaml",
            "--accelerator",
            "cpu",
            "--username",
            "testuser",
            "--title",
            title,
            "--out",
            str(staging),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    meta = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["title"] == title


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


def test_copy_outputs_to_kaggle_working(tmp_path, repo_root, monkeypatch):
    import importlib.util

    path = repo_root / "kaggle" / "runner.py"
    spec = importlib.util.spec_from_file_location("kaggle_runner_copy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    working = tmp_path / "working"
    working.mkdir()
    src_root = tmp_path / "src"
    (src_root / "oof" / "lgbm_nocat").mkdir(parents=True)
    (src_root / "oof" / "lgbm_nocat" / "metrics.json").write_text("{}", encoding="utf-8")
    (src_root / "submissions").mkdir()
    (src_root / "submissions" / "lgbm_nocat.csv").write_text("id,pred\n1,0.2\n", encoding="utf-8")
    (src_root / "experiments").mkdir()
    (src_root / "experiments" / "lgbm_nocat.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(module, "KAGGLE_WORKING", working)
    module.copy_outputs_to_kaggle_working(src_root)
    assert (working / "oof" / "lgbm_nocat" / "metrics.json").is_file()
    assert (working / "submissions" / "lgbm_nocat.csv").is_file()
    assert (working / "experiments" / "lgbm_nocat.json").is_file()
