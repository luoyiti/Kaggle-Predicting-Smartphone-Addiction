from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from scripts_loader import load_script


def load_runner(repo_root: Path):
    path = repo_root / "kaggle" / "runner.py"
    spec = importlib.util.spec_from_file_location("kaggle_runner_cloud_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optional_requirement_for_catboost(repo_root):
    runner = load_runner(repo_root)
    assert runner.optional_requirements({"model": {"name": "catboost"}}) == [
        ("catboost", "catboost>=1.2.8,<2")
    ]


def test_optional_requirement_for_lookup_transformer_aliases(repo_root):
    runner = load_runner(repo_root)
    expected = [("torch", "torch>=2.2,<3")]

    assert runner.optional_requirements(
        {"model": {"name": "lookup_transformer"}}
    ) == expected
    assert runner.optional_requirements({"model": {"name": "lookup"}}) == expected


def test_optional_requirement_for_lightgbm_is_empty(repo_root):
    runner = load_runner(repo_root)
    assert runner.optional_requirements({"model": {"name": "lightgbm"}}) == []


def test_maybe_install_requirements_installs_only_missing_backend_package(tmp_path, repo_root, monkeypatch):
    runner = load_runner(repo_root)
    (tmp_path / "requirements.txt").write_text("pandas>=2.1\n", encoding="utf-8")
    config = tmp_path / "configs" / "catboost.yaml"
    config.parent.mkdir()
    config.write_text("model:\n  name: catboost\n", encoding="utf-8")
    ctx = {"config": "configs/catboost.yaml"}
    installed: list[list[str]] = []

    monkeypatch.setattr(runner, "ensure_core_requirements", lambda root: None)
    monkeypatch.setattr(
        runner.importlib.util,
        "find_spec",
        lambda name: None if name == "catboost" else object(),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda cmd, check=True: installed.append(cmd),
    )

    runner.maybe_install_requirements(tmp_path, ctx)
    assert installed == [
        [
            runner.sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "catboost>=1.2.8,<2",
        ]
    ]


def test_maybe_install_requirements_skips_present_backend_package(tmp_path, repo_root, monkeypatch):
    runner = load_runner(repo_root)
    (tmp_path / "requirements.txt").write_text("pandas>=2.1\n", encoding="utf-8")
    config = tmp_path / "configs" / "catboost.yaml"
    config.parent.mkdir()
    config.write_text("model:\n  name: catboost\n", encoding="utf-8")
    installed: list[list[str]] = []

    monkeypatch.setattr(runner, "ensure_core_requirements", lambda root: None)
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda cmd, check=True: installed.append(cmd),
    )

    runner.maybe_install_requirements(tmp_path, {"config": "configs/catboost.yaml"})
    assert installed == []


def test_selected_lookup_config_installs_only_missing_torch(
    tmp_path, repo_root, monkeypatch
):
    runner = load_runner(repo_root)
    config = tmp_path / "configs" / "lookup.yaml"
    config.parent.mkdir()
    config.write_text("model:\n  name: lookup_transformer\n", encoding="utf-8")
    installed: list[list[str]] = []

    monkeypatch.setattr(runner, "ensure_core_requirements", lambda root: None)
    monkeypatch.setattr(
        runner.importlib.util,
        "find_spec",
        lambda name: None if name == "torch" else object(),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda cmd, check=True: installed.append(cmd),
    )

    runner.maybe_install_requirements(tmp_path, {"config": "configs/lookup.yaml"})

    assert installed == [
        [
            runner.sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch>=2.2,<3",
        ]
    ]


def test_selected_lookup_config_skips_torch_when_import_is_available(
    tmp_path, repo_root, monkeypatch
):
    runner = load_runner(repo_root)
    config = tmp_path / "configs" / "lookup.yaml"
    config.parent.mkdir()
    config.write_text("model:\n  name: lookup\n", encoding="utf-8")
    installed: list[list[str]] = []

    monkeypatch.setattr(runner, "ensure_core_requirements", lambda root: None)
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda cmd, check=True: installed.append(cmd),
    )

    runner.maybe_install_requirements(tmp_path, {"config": "configs/lookup.yaml"})

    assert installed == []


def test_kaggle_workflow_cannot_submit_to_leaderboard(repo_root):
    workflow_text = (repo_root / ".github" / "workflows" / "kaggle-train.yml").read_text(
        encoding="utf-8"
    )
    assert "submit_to_kaggle" not in workflow_text
    assert "kaggle competitions submit" not in workflow_text


def test_legacy_json_token(tmp_path, monkeypatch):
    setup = load_script("setup_kaggle_auth.py")
    monkeypatch.setattr(setup, "KAGGLE_DIR", tmp_path / ".kaggle")
    monkeypatch.setenv(
        "KAGGLE_API_TOKEN",
        json.dumps({"username": "alice", "key": "not-a-real-key"}),
    )
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    resolved = setup.configure()
    assert resolved["username"] == "alice"
    payload = json.loads((tmp_path / ".kaggle" / "kaggle.json").read_text(encoding="utf-8"))
    assert payload["username"] == "alice"


def test_opaque_token_requires_username(tmp_path, monkeypatch):
    setup = load_script("setup_kaggle_auth.py")
    monkeypatch.setattr(setup, "KAGGLE_DIR", tmp_path / ".kaggle")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "opaque-token-value")
    monkeypatch.setenv("KAGGLE_USERNAME", "bob")
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    resolved = setup.configure()
    assert resolved["username"] == "bob"
    assert (tmp_path / ".kaggle" / "access_token").read_text(encoding="utf-8") == "opaque-token-value"


def test_gha_summary_from_metrics(tmp_path):
    summary_script = load_script("write_gha_summary.py")
    out = tmp_path / "output"
    oof = out / "oof" / "baseline"
    oof.mkdir(parents=True)
    (oof / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "baseline",
                "accelerator": "cpu",
                "oof_auc": 0.812345,
                "cv_mean": 0.81,
                "cv_std": 0.01,
                "n_splits": 5,
                "runtime_seconds": 120,
                "seed": 42,
                "git_commit": "abc",
                "data_version": "v1",
                "feature_version": "v1",
                "model_version": "lgbm_v1",
            }
        ),
        encoding="utf-8",
    )
    (oof / "oof.npy").write_bytes(b"fake")
    sub = out / "submissions"
    sub.mkdir()
    (sub / "baseline.csv").write_text("id,pred\n1,0.2\n", encoding="utf-8")
    markdown = summary_script.build_markdown(
        type("A", (), {
            "config": "configs/baseline.yaml",
            "accelerator": "cpu",
            "kernel_id": "user/s6e8-cloud-train",
            "status": "complete",
        })(),
        json.loads((oof / "metrics.json").read_text(encoding="utf-8")),
        sub / "baseline.csv",
        out,
    )
    assert "0.812345" in markdown
    assert "generated" in markdown
    assert Path("configs/baseline.yaml").as_posix() in markdown or "configs/baseline.yaml" in markdown


def test_kernel_status_parser():
    wait = load_script("wait_kaggle_kernel.py")
    assert wait.normalize_status('luoyiti/s6e8-cloud-train has status "complete"') == "complete"
    assert wait.normalize_status('luoyiti/s6e8-cloud-train has status "running"') == "running"
    assert wait.normalize_status('luoyiti/s6e8-cloud-train has status "queued"') == "queued"
    assert wait.normalize_status('luoyiti/s6e8-cloud-train has status "error"') == "error"
    assert wait.normalize_status('x has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert wait.normalize_status('x has status "COMPLETED"') == "complete"
    assert wait.normalize_status("complete") == "complete"


def test_wait_script_finds_log_files(tmp_path):
    wait = load_script("wait_kaggle_kernel.py")
    (tmp_path / "kernel.log").write_text("Traceback (most recent call last):\nboom\n", encoding="utf-8")
    (tmp_path / "oof.npy").write_bytes(b"\x00\x01")
    found = wait.iter_log_files(tmp_path)
    assert [p.name for p in found] == ["kernel.log"]


def test_decode_kaggle_json_log():
    wait = load_script("wait_kaggle_kernel.py")
    raw = json.dumps(
        [
            {"stream_name": "stdout", "time": 1.0, "data": "cwd=/kaggle/working\n"},
            {
                "stream_name": "stderr",
                "time": 2.0,
                "data": "FileNotFoundError: Train file not found\n",
            },
        ]
    )
    decoded = wait.decode_kaggle_log_text(raw)
    assert "cwd=/kaggle/working" in decoded
    assert "FileNotFoundError" in decoded


def test_gha_summary_includes_kernel_log(tmp_path):
    summary_script = load_script("write_gha_summary.py")
    out = tmp_path / "output"
    out.mkdir()
    (out / "kernel.log").write_text("FileNotFoundError: Train file not found\n", encoding="utf-8")
    markdown = summary_script.build_markdown(
        type("A", (), {
            "config": "configs/baseline.yaml",
            "accelerator": "cpu",
            "kernel_id": "user/s6e8-cloud-train",
            "status": "error",
        })(),
        {},
        None,
        out,
    )
    assert "Kernel log excerpt" in markdown
    assert "FileNotFoundError" in markdown
