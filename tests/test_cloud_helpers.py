from __future__ import annotations

import json
from pathlib import Path

from scripts_loader import load_script


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
