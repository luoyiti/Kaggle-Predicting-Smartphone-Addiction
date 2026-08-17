from __future__ import annotations

from pathlib import Path

import pytest

from s6e8.runtime import (
    apply_model_device,
    get_accelerator,
    normalize_accelerator,
    resolve_input_path,
)


def test_accelerator_defaults_to_cpu():
    assert get_accelerator({"runtime": {}}) == "cpu"
    assert get_accelerator({}) == "cpu"


def test_accelerator_override_and_env(monkeypatch):
    monkeypatch.setenv("S6E8_ACCELERATOR", "gpu")
    assert get_accelerator({"runtime": {"accelerator": "cpu"}}) == "gpu"
    assert get_accelerator({"runtime": {"accelerator": "cpu"}}, override="cpu") == "cpu"


def test_kaggle_input_root_falls_back_to_any_train_csv(tmp_path):
    from s6e8.runtime import discover_kaggle_input_root

    mounted = tmp_path / "Predicting Smartphone Addiction"
    mounted.mkdir()
    (mounted / "train.csv").write_text("id,x\n1,1\n", encoding="utf-8")
    found = discover_kaggle_input_root(tmp_path, "playground-series-s6e8")
    assert found == mounted


def test_kaggle_input_root_prefers_official_slug(tmp_path):
    from s6e8.runtime import discover_kaggle_input_root

    slug_dir = tmp_path / "playground-series-s6e8"
    slug_dir.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (other / "train.csv").write_text("id,x\n1,1\n", encoding="utf-8")
    found = discover_kaggle_input_root(tmp_path, "playground-series-s6e8")
    assert found == slug_dir


def test_kaggle_input_root_uses_competitions_subdir(tmp_path):
    from s6e8.runtime import discover_kaggle_input_root

    mounted = tmp_path / "competitions" / "playground-series-s6e8"
    mounted.mkdir(parents=True)
    (mounted / "train.csv").write_text("id,x\n1,1\n", encoding="utf-8")
    (mounted / "test.csv").write_text("id,x\n2,2\n", encoding="utf-8")
    found = discover_kaggle_input_root(tmp_path, "playground-series-s6e8")
    assert found == mounted


def test_invalid_accelerator():
    with pytest.raises(ValueError):
        normalize_accelerator("tpu")


def test_lightgbm_gpu_device_is_opt_in():
    cpu_params = apply_model_device({"learning_rate": 0.05}, "lightgbm", "cpu")
    assert "device_type" not in cpu_params
    gpu_params = apply_model_device({"learning_rate": 0.05}, "lightgbm", "gpu")
    assert gpu_params["device_type"] == "gpu"


def test_kaggle_input_path_uses_mounted_competition(tmp_path, monkeypatch):
    import s6e8.runtime as runtime

    kaggle_root = tmp_path / "playground-series-s6e8"
    kaggle_root.mkdir()
    (kaggle_root / "train.csv").write_text("id,x\n1,1\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "is_kaggle", lambda: True)
    monkeypatch.setattr(runtime, "kaggle_input_root", lambda config: kaggle_root)

    config = {"competition": {"slug": "playground-series-s6e8"}, "paths": {}}
    resolved = resolve_input_path("data/raw/train.csv", config)
    assert resolved == kaggle_root / "train.csv"


def test_local_input_path(tmp_path, monkeypatch):
    import s6e8.runtime as runtime

    monkeypatch.setattr(runtime, "is_kaggle", lambda: False)
    local = tmp_path / "train.csv"
    local.write_text("id,x\n1,1\n", encoding="utf-8")
    resolved = resolve_input_path(local, {"competition": {"slug": "playground-series-s6e8"}})
    assert resolved == local
    assert Path(resolved).exists()
