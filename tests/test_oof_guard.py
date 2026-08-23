from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from s6e8.oof_guard import (
    CPU_DROPCATS_EXPERIMENT,
    OofProtocolError,
    as_1d_float,
    is_cpu_dropcats_budget,
    looks_like_pr11_origcats,
    validate_blend_protocol,
    validate_cpu_dropcats_budget_metrics,
    validate_prediction_frames,
    warn_if_budget_v1_not_cpu_dropcats,
)
from scripts_loader import load_script


def _cpu_dropcats_metrics(**overrides):
    metrics = {
        "experiment": CPU_DROPCATS_EXPERIMENT,
        "accelerator": "cpu",
        "n_categorical_features": 9,
        "diagnostic": False,
        "n_splits": 5,
        "n_train": 691369,
        "n_test": 296302,
        "oof_auc": 0.97,
    }
    metrics.update(overrides)
    return metrics


def _pr11_metrics():
    return {
        "experiment": CPU_DROPCATS_EXPERIMENT,
        "accelerator": "gpu",
        "n_categorical_features": 12,
        "diagnostic": False,
        "n_splits": 5,
        "n_train": 691369,
        "n_test": 296302,
        "oof_auc": 0.967681,
    }


def _frames(n_oof=4, n_test=3, pred=None):
    oof_pred = np.linspace(0.1, 0.9, n_oof) if pred is None else pred
    oof = pd.DataFrame(
        {
            "id": np.arange(n_oof),
            "addicted_label": [0, 1, 0, 1][:n_oof],
            "pred": oof_pred,
        }
    )
    test = pd.DataFrame(
        {
            "id": np.arange(100, 100 + n_test),
            "pred": np.linspace(0.2, 0.8, n_test),
        }
    )
    return oof, test


def test_cpu_dropcats_metrics_accept_this_branch_yaml():
    validate_cpu_dropcats_budget_metrics(_cpu_dropcats_metrics())
    assert is_cpu_dropcats_budget(_cpu_dropcats_metrics())


def test_cpu_dropcats_metrics_refuse_pr11_gpu_origcats():
    with pytest.raises(OofProtocolError, match="accelerator='gpu'"):
        validate_cpu_dropcats_budget_metrics(_pr11_metrics())
    assert looks_like_pr11_origcats(_pr11_metrics())
    assert not is_cpu_dropcats_budget(_pr11_metrics())


def test_cpu_dropcats_metrics_refuse_wrong_n_cat():
    with pytest.raises(OofProtocolError, match="n_categorical_features=12"):
        validate_cpu_dropcats_budget_metrics(_cpu_dropcats_metrics(n_categorical_features=12))


def test_warn_budget_v1_when_pr11_dump_present():
    msg = warn_if_budget_v1_not_cpu_dropcats(CPU_DROPCATS_EXPERIMENT, _pr11_metrics())
    assert msg is not None
    assert "Refuse to install/overwrite" in msg
    assert warn_if_budget_v1_not_cpu_dropcats("lgbm_nocat", _pr11_metrics()) is None


def test_as_1d_float_rejects_2d():
    with pytest.raises(ValueError, match="not 1-d"):
        as_1d_float("bad", np.zeros((4, 2)))


def test_validate_prediction_frames_rejects_non_numeric():
    oof, test = _frames()
    oof = oof.copy()
    oof["pred"] = ["0.1", "0.2", "0.3", "0.4"]
    with pytest.raises(ValueError, match="not numeric"):
        validate_prediction_frames("bad", oof, test)


def test_validate_prediction_frames_metrics_n_train():
    oof, test = _frames()
    with pytest.raises(ValueError, match="n_train=691369"):
        validate_prediction_frames(
            "short",
            oof,
            test,
            {"n_train": 691369, "n_test": 3},
        )


def test_as_1d_float_rejects_nan():
    with pytest.raises(ValueError, match="NaN/Inf"):
        as_1d_float("x", np.array([0.1, np.nan]))


def test_blend_protocol_refuses_diagnostic_mix():
    with pytest.raises(ValueError, match="diagnostic"):
        validate_blend_protocol(
            [
                ("full", {"diagnostic": False, "n_train": 691369, "n_splits": 5}),
                ("diag", {"diagnostic": True, "n_train": 691369, "n_splits": 3}),
            ]
        )


def test_blend_protocol_refuses_n_splits_mismatch():
    with pytest.raises(ValueError, match="n_splits"):
        validate_blend_protocol(
            [
                ("a", {"n_splits": 5, "n_train": 691369}),
                ("b", {"n_splits": 3, "n_train": 691369}),
            ]
        )


def test_blend_protocol_allows_missing_n_splits_on_seedavg():
    validate_blend_protocol(
        [
            ("lgbm_nocat_seedavg", {"n_train": 691369}),
            ("cb", {"diagnostic": False, "n_train": 691369, "n_splits": 5}),
        ]
    )


def _write_oof(folder, metrics, n_oof=4, n_test=3, pred=None):
    folder.mkdir(parents=True)
    oof, test = _frames(n_oof=n_oof, n_test=n_test, pred=pred)
    oof.to_parquet(folder / "oof.parquet", index=False)
    test.to_parquet(folder / "test.parquet", index=False)
    np.save(folder / "oof.npy", oof["pred"].to_numpy())
    np.save(folder / "test.npy", test["pred"].to_numpy())
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return oof, test


def test_install_refuses_pr11_gpu_dump_as_budget_v1(tmp_path):
    install = load_script("install_kernel_oof.py")
    src = tmp_path / "kernel"
    incoming = src / "oof" / CPU_DROPCATS_EXPERIMENT
    _write_oof(incoming, _pr11_metrics() | {"n_train": 4, "n_test": 3})
    with pytest.raises(OofProtocolError, match="accelerator='gpu'"):
        install.install_experiment(
            src,
            CPU_DROPCATS_EXPERIMENT,
            tmp_path / "oof",
            preserve_existing_as="catboost_exactcat_budget_pr11_origcats",
            overwrite=False,
        )


def test_install_accepts_cpu_dropcats_and_preserves_pr11(tmp_path, monkeypatch):
    import s6e8.oof_guard as guard

    monkeypatch.setattr(guard, "FULL_N_TRAIN", 4)
    monkeypatch.setattr(guard, "FULL_N_TEST", 3)
    install = load_script("install_kernel_oof.py")
    src = tmp_path / "kernel"
    incoming = src / "oof" / CPU_DROPCATS_EXPERIMENT
    _write_oof(incoming, _cpu_dropcats_metrics(n_train=4, n_test=3, n_splits=5))

    dest_root = tmp_path / "oof"
    existing = dest_root / CPU_DROPCATS_EXPERIMENT
    _write_oof(existing, _pr11_metrics() | {"n_train": 4, "n_test": 3})

    dest = install.install_experiment(
        src,
        CPU_DROPCATS_EXPERIMENT,
        dest_root,
        preserve_existing_as="catboost_exactcat_budget_pr11_origcats",
        overwrite=False,
    )
    installed = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
    assert installed["accelerator"] == "cpu"
    assert installed["n_categorical_features"] == 9
    alias = dest_root / "catboost_exactcat_budget_pr11_origcats" / "metrics.json"
    assert alias.is_file()
    preserved = json.loads(alias.read_text(encoding="utf-8"))
    assert preserved["accelerator"] == "gpu"
    assert preserved["n_categorical_features"] == 12


def test_install_refuses_overwrite_existing_cpu_dropcats(tmp_path, monkeypatch):
    import s6e8.oof_guard as guard

    monkeypatch.setattr(guard, "FULL_N_TRAIN", 4)
    monkeypatch.setattr(guard, "FULL_N_TEST", 3)
    install = load_script("install_kernel_oof.py")
    src = tmp_path / "kernel"
    incoming = src / "oof" / CPU_DROPCATS_EXPERIMENT
    metrics = _cpu_dropcats_metrics(n_train=4, n_test=3)
    _write_oof(incoming, metrics)
    dest_root = tmp_path / "oof"
    _write_oof(dest_root / CPU_DROPCATS_EXPERIMENT, metrics)
    with pytest.raises(OofProtocolError, match="already looks like"):
        install.install_experiment(
            src,
            CPU_DROPCATS_EXPERIMENT,
            dest_root,
            preserve_existing_as="alias",
            overwrite=False,
        )


def test_blend_oof_fails_loud_on_npy_shape_mismatch(tmp_path):
    blend = load_script("blend_oof.py")
    folder = tmp_path / "broken"
    oof, _test = _write_oof(
        folder,
        {"oof_auc": 0.5, "n_train": 4, "n_test": 3, "diagnostic": False},
    )
    np.save(folder / "oof.npy", np.zeros(2))
    with pytest.raises(ValueError, match="oof.npy shape"):
        blend._load("broken", tmp_path)
