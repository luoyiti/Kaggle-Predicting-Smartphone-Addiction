"""Error-band helpers. Synthetic frames only."""

import numpy as np
import pandas as pd
import pytest

from s6e8.error_band import analyze_error_band, band_mask
from scripts_loader import load_script


def test_band_mask_is_open_interval():
    pred = np.array([0.3, 0.30001, 0.5, 0.69999, 0.7])
    mask = band_mask(pred, 0.3, 0.7)
    np.testing.assert_array_equal(mask, [False, True, True, True, False])


def test_analyze_error_band_reports_hard_slice_and_feature_auc():
    oof = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "addicted_label": [0, 1, 0, 1, 0, 1],
            "pred": [0.1, 0.2, 0.4, 0.6, 0.85, 0.95],
        }
    )
    features = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "age": [10, 20, 11, 21, 12, 22],
            "notifications_per_day": [1, 2, 1, 2, 1, 2],
        }
    )
    report = analyze_error_band(
        oof,
        features=features,
        feature_columns=["age", "notifications_per_day"],
    )
    assert report["band"]["n"] == 2
    assert report["band"]["pos_rate"] == 0.5
    assert report["auc_all"] is not None
    assert "age" in report["feature_auc_in_band"]
    assert "notifications_per_day" in report["residual_corr_in_band"]


def test_compare_oof_is_scored_on_primary_band():
    oof = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "addicted_label": [0, 1, 0, 1],
            "pred": [0.4, 0.6, 0.1, 0.9],
        }
    )
    compare = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "addicted_label": [0, 1, 0, 1],
            "pred": [0.05, 0.95, 0.2, 0.8],
        }
    )
    report = analyze_error_band(oof, compare=compare, compare_name="partner")
    assert report["compare"]["name"] == "partner"
    assert report["compare"]["auc_on_primary_band"] is not None
    assert report["compare"]["auc_on_primary_band"] >= report["band"]["auc"]


def test_analyze_error_band_cli_writes_json(tmp_path):
    oof_dir = tmp_path / "oof" / "toy"
    oof_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "addicted_label": [0, 1, 0, 1],
            "pred": [0.4, 0.6, 0.05, 0.99],
        }
    ).to_parquet(oof_dir / "oof.parquet", index=False)
    script = load_script("analyze_error_band.py")
    out = tmp_path / "band.json"
    script.sys.argv = [
        "analyze_error_band.py",
        "--experiment",
        "toy",
        "--oof-dir",
        str(tmp_path / "oof"),
        "--out",
        str(out),
        "--lo",
        "0.3",
        "--hi",
        "0.7",
    ]
    script.main()
    import json

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["experiment"] == "toy"
    assert report["band"]["n"] == 2


def test_missing_oof_explains_download(tmp_path):
    script = load_script("analyze_error_band.py")
    with pytest.raises(FileNotFoundError, match="Download a kernel OOF"):
        script._load_oof(tmp_path, "lgbm_nocat")
