from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from s6e8.data import load_config
from s6e8.eda import run_eda
from s6e8.eda_report import write_html_report


def _synthetic_frames(n_train: int = 240, n_test: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i in range(n_train + n_test):
        screen = 3.0 + (i % 12) * 0.7
        rows.append(
            {
                "id": i,
                "age": 18 + (i % 18),
                "daily_screen_time_hours": screen if i % 7 else None,
                "social_media_hours": 0.4 + (i % 9) * 0.3,
                "gaming_hours": 0.2 + (i % 5) * 0.2,
                "work_study_hours": 0.5 + (i % 6) * 0.25,
                "sleep_hours": 5.0 + (i % 4) * 0.4,
                "notifications_per_day": 20 + i % 40,
                "app_opens_per_day": 15 + i % 30,
                "weekend_screen_time": screen * 1.2,
                "gender": ["Male", "Female", "Other", None][i % 4],
                "stress_level": ["Low", "Medium", "High", None][i % 4],
                "academic_work_impact": ["Yes", "No", None][i % 3],
                "addicted_label": int(screen > 7.0),
            }
        )
    df = pd.DataFrame(rows)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].drop(columns=["addicted_label"]).copy()
    return train, test


def test_eda_help_subprocess():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/eda.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in proc.stdout
    assert "--output" in proc.stdout
    assert "--sample-size" in proc.stdout


def test_eda_synthetic_html_is_self_contained(tmp_path: Path, baseline_config_path: Path):
    train_df, test_df = _synthetic_frames()
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    raw = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "synthetic_eda"
    raw["paths"]["train"] = str(train_path)
    raw["paths"]["test"] = str(test_path)
    raw["paths"]["sample_submission"] = str(tmp_path / "missing.csv")
    cfg_path = tmp_path / "eda.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(cfg_path)

    import re

    result = run_eda(train_df, test_df, config, sample_size=120)
    assert result.findings, "expected dynamically generated findings"
    assert result.univariate["auc"].notna().any()
    out = write_html_report(result, tmp_path / "eda_report.html")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "Plotly" in text or "plotly" in text
    assert "执行摘要" in text
    assert re.search(r"<script[^>]+src=['\"]https?://", text, flags=re.I) is None
    assert re.search(r"<link[^>]+href=['\"]https?://", text, flags=re.I) is None
    assert re.search(r"<img[^>]+src=['\"]https?://", text, flags=re.I) is None
    assert "fonts.googleapis.com" not in text.lower()
    assert str(result.target["n_positive"]) in text or "正类" in text
