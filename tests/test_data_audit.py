"""Data-contract audit on synthetic tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from s6e8.data_audit import audit_tables


def _ok_frames(n_train=40, n_test=20):
    rng = np.random.default_rng(0)
    def rows(n, start_id, with_y):
        data = {
            "id": np.arange(start_id, start_id + n),
            "age": rng.integers(18, 50, size=n),
            "daily_screen_time_hours": rng.uniform(2, 10, size=n),
            "social_media_hours": rng.uniform(0, 3, size=n),
            "gaming_hours": rng.uniform(0, 2, size=n),
            "work_study_hours": rng.uniform(0, 3, size=n),
            "sleep_hours": rng.uniform(5, 9, size=n),
            "notifications_per_day": rng.integers(10, 80, size=n),
            "app_opens_per_day": rng.integers(5, 40, size=n),
            "weekend_screen_time": rng.uniform(3, 12, size=n),
            "gender": rng.choice(["Male", "Female", "Other"], size=n),
            "stress_level": rng.choice(["Low", "Medium", "High"], size=n),
            "academic_work_impact": rng.choice(["Yes", "No"], size=n),
        }
        df = pd.DataFrame(data)
        if with_y:
            df["addicted_label"] = (df["daily_screen_time_hours"] > 6).astype(int)
        return df
    return rows(n_train, 0, True), rows(n_test, n_train, False)


def test_clean_tables_ok():
    train, test = _ok_frames()
    report = audit_tables(train, test)
    assert report["ok"] is True
    assert report["id_overlap"] == 0
    assert report["train_id_unique"] is True


def test_schema_gap_is_error():
    train, test = _ok_frames()
    train = train.drop(columns=["sleep_hours"])
    report = audit_tables(train, test)
    assert report["ok"] is False
    assert "sleep_hours" in report["missing_train_columns"]


def test_id_overlap_is_error():
    train, test = _ok_frames()
    test = test.copy()
    test.loc[0, "id"] = train.loc[0, "id"]
    report = audit_tables(train, test)
    assert report["ok"] is False
    assert report["id_overlap"] >= 1


def test_unknown_category_is_warning_not_error():
    train, test = _ok_frames()
    train = train.copy()
    train.loc[0, "gender"] = "NonBinary"
    report = audit_tables(train, test)
    assert report["ok"] is True
    titles = [f["title"] for f in report["findings"]]
    assert any("gender" in t for t in titles)


def test_adversarial_auc_near_chance_when_iid():
    train, test = _ok_frames(n_train=180, n_test=180)
    report = audit_tables(train, test, adversarial_max_rows=400)
    adv = report["adversarial"]
    assert adv["n"] == 360
    assert 0.35 <= adv["auc"] <= 0.65


def test_adversarial_auc_detects_mean_shift():
    train, test = _ok_frames(n_train=180, n_test=180)
    test = test.copy()
    test["age"] = test["age"] + 80
    report = audit_tables(train, test, adversarial_max_rows=400)
    assert report["adversarial"]["auc_flipped"] >= 0.80
    titles = [f["title"] for f in report["findings"]]
    assert any("adversarial" in t for t in titles)

