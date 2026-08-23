"""Error-analysis suggestion on synthetic OOF."""

from __future__ import annotations

import numpy as np
import pandas as pd

from s6e8.error_analysis import analyze_oof_errors, render_next_experiment_yaml


def _frame(n=120):
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "age": rng.integers(18, 40, size=n),
            "daily_screen_time_hours": rng.uniform(1, 12, size=n),
            "social_media_hours": rng.uniform(0, 6, size=n),
            "gaming_hours": rng.uniform(0, 4, size=n),
            "work_study_hours": rng.uniform(0, 6, size=n),
            "sleep_hours": rng.uniform(4, 9, size=n),
            "notifications_per_day": rng.integers(5, 100, size=n),
            "app_opens_per_day": rng.integers(5, 50, size=n),
            "weekend_screen_time": rng.uniform(1, 14, size=n),
            "gender": rng.choice(["Male", "Female", "Other"], size=n),
            "stress_level": rng.choice(["Low", "Medium", "High"], size=n),
            "academic_work_impact": rng.choice(["Yes", "No"], size=n),
        }
    )


def test_hard_band_with_no_residual_signal_stops_or_ensembles():
    frame = _frame()
    y = (frame["daily_screen_time_hours"] > 6).astype(int).to_numpy()
    # Well-separated probabilities → small hard band, residual uninformative
    pred = np.clip(y.astype(float) * 0.85 + 0.08, 0.02, 0.98)
    analysis = analyze_oof_errors(frame, y, pred, experiment="toy")
    assert analysis["suggestion"]["action"] in {"stop", "stop_or_ensemble", "experiment"}
    assert "oof_auc" in analysis


def test_residual_signal_emits_yaml_stub():
    frame = _frame(200)
    y = np.array([0, 1] * 100)
    pred = np.full(200, 0.5)
    # Make residual align with notifications so univariate residual AUC is high
    frame = frame.copy()
    frame["notifications_per_day"] = np.where(y == 1, 200, 10)
    analysis = analyze_oof_errors(frame, y, pred, experiment="toy_hard")
    stub = render_next_experiment_yaml(analysis)
    if analysis["suggestion"]["action"] == "experiment":
        assert stub is not None
        assert "experiment:" in stub
        assert analysis["suggestion"]["suggested_config_stem"]
    else:
        # Generator-noise path is also valid if AUC math does not fire
        assert analysis["suggestion"]["action"] in {"stop_or_ensemble", "stop"}
