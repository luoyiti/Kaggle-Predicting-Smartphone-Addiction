from pathlib import Path


def test_mle_modeling_report_has_required_sections():
    path = Path("reports/mle_modeling_report.html")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for needle in (
        "Iteration Compact",
        "Prediction and data contracts",
        "Experiment ledger",
        "Path coverage matrix",
        "How to run",
        "Honest metrics",
        "0.963771",
        "synthetic smoke",
        "no further high-value local modeling path",
        "scripts/train.py",
        "scripts/promote.py",
    ):
        assert needle in text, needle
    assert "MLflow Server" not in text or "not" in text.lower()
