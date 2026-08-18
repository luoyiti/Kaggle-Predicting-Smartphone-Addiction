from __future__ import annotations

from pathlib import Path

from s6e8.reports_index import collect_reports, inspect_report, render_index, write_index
from tests.scripts_loader import load_script


def test_build_reports_index_help_subprocess():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/build_reports_index.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--reports-dir" in proc.stdout
    assert "--check" in proc.stdout


def test_inspect_report_reads_title_time_and_description(tmp_path: Path):
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head><title>示例报告</title></head>
<body>
<header class="hero">
  <h1>示例标题</h1>
  <p>这是一段简短描述。</p>
</header>
<footer>生成时间 2026-08-17T17:20:55Z。HTML 为单文件自包含。</footer>
</body>
</html>
"""
    path = tmp_path / "sample_report.html"
    path.write_text(html, encoding="utf-8")
    entry = inspect_report(path)
    assert entry.filename == "sample_report.html"
    assert entry.href == "sample_report.html"
    assert entry.name == "示例报告"
    assert entry.generated_at == "2026-08-17T17:20:55Z"
    assert "简短描述" in entry.description


def test_write_index_lists_reports_and_skips_index_itself(tmp_path: Path):
    (tmp_path / "eda_report.html").write_text(
        "<html><head><title>S6E8 EDA 报告</title></head>"
        "<body><header class='hero'><h1>EDA</h1><p>探索性分析。</p></header>"
        "<footer>生成时间 2026-01-02T03:04:05Z。</footer></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "index.html").write_text("<html><title>old</title></html>", encoding="utf-8")
    written = write_index(tmp_path, generated_at="2026-08-18T00:00:00Z")
    text = written.read_text(encoding="utf-8")
    reports = collect_reports(tmp_path)
    assert [item.filename for item in reports] == ["eda_report.html"]
    assert 'href="eda_report.html"' in text
    assert "S6E8 EDA 报告" in text
    assert "2026-01-02T03:04:05Z" in text
    assert "探索性分析" in text
    assert "old" not in text
    assert "总览页生成时间 2026-08-18T00:00:00Z" in text


def test_render_index_empty_directory():
    html = render_index([], generated_at="2026-08-18T00:00:00Z")
    assert "当前没有可展示的分报告" in html
    assert "<table>" not in html


def test_cli_check_detects_stale_index(tmp_path: Path):
    import subprocess
    import sys

    (tmp_path / "a.html").write_text(
        "<html><head><title>A</title></head><body><p>desc</p>"
        "<footer>生成时间 2026-01-01T00:00:00Z。</footer></body></html>",
        encoding="utf-8",
    )
    script = load_script("build_reports_index.py")
    assert script is not None
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_reports_index.py",
            "--reports-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote:" in proc.stdout
    stale = tmp_path / "index.html"
    stale.write_text(stale.read_text(encoding="utf-8").replace("A", "stale"), encoding="utf-8")
    check = subprocess.run(
        [
            sys.executable,
            "scripts/build_reports_index.py",
            "--reports-dir",
            str(tmp_path),
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    assert check.returncode != 0
    assert "stale" in check.stderr or "stale" in check.stdout
