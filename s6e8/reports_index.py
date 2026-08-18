"""Scan reports/*.html and render a simple overview index page."""

from __future__ import annotations

import html
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = ROOT / "reports"
INDEX_NAME = "index.html"

_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_H1_RE = re.compile(r"<h1\b[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_META_DESC_RE = re.compile(
    r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)
_HERO_P_RE = re.compile(
    r"<header\b[^>]*>.*?<p\b[^>]*>(.*?)</p>",
    re.IGNORECASE | re.DOTALL,
)
_FIRST_P_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
_FOOTER_TIME_RE = re.compile(r"生成时间\s+([^。<]+)")
_ENV_TIME_RE = re.compile(
    r"环境\s*/\s*时间</dt>\s*<dd>[^<]*·\s*([^<]+)</dd>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Filename fallbacks when a report does not expose a description.
KNOWN_DESCRIPTIONS = {
    "eda_report.html": (
        "全量 train/test 的探索性数据分析：单变量 AUC、缺失、偏移与泄漏检查。"
    ),
}


@dataclass(frozen=True)
class ReportEntry:
    filename: str
    href: str
    name: str
    generated_at: str
    description: str


def _strip_tags(text: str) -> str:
    text = html.unescape(_TAG_RE.sub(" ", text))
    return _WS_RE.sub(" ", text).strip()


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    value = _strip_tags(match.group(1))
    return value or None


def _git_commit_time(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def _mtime_iso(path: Path) -> str:
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def inspect_report(path: Path, reports_dir: Path | None = None) -> ReportEntry:
    text = path.read_text(encoding="utf-8", errors="replace")
    name = _first(_TITLE_RE, text) or _first(_H1_RE, text) or path.stem
    description = (
        _first(_META_DESC_RE, text)
        or _first(_HERO_P_RE, text)
        or _first(_FIRST_P_RE, text)
        or KNOWN_DESCRIPTIONS.get(path.name)
        or "HTML 报告"
    )
    generated_at = (
        _first(_FOOTER_TIME_RE, text)
        or _first(_ENV_TIME_RE, text)
        or _git_commit_time(path)
        or _mtime_iso(path)
    )
    if reports_dir is not None:
        try:
            href = path.relative_to(reports_dir).as_posix()
        except ValueError:
            href = path.name
    else:
        href = path.name
    return ReportEntry(
        filename=path.name,
        href=href,
        name=name,
        generated_at=generated_at,
        description=description,
    )


def iter_report_files(reports_dir: Path) -> Iterable[Path]:
    files = [
        path
        for path in reports_dir.rglob("*.html")
        if path.is_file() and path.name != INDEX_NAME
    ]
    files.sort(key=lambda p: p.name.lower())
    return files


def collect_reports(reports_dir: Path) -> list[ReportEntry]:
    return [inspect_report(path, reports_dir) for path in iter_report_files(reports_dir)]


def render_index(
    reports: list[ReportEntry],
    *,
    generated_at: str | None = None,
) -> str:
    now = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    if reports:
        for item in reports:
            rows.append(
                "<tr>"
                f'<td><a href="{html.escape(item.href, quote=True)}">'
                f"{html.escape(item.name)}</a></td>"
                f"<td>{html.escape(item.generated_at)}</td>"
                f"<td>{html.escape(item.description)}</td>"
                "</tr>"
            )
        table_body = "\n".join(rows)
        listing = f"""<table>
<thead>
<tr><th>报告名称</th><th>生成时间</th><th>说明</th></tr>
</thead>
<tbody>
{table_body}
</tbody>
</table>"""
    else:
        listing = "<p>当前没有可展示的分报告。</p>"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>S6E8 报告总览</title>
</head>
<body>
<h1>S6E8 报告总览</h1>
<p>Kaggle Playground Series S6E8 — Predicting Smartphone Addiction。本页列出 <code>reports/</code> 目录中的 HTML 分报告。</p>
{listing}
<p>总览页生成时间 {html.escape(now)}。由 <code>scripts/build_reports_index.py</code> 扫描生成。</p>
</body>
</html>
"""


def write_index(
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    *,
    output_path: Path | str | None = None,
    generated_at: str | None = None,
) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    output = Path(output_path) if output_path else reports_dir / INDEX_NAME
    html_text = render_index(collect_reports(reports_dir), generated_at=generated_at)
    output.write_text(html_text, encoding="utf-8")
    return output
