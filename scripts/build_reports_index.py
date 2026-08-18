#!/usr/bin/env python3
"""Scan reports/*.html and write reports/index.html."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s6e8.reports_index import DEFAULT_REPORTS_DIR, render_index, collect_reports, write_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the reports/ overview index page")
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory that contains HTML reports (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Index HTML path (default: <reports-dir>/index.html)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the existing index.html is stale (do not write)",
    )
    return parser.parse_args()


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    return path


def main() -> None:
    args = parse_args()
    reports_dir = _resolve(args.reports_dir)
    output = _resolve(args.output) if args.output else reports_dir / "index.html"
    reports = collect_reports(reports_dir)
    # Keep --check stable: reuse the timestamp already stored in the file when present.
    existing = output.read_text(encoding="utf-8") if output.exists() else ""
    generated_at = None
    if args.check and existing:
        marker = "总览页生成时间 "
        idx = existing.find(marker)
        if idx >= 0:
            rest = existing[idx + len(marker) :]
            generated_at = rest.split("。", 1)[0].strip() or None
    html_text = render_index(reports, generated_at=generated_at)
    if args.check:
        if existing != html_text:
            raise SystemExit(f"stale reports index: {output}")
        print(f"ok: {output} ({len(reports)} reports)")
        return
    written = write_index(reports_dir, output_path=output, generated_at=generated_at)
    print(f"wrote: {written} ({len(reports)} reports)")
    for item in reports:
        print(f"  {item.filename}\t{item.name}\t{item.generated_at}")


if __name__ == "__main__":
    main()
