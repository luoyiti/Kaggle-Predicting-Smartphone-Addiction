#!/usr/bin/env python3
"""Write a GitHub Actions job summary from downloaded Kaggle kernel outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write GHA summary from kernel artifacts")
    parser.add_argument("--output-dir", required=True, help="Directory from kaggle kernels output")
    parser.add_argument("--config", default="")
    parser.add_argument("--accelerator", default="")
    parser.add_argument("--kernel-id", default="")
    parser.add_argument("--status", default="unknown")
    parser.add_argument(
        "--summary-file",
        default="",
        help="Defaults to $GITHUB_STEP_SUMMARY",
    )
    return parser.parse_args()


def _find_json(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def _find_submission(root: Path) -> Path | None:
    hits = sorted(root.rglob("*.csv"))
    preferred = [p for p in hits if "submission" in p.as_posix().lower() or p.parent.name == "submissions"]
    if preferred:
        return preferred[0]
    return hits[0] if hits else None


def _load(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _fmt_runtime(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    minutes = seconds / 60.0
    if minutes < 1:
        return f"{seconds:.1f} sec"
    return f"{minutes:.1f} min"


def _fmt_auc(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def build_markdown(args: argparse.Namespace, metrics: dict, submission: Path | None, output_dir: Path) -> str:
    experiment = metrics.get("experiment", Path(args.config).stem or "unknown")
    accelerator = str(metrics.get("accelerator") or args.accelerator or "n/a").upper()
    n_splits = metrics.get("n_splits", "n/a")
    auc = metrics.get("oof_auc", metrics.get("cv_auc"))
    has_oof = bool(list(output_dir.rglob("oof.npy")) or list(output_dir.rglob("oof.parquet")))
    has_test = bool(list(output_dir.rglob("test.npy")) or list(output_dir.rglob("test.parquet")))
    lines = [
        f"## Experiment: `{experiment}`",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Experiment | `{experiment}` |",
        f"| Config | `{args.config or metrics.get('config') or 'n/a'}` |",
        f"| Accelerator | {accelerator} |",
        f"| CV | {n_splits} Fold |",
        f"| OOF ROC-AUC | {_fmt_auc(auc)} |",
        f"| CV mean ± std | {_fmt_auc(metrics.get('cv_mean'))} ± {_fmt_auc(metrics.get('cv_std'))} |",
        f"| Runtime | {_fmt_runtime(metrics.get('runtime_seconds'))} |",
        f"| Seed | {metrics.get('seed', 'n/a')} |",
        f"| Git commit | `{metrics.get('git_commit') or 'unavailable'}` |",
        f"| Data / feature / model | `{metrics.get('data_version')}` / `{metrics.get('feature_version')}` / `{metrics.get('model_version')}` |",
        f"| Kernel | `{args.kernel_id or 'n/a'}` |",
        f"| Kernel status | `{args.status}` |",
        f"| Artifacts | {'available' if (has_oof or has_test or submission) else 'missing'} |",
        f"| OOF | {'yes' if has_oof else 'no'} |",
        f"| Test prediction | {'yes' if has_test else 'no'} |",
        f"| Submission | {'generated' if submission else 'missing'} |",
        "",
    ]
    if submission:
        lines.append(f"Submission file: `{submission}`")
        lines.append("")
    if not metrics:
        lines.append(
            "> metrics.json was not found in kernel output. "
            "The kernel may have failed before `save_artifacts`."
        )
        lines.append("")
    log_excerpt = collect_log_excerpt(output_dir)
    if log_excerpt:
        lines.append("### Kernel log excerpt")
        lines.append("")
        lines.append("```text")
        lines.append(log_excerpt.rstrip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _decode_kaggle_log_text(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("["):
        return text
    try:
        events = json.loads(stripped)
    except json.JSONDecodeError:
        return text
    if not isinstance(events, list):
        return text
    decoded = "".join(str(event.get("data", "")) for event in events if isinstance(event, dict))
    return decoded or text


def collect_log_excerpt(output_dir: Path, max_chars: int = 8000) -> str:
    if not output_dir.is_dir():
        return ""
    hits = sorted(
        p
        for p in output_dir.rglob("*")
        if p.is_file() and (p.suffix.lower() in {".log", ".txt"} or "log" in p.name.lower())
    )
    chunks: list[str] = []
    remaining = max_chars
    for path in hits:
        raw = path.read_bytes()
        if b"\x00" in raw[:4096]:
            continue
        text = _decode_kaggle_log_text(raw.decode("utf-8", errors="replace"))
        if len(text) > remaining:
            text = text[-remaining:]
        chunks.append(f"----- {path.name} -----\n{text}")
        remaining -= len(text)
        if remaining <= 0:
            break
    return "\n\n".join(chunks)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    metrics = _load(_find_json(output_dir, "metrics.json"))
    if not metrics:
        metrics = _load(_find_json(output_dir, "experiment.json"))
    submission = _find_submission(output_dir)
    markdown = build_markdown(args, metrics, submission, output_dir)

    summary_path = args.summary_file or __import__("os").environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(summary_path).open("a", encoding="utf-8") as fh:
            fh.write(markdown)
            if not markdown.endswith("\n"):
                fh.write("\n")
    print(markdown)


if __name__ == "__main__":
    main()
