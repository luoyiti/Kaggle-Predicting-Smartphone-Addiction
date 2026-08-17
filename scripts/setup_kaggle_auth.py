#!/usr/bin/env python3
"""Configure Kaggle CLI credentials from environment variables. Never logs secrets."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

KAGGLE_DIR = Path.home() / ".kaggle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Kaggle CLI credentials from env")
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT"),
        help="Optional GitHub Actions output file",
    )
    return parser.parse_args()


def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _parse_json_token(raw: str) -> dict | None:
    text = raw.strip()
    if not text.startswith("{"):
        return None
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("KAGGLE_API_TOKEN JSON must be an object")
    return data


def configure() -> dict[str, str]:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()

    if not token and not (username and key):
        raise SystemExit(
            "Missing Kaggle credentials.\n"
            "Add a GitHub Actions secret named KAGGLE_API_TOKEN "
            "(token from https://www.kaggle.com/settings/api).\n"
            "If the token is the new opaque string, also set KAGGLE_USERNAME.\n"
            "Legacy fallback: KAGGLE_USERNAME + KAGGLE_KEY (kaggle.json)."
        )

    json_token = _parse_json_token(token) if token else None
    if json_token is not None:
        username = str(json_token.get("username") or username).strip()
        key = str(json_token.get("key") or key).strip()
        if username and key:
            _write_private(KAGGLE_DIR / "kaggle.json", json.dumps({"username": username, "key": key}))
        else:
            _write_private(KAGGLE_DIR / "access_token", token)
    elif token:
        _write_private(KAGGLE_DIR / "access_token", token)
    elif username and key:
        _write_private(
            KAGGLE_DIR / "kaggle.json",
            json.dumps({"username": username, "key": key}),
        )

    if not username:
        raise SystemExit(
            "Could not determine Kaggle username for kernel id (username/slug).\n"
            "Set GitHub secret KAGGLE_USERNAME, or use a legacy kaggle.json token "
            "that includes the username field."
        )

    return {"username": username}


def _append_github_output(path: str | None, values: dict[str, str]) -> None:
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def main() -> None:
    args = parse_args()
    resolved = configure()
    print(f"kaggle_username={resolved['username']}")
    _append_github_output(args.github_output, {"kaggle_username": resolved["username"]})


if __name__ == "__main__":
    main()
