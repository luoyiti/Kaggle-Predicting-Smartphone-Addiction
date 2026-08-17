#!/usr/bin/env python3
"""Poll `kaggle kernels status` until the latest run finishes."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

SUCCESS = {"complete"}
FAILURE = {
    "error",
    "failed",
    "failure",
    "cancelled",
    "canceled",
    "cancelacknowledged",
    "cancel_acknowledged",
}
STATUS_ALIASES = {
    "complete": "complete",
    "completed": "complete",
    "success": "complete",
    "succeeded": "complete",
    "running": "running",
    "queued": "queued",
    "pending": "queued",
    "error": "error",
    "failed": "error",
    "failure": "error",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "cancelacknowledged": "cancelled",
    "cancel_acknowledged": "cancelled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for a Kaggle kernel run")
    parser.add_argument("kernel_id", help="username/slug")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=14400, help="Seconds before giving up")
    return parser.parse_args()


def _status(kernel_id: str) -> str:
    proc = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    output = output.strip()
    if proc.returncode != 0 and not output:
        raise RuntimeError(f"kaggle kernels status failed with code {proc.returncode}")
    return output


def normalize_status(raw: str) -> str:
    """Extract a comparable status token from `kaggle kernels status` output."""
    text = raw.strip()
    if not text:
        return "unknown"

    quoted = re.search(r'has status\s+"([^"]+)"', text, flags=re.IGNORECASE)
    if quoted:
        token = quoted.group(1)
    else:
        line = text.splitlines()[-1]
        token = line.split()[-1] if line.split() else line

    token = token.strip().strip('"').strip("'").lower()
    token = re.sub(r"^(kernelworkerstatus|kernel_session_status|kernel_worker_status)[._]?", "", token)
    token = token.replace("-", "_").split(".")[-1]
    return STATUS_ALIASES.get(token, token or "unknown")


def main() -> None:
    args = parse_args()
    print(
        f"Waiting for Kaggle kernel {args.kernel_id}. "
        "Full 5-fold LightGBM on this dataset often takes 30–180 minutes on CPU. "
        "This step prints a heartbeat until the kernel finishes.",
        flush=True,
    )
    print(f"Open https://www.kaggle.com/code/{args.kernel_id} (private kernels require login).", flush=True)
    started = time.time()
    last = ""
    while True:
        raw = _status(args.kernel_id)
        status = normalize_status(raw)
        elapsed = int(time.time() - started)
        print(f"[{elapsed:>5}s] kernel={args.kernel_id} status={status}", flush=True)
        if status != last and raw.strip() and raw.strip() != status:
            print(raw.strip(), flush=True)
            last = status
        elif status != last:
            last = status

        if status in SUCCESS:
            print("Kernel completed successfully.", flush=True)
            return
        if status in FAILURE:
            print(f"Kernel failed with status={status}", flush=True)
            print(raw, flush=True)
            raise SystemExit(1)
        if "403" in raw or "401" in raw:
            print(raw, flush=True)
            raise SystemExit("Kaggle authentication/authorization failed.")

        if time.time() - started > args.timeout:
            print(f"Timed out after {args.timeout}s waiting for {args.kernel_id}", flush=True)
            print(f"last_status={raw}", flush=True)
            raise SystemExit(1)
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
