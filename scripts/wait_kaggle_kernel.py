#!/usr/bin/env python3
"""Poll `kaggle kernels status` until the latest run finishes."""

from __future__ import annotations

import argparse
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


def _normalize(raw: str) -> str:
    line = raw.strip().splitlines()[-1] if raw.strip() else ""
    token = line.split()[-1] if line.split() else line
    return token.strip().strip('"').strip("'").lower()


def main() -> None:
    args = parse_args()
    started = time.time()
    last = ""
    while True:
        raw = _status(args.kernel_id)
        status = _normalize(raw)
        if status != last:
            print(f"kernel={args.kernel_id} status={status}", flush=True)
            if raw.strip() and raw.strip() != status:
                print(raw.strip(), flush=True)
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
