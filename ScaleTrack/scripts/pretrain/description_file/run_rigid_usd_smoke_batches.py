#!/usr/bin/env python3
"""Run rigid-USD stability workers in isolated Kit processes and merge reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _tail(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-limit:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--blacklist", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-assets", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=120.0,
        help="Seconds before a worker range is split to isolate a pathological asset",
    )
    parser.add_argument(
        "--single-retries",
        type=int,
        default=2,
        help="Retry an isolated asset when Kit exits before writing a report",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.worker_timeout <= 0:
        raise ValueError("--worker-timeout must be positive")
    if args.single_retries < 0:
        raise ValueError("--single-retries must be non-negative")

    root = args.usd_root.expanduser().resolve()
    paths = sorted(
        path for path in root.rglob("*.usd") if "configuration" not in path.parts
    )
    if args.max_assets is not None:
        paths = paths[: args.max_assets]
    if not paths:
        raise ValueError(f"no top-level USD assets below {root}")

    worker = Path(__file__).with_name("smoke_test_rigid_usds.py")
    merged_results = {}
    merged_blacklist = []
    worker_logs = []
    with tempfile.TemporaryDirectory(prefix="rigid-usd-smoke-") as temp_dir:
        temp = Path(temp_dir)
        worker_id = 0

        def run_range(
            start: int,
            count: int,
            retries_remaining: int | None = None,
        ) -> tuple[dict, list[str]]:
            """Run one worker range, recursively splitting crashes and timeouts."""

            nonlocal worker_id
            if retries_remaining is None:
                retries_remaining = args.single_retries
            current_worker_id = worker_id
            worker_id += 1
            report_path = temp / f"report_{current_worker_id:04d}.json"
            blacklist_path = temp / f"blacklist_{current_worker_id:04d}.json"
            command = [
                sys.executable,
                str(worker),
                str(root),
                "--report",
                str(report_path),
                "--blacklist",
                str(blacklist_path),
                "--start-index",
                str(start),
                "--max-assets",
                str(count),
                "--batch-size",
                str(count),
                "--device",
                args.device,
                "--fast-exit",
                "--headless",
            ]
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=args.worker_timeout,
                )
                failure = None
                stdout_tail = _tail(completed.stdout)
                stderr_tail = _tail(completed.stderr)
                returncode = completed.returncode
            except subprocess.TimeoutExpired as exc:
                failure = f"worker timed out after {args.worker_timeout:g} seconds"
                stdout_tail = _tail(exc.stdout)
                stderr_tail = _tail(exc.stderr)
                returncode = None

            if failure is None and report_path.is_file() and blacklist_path.is_file():
                batch_results = json.loads(report_path.read_text(encoding="utf-8"))
                batch_blacklist = json.loads(
                    blacklist_path.read_text(encoding="utf-8")
                )["usd_paths"]
                expected = {str(path) for path in paths[start : start + count]}
                if set(batch_results) == expected:
                    return batch_results, batch_blacklist
                failure = (
                    f"worker returned {len(batch_results)} results, "
                    f"expected {len(expected)}"
                )
            elif failure is None:
                failure = "worker exited without complete report files"

            worker_logs.append(
                {
                    "worker": current_worker_id,
                    "start": start,
                    "count": count,
                    "returncode": returncode,
                    "failure": failure,
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                }
            )
            if count == 1 and retries_remaining > 0:
                print(
                    f"[RETRY] {paths[start]}: {failure}; "
                    f"remaining={retries_remaining}",
                    flush=True,
                )
                return run_range(start, count, retries_remaining - 1)
            if count > 1:
                left_count = count // 2
                print(
                    f"[ISOLATE] start={start} count={count}: {failure}; "
                    f"split={left_count}+{count - left_count}",
                    flush=True,
                )
                left_results, left_blacklist = run_range(start, left_count)
                right_results, right_blacklist = run_range(
                    start + left_count, count - left_count
                )
                return (
                    {**left_results, **right_results},
                    [*left_blacklist, *right_blacklist],
                )

            usd_path = str(paths[start])
            print(f"[BLACKLIST] {usd_path}: {failure}", flush=True)
            return (
                {
                    usd_path: {
                        "stable": False,
                        "error": failure,
                        "worker_stdout_tail": stdout_tail,
                        "worker_stderr_tail": stderr_tail,
                    }
                },
                [usd_path],
            )

        for start in range(0, len(paths), args.batch_size):
            count = min(args.batch_size, len(paths) - start)
            batch_results, batch_blacklist = run_range(start, count)
            if set(batch_results).intersection(merged_results):
                raise RuntimeError(f"start {start}: duplicate result paths")
            merged_results.update(batch_results)
            merged_blacklist.extend(batch_blacklist)
            print(
                f"[BATCH] {min(start + count, len(paths))}/{len(paths)} "
                f"blacklist={len(merged_blacklist)}",
                flush=True,
            )

    if worker_logs:
        failure_path = args.report.with_suffix(".worker_failures.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(worker_logs, indent=2) + "\n", encoding="utf-8"
        )
    if len(merged_results) != len(paths):
        raise RuntimeError(
            f"expected {len(paths)} results, received {len(merged_results)}"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.blacklist.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(merged_results, indent=2) + "\n", encoding="utf-8"
    )
    args.blacklist.write_text(
        json.dumps({"usd_paths": merged_blacklist}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"assets={len(paths)} stable={len(paths) - len(merged_blacklist)} "
        f"blacklist={len(merged_blacklist)} report={args.report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
