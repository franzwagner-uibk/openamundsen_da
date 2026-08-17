#!/usr/bin/env python3
"""Select and verify bounded compact-runtime deletion parallelism.

The benchmark creates only marker-owned trees below an explicit scratch root.
Candidate worker counts run against smaller but structurally identical trees;
the fastest candidate then deletes one optional full-scale tree.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from openamundsen_da.manifests import write_manifest_atomic
from openamundsen_da.util.retention import delete_quarantined_runtime_tree


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def _worker_counts(value: str) -> tuple[int, ...]:
    counts = tuple(dict.fromkeys(_positive_int(item.strip()) for item in value.split(",")))
    if not counts:
        raise argparse.ArgumentTypeError("at least one worker count is required")
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--workers", type=_worker_counts, default=(1, 2, 4, 8, 12, 16))
    parser.add_argument("--sample-files", type=_positive_int, default=100_000)
    parser.add_argument("--sample-bytes", type=_positive_int, default=8_000_000_000)
    parser.add_argument("--sample-units", type=_positive_int, default=800)
    parser.add_argument("--full-files", type=int, default=0)
    parser.add_argument("--full-bytes", type=int, default=0)
    parser.add_argument("--full-units", type=_positive_int, default=20_400)
    parser.add_argument(
        "--allocate",
        action="store_true",
        help="Physically allocate payload blocks with posix_fallocate",
    )
    parser.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def _create_payload(path: Path, size: int, *, allocate: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    try:
        if allocate:
            if not hasattr(os, "posix_fallocate"):
                raise RuntimeError("--allocate requires os.posix_fallocate on this platform")
            os.posix_fallocate(fd, 0, size)
        else:
            os.ftruncate(fd, size)
    finally:
        os.close(fd)


def _build_tree(
    root: Path,
    *,
    files: int,
    payload_bytes: int,
    units: int,
    allocate: bool,
) -> dict[str, object]:
    if root.exists():
        raise FileExistsError(f"Benchmark trial already exists: {root}")
    root.mkdir(parents=True)
    base_size, remainder = divmod(payload_bytes, files)
    if base_size < 1:
        raise ValueError("Requested payload bytes must be at least the file count")
    started = time.perf_counter()
    free_before = shutil.disk_usage(root).free
    for index in range(files):
        unit = index % units
        step = unit // 51
        member = unit % 51
        member_name = "open_loop" if member == 0 else f"member_{member:03d}"
        directory = (
            root
            / "steps"
            / f"step_{step:03d}"
            / "ensembles"
            / "prior"
            / member_name
            / "results"
        )
        directory.mkdir(parents=True, exist_ok=True)
        size = base_size + (1 if index < remainder else 0)
        _create_payload(directory / f"artifact_{index:08d}.bin", size, allocate=allocate)
        if (index + 1) % 100_000 == 0 or index + 1 == files:
            print(f"create {root.name}: {index + 1:,}/{files:,} files", flush=True)
    free_after = shutil.disk_usage(root).free
    return {
        "files": files,
        "payload_bytes": payload_bytes,
        "units": units,
        "physically_allocated": allocate,
        "filesystem_allocated_bytes": max(0, free_before - free_after),
        "creation_seconds": time.perf_counter() - started,
    }


def _run_trial(
    scratch_root: Path,
    *,
    label: str,
    workers: int,
    files: int,
    payload_bytes: int,
    units: int,
    allocate: bool,
) -> dict[str, object]:
    trial = scratch_root / label
    built = _build_tree(
        trial,
        files=files,
        payload_bytes=payload_bytes,
        units=units,
        allocate=allocate,
    )
    free_before = shutil.disk_usage(scratch_root).free
    started = time.perf_counter()
    deleted = delete_quarantined_runtime_tree(trial, workers=workers)
    duration = time.perf_counter() - started
    free_after = shutil.disk_usage(scratch_root).free
    if trial.exists():
        raise RuntimeError(f"Deletion returned with a surviving trial tree: {trial}")
    return {
        **built,
        "workers": workers,
        "cleanup_seconds": duration,
        "deleted_files": deleted.files,
        "deleted_directories": deleted.directories,
        "filesystem_freed_bytes": max(0, free_after - free_before),
    }


def main() -> int:
    args = _parse_args()
    scratch_root = args.scratch_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    if any(scratch_root.iterdir()):
        raise RuntimeError(f"Scratch root must be empty: {scratch_root}")
    result_json = args.result_json.resolve()
    try:
        result_json.relative_to(scratch_root)
    except ValueError:
        pass
    else:
        raise ValueError("--result-json must be outside the disposable scratch root")
    if args.full_files < 0 or args.full_bytes < 0:
        raise ValueError("Full-scale file and byte counts cannot be negative")
    if bool(args.full_files) != bool(args.full_bytes):
        raise ValueError("--full-files and --full-bytes must be supplied together")
    disk = shutil.disk_usage(scratch_root)
    largest_files, largest_bytes = max(
        (
            (args.sample_files, args.sample_bytes),
            (args.full_files, args.full_bytes),
        ),
        key=lambda item: item[1],
    )
    allocation_floor = max(largest_bytes, largest_files * 4096)
    benchmark_reserve = int(allocation_floor * 1.10)
    if disk.used + benchmark_reserve + int(disk.total * 0.05) >= int(disk.total * 0.90):
        raise RuntimeError(
            "Benchmark tree would violate the 90% emergency limit plus 5% reserve: "
            f"used={disk.used}, benchmark={benchmark_reserve}, total={disk.total}"
        )

    sample_results = []
    for workers in args.workers:
        print(f"benchmark sample: workers={workers}", flush=True)
        sample_results.append(
            _run_trial(
                scratch_root,
                label=f"sample-w{workers}",
                workers=workers,
                files=args.sample_files,
                payload_bytes=args.sample_bytes,
                units=args.sample_units,
                allocate=args.allocate,
            )
        )
    selected = min(sample_results, key=lambda row: float(row["cleanup_seconds"]))
    full_result = None
    if args.full_files:
        print(f"full-scale confirmation: workers={selected['workers']}", flush=True)
        full_result = _run_trial(
            scratch_root,
            label=f"full-w{selected['workers']}",
            workers=int(selected["workers"]),
            files=args.full_files,
            payload_bytes=args.full_bytes,
            units=args.full_units,
            allocate=args.allocate,
        )
    payload = {
        "contract": "runtime-cleanup-benchmark-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scratch_root": str(scratch_root),
        "filesystem_device": int(scratch_root.stat().st_dev),
        "selected_workers": int(selected["workers"]),
        "sample_results": sample_results,
        "full_result": full_result,
    }
    write_manifest_atomic(result_json, payload)
    print(json.dumps(payload, indent=2), flush=True)
    scratch_root.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
