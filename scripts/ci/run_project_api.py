#!/usr/bin/env python3
"""Exercise the installed public project API from a multiprocessing-safe entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from openamundsen_da import prepare_project, run_project


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("project_dir", type=Path)
    prepare_parser.add_argument("--overwrite", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("project_dir", type=Path)
    run_parser.add_argument("--max-workers", type=int)
    return parser.parse_args()


def main() -> int:
    """Run one public project operation and print its stable result fields."""
    args = _arguments()
    if args.operation == "prepare":
        result = prepare_project(args.project_dir, overwrite=args.overwrite)
    else:
        result = run_project(args.project_dir, max_workers=args.max_workers)
    print(f"{result.status.value}: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
