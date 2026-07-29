#!/usr/bin/env python3
"""Regenerate publication outputs from a validated completed Rofental run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from openamundsen_da.api import render_project
from openamundsen_da.benchmark.pipeline import run_project_benchmark
from openamundsen_da.exceptions import OpenAmundsenDAError
from render_manuscript_profile import render_manuscript_profile
from validate_manuscript_reference import (
    DEFAULT_ASSET_MANIFEST,
    DEFAULT_CONTRACT,
    ManuscriptReferenceError,
    _read_json,
    validate_reference,
)


PROJECT_NAME = "project_2022_2023"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Completed selected Rofental setup root")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--asset-manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument("--manuscript-root", type=Path)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Regenerate benchmark, plots, maps, report and manuscript profile",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        contract = _read_json(args.contract.resolve(strict=True))
        assets = _read_json(args.asset_manifest.resolve(strict=True))
        differences = validate_reference(
            root,
            contract,
            assets,
            stage="simulation",
        )
        if differences:
            print("\n".join(differences))
            return 1
        if not args.apply:
            print("Selected simulation contract matches")
            print("Publication refresh preview: benchmark, plots, maps, report and paper profile")
            return 0

        project_dir = root / "projects" / PROJECT_NAME
        run_project_benchmark(
            project_dir=project_dir,
            setup_dir=root,
            max_workers=args.max_workers,
            overwrite=True,
            reuse_existing_prerequisites=True,
        )
        render_project(project_dir, max_workers=args.max_workers)
        render_manuscript_profile(root)
        differences = validate_reference(
            root,
            contract,
            assets,
            manuscript_root=args.manuscript_root,
            stage="publication",
        )
    except (OSError, ManuscriptReferenceError, OpenAmundsenDAError, RuntimeError, ValueError) as exc:
        print(f"manuscript output refresh error: {exc}")
        return 2
    if differences:
        print("\n".join(differences))
        return 1
    print(f"Publication outputs match after refresh: {project_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
