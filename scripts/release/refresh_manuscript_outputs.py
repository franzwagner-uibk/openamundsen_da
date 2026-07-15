#!/usr/bin/env python3
"""Reproduce the publication-analysis state from a validated selected run."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from openamundsen_da.api import render_project
from openamundsen_da.benchmark.pipeline import run_project_benchmark
from openamundsen_da.exceptions import OpenAmundsenDAError
from render_manuscript_profile import render_manuscript_profile
from validate_manuscript_reference import (
    DEFAULT_ASSET_MANIFEST,
    DEFAULT_CONTRACT,
    DEFAULT_INPUT_MANIFEST,
    ManuscriptReferenceError,
    _read_json,
    validate_reference,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_NAME = "project_2022_2023"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_publication_overlay(
    root: Path,
    contract: Mapping[str, Any],
    *,
    source_root: Path = REPO_ROOT,
) -> tuple[Path, ...]:
    """Apply checksum-validated publication-analysis inputs to a selected run."""
    root = root.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    written: list[Path] = []
    for record in contract.get("publication_analysis_overlay", ()):  # type: ignore[assignment]
        relative = Path(str(record["path"]))
        source_relative = Path(str(record["source"]))
        if (
            relative.is_absolute()
            or source_relative.is_absolute()
            or ".." in relative.parts
            or ".." in source_relative.parts
        ):
            raise ManuscriptReferenceError(
                f"Invalid publication overlay mapping: {source_relative} -> {relative}"
            )
        source = source_root / source_relative
        if not source.is_file():
            raise ManuscriptReferenceError(f"Missing publication overlay source: {source}")
        actual = _sha256(source)
        expected = str(record["sha256"])
        if actual != expected:
            raise ManuscriptReferenceError(
                f"Publication overlay source checksum differs for {source_relative}: "
                f"expected {expected}, got {actual}"
            )
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        written.append(destination.resolve())
    return tuple(written)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Completed selected Rofental setup root")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--asset-manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--manuscript-root", type=Path)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the overlay and regenerate benchmark, plots, maps and report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        contract = _read_json(args.contract.resolve(strict=True))
        assets = _read_json(args.asset_manifest.resolve(strict=True))
        inputs = _read_json(args.input_manifest.resolve(strict=True))
        differences = validate_reference(
            root,
            contract,
            assets,
            inputs,
            stage="simulation",
        )
        if differences:
            print("\n".join(differences))
            return 1
        overlay = tuple(str(record["path"]) for record in contract["publication_analysis_overlay"])
        if not args.apply:
            print("Selected simulation contract matches")
            print("Publication overlay preview: " + ", ".join(overlay))
            return 0

        written = apply_publication_overlay(root, contract)
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
            inputs,
            manuscript_root=args.manuscript_root,
            stage="publication",
        )
    except (OSError, ManuscriptReferenceError, OpenAmundsenDAError, RuntimeError, ValueError) as exc:
        print(f"manuscript output refresh error: {exc}")
        return 2
    if differences:
        print("\n".join(differences))
        return 1
    print(
        f"Publication outputs match after refreshing {len(written)} analysis input(s): "
        f"{project_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
