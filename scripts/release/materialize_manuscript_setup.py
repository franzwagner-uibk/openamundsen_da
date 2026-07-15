#!/usr/bin/env python3
"""Materialize the exact Rofental input snapshot used by the manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "examples" / "rofental"
DEFAULT_SNAPSHOT = REPO_ROOT / "tests" / "baselines" / "rofental_es30_manuscript_inputs"
DEFAULT_MANIFEST = DEFAULT_SNAPSHOT / "manifest.json"
SCHEMA_VERSION = 1


class ManuscriptSetupError(ValueError):
    """Raised when the frozen manuscript setup cannot be materialized safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManuscriptSetupError(f"Cannot read manuscript input manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManuscriptSetupError(
            f"Manuscript input manifest schema_version must be {SCHEMA_VERSION}: {path}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ManuscriptSetupError(f"Manuscript input manifest requires a non-empty files list: {path}")
    return manifest


def _snapshot_file(snapshot_root: Path, record: Mapping[str, Any]) -> tuple[Path, Path]:
    relative = Path(str(record.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ManuscriptSetupError(f"Invalid manuscript snapshot path: {relative}")
    source = (snapshot_root / relative).resolve(strict=True)
    try:
        source.relative_to(snapshot_root)
    except ValueError as exc:
        raise ManuscriptSetupError(f"Manuscript snapshot path escapes its root: {relative}") from exc
    expected = str(record.get("sha256", ""))
    actual = _sha256(source)
    if actual != expected:
        raise ManuscriptSetupError(
            f"Manuscript snapshot checksum differs for {relative}: expected {expected}, got {actual}"
        )
    return relative, source


def materialize_setup(
    *,
    base_setup: Path,
    snapshot_root: Path,
    manifest_path: Path,
    target: Path,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Copy a shipped setup and apply the checksum-validated manuscript overlay."""
    base_setup = base_setup.resolve(strict=True)
    snapshot_root = snapshot_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    target = target.expanduser().resolve()
    for protected in (base_setup, snapshot_root):
        if target == protected or protected in target.parents or target in protected.parents:
            raise ManuscriptSetupError(f"Target must be outside protected source tree: {target}")
    if target.exists():
        if not overwrite:
            raise ManuscriptSetupError(f"Target already exists; pass --overwrite to replace it: {target}")
        if target == Path(target.anchor):
            raise ManuscriptSetupError(f"Refusing to remove filesystem root: {target}")
        shutil.rmtree(target)

    manifest = _read_manifest(manifest_path)
    snapshot_files = tuple(_snapshot_file(snapshot_root, record) for record in manifest["files"])
    shutil.copytree(base_setup, target)
    written: list[Path] = []
    for relative, source in snapshot_files:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        written.append(destination.resolve())
    return tuple(written)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="New setup directory to create")
    parser.add_argument("--base-setup", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        written = materialize_setup(
            base_setup=args.base_setup,
            snapshot_root=args.snapshot,
            manifest_path=args.manifest,
            target=args.target,
            overwrite=bool(args.overwrite),
        )
    except (OSError, ManuscriptSetupError) as exc:
        print(f"manuscript setup error: {exc}")
        return 1
    print(f"Materialized manuscript setup at {args.target.resolve()} ({len(written)} frozen inputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
