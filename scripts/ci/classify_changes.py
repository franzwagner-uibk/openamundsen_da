#!/usr/bin/env python3
"""Classify a verified changed-path list for safe documentation-only CI."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


TUTORIAL_ASSET_MANIFEST = PurePosixPath(
    "tests/baselines/rofental_es30_tutorial_assets.json"
)
ROOT_CHANGELOG = PurePosixPath("CHANGELOG.md")


def is_documentation_only_path(raw_path: str) -> bool:
    """Return whether one repository-relative path is safe for docs-only CI."""

    if not raw_path or "\x00" in raw_path or "\n" in raw_path or "\r" in raw_path:
        return False
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        return False
    return (
        path in {ROOT_CHANGELOG, TUTORIAL_ASSET_MANIFEST}
        or path.parts[0] == "docs"
    )


def classify_paths(paths: Iterable[str], *, force_full: bool = False) -> bool:
    """Return ``True`` only for a non-empty, proven documentation-only diff."""

    normalized = tuple(paths)
    return (
        not force_full
        and bool(normalized)
        and all(is_documentation_only_path(path) for path in normalized)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-file", type=Path, help="Newline-delimited changed paths"
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Fail safe to full CI without reading changed paths",
    )
    parser.add_argument(
        "--github-output", type=Path, help="Append GitHub Actions outputs"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths: tuple[str, ...] = ()
    if not args.force_full:
        if args.paths_file is None or not args.paths_file.is_file():
            args.force_full = True
        else:
            paths = tuple(args.paths_file.read_text(encoding="utf-8").splitlines())

    docs_only = classify_paths(paths, force_full=args.force_full)
    output = f"docs_only={str(docs_only).lower()}\n"
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as stream:
            stream.write(output)
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
