#!/usr/bin/env python3
"""Validate an exact stable or release-candidate Git tag against distributions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


TAG_PATTERN = re.compile(r"^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:rc[1-9][0-9]*)?)$")


def release_from_tag(tag: str) -> tuple[str, bool]:
    match = TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise ValueError(
            f"Unsupported release tag {tag!r}; expected vMAJOR.MINOR.PATCH or vMAJOR.MINOR.PATCHrcN"
        )
    version = match.group("version")
    return version, "rc" in version


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument("--github-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    version, prerelease = release_from_tag(args.tag)
    source_dir = args.source_dir.resolve()
    validator = source_dir / "scripts" / "ci" / "validate_distribution.py"
    subprocess.run(
        [
            sys.executable,
            str(validator),
            str(args.dist_dir),
            "--source-dir",
            str(source_dir),
            "--expected-version",
            version,
        ],
        check=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )

    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"version={version}\n")
            output.write(f"prerelease={'true' if prerelease else 'false'}\n")
            output.write(f"index={'testpypi' if prerelease else 'pypi'}\n")
    print(f"Release contract passed: tag={args.tag}, version={version}, prerelease={prerelease}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
