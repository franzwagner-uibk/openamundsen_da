#!/usr/bin/env python3
"""Validate release archives and reject workspace content leakage."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


FORBIDDEN_PARTS = {
    ".github",
    ".git",
    "context",
    "dev_examples",
    "docs",
    "examples",
    "scripts",
    "testdata",
    "tests",
}
SDIST_ROOT_FILES = {
    "CHANGELOG.md",
    "LICENSE",
    "MANIFEST.in",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
    "setup.cfg",
}
EGG_INFO_FILES = {
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
}
DIST_INFO_FILES = {
    "METADATA",
    "RECORD",
    "WHEEL",
    "entry_points.txt",
    "top_level.txt",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path.cwd())
    parser.add_argument("--expected-version")
    parser.add_argument("--github-output", type=Path)
    return parser.parse_args()


def _metadata_version(raw: str) -> str:
    version = Parser().parsestr(raw).get("Version")
    if not version:
        raise RuntimeError("Distribution metadata has no Version field")
    return version


def _source_package_files(source_dir: Path) -> set[str]:
    package = source_dir / "openamundsen_da"
    expected = {
        path.relative_to(source_dir).as_posix()
        for path in package.rglob("*.py")
        if path.name != "_version.py" and "__pycache__" not in path.parts
    }
    expected.add("openamundsen_da/_version.py")
    expected.add("openamundsen_da/py.typed")
    return expected


def _validate_wheel(wheel: Path, *, source_dir: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        members = {name for name in archive.namelist() if not name.endswith("/")}
        metadata_members = [name for name in members if name.endswith(".dist-info/METADATA")]
        if len(metadata_members) != 1:
            raise RuntimeError(f"Expected one wheel METADATA file, found {metadata_members}")
        version = _metadata_version(archive.read(metadata_members[0]).decode("utf-8"))

    leaked = sorted(name for name in members if FORBIDDEN_PARTS.intersection(PurePosixPath(name).parts))
    if leaked:
        raise RuntimeError(f"Wheel contains forbidden workspace content: {leaked[:10]}")

    package_members = {name for name in members if name.startswith("openamundsen_da/")}
    expected_package_members = _source_package_files(source_dir)
    missing = sorted(expected_package_members - package_members)
    unexpected = sorted(
        name
        for name in package_members - expected_package_members
        if not name.endswith(".pyc")
    )
    if missing or unexpected:
        raise RuntimeError(
            "Wheel runtime package mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    dist_info_roots = {
        PurePosixPath(name).parts[0]
        for name in members
        if PurePosixPath(name).parts[0].endswith(".dist-info")
    }
    if len(dist_info_roots) != 1:
        raise RuntimeError(f"Expected one .dist-info root, found {sorted(dist_info_roots)}")
    dist_info_root = next(iter(dist_info_roots))
    for name in sorted(members - package_members):
        path = PurePosixPath(name)
        if path.parts[0] != dist_info_root:
            raise RuntimeError(f"Unexpected wheel root content: {name}")
        relative = PurePosixPath(*path.parts[1:])
        if len(relative.parts) == 1 and relative.name in DIST_INFO_FILES:
            continue
        if len(relative.parts) == 2 and relative.parts[0] == "licenses" and relative.name == "LICENSE":
            continue
        raise RuntimeError(f"Unexpected wheel metadata content: {name}")

    return version


def _validate_sdist(sdist: Path, *, source_dir: Path) -> str:
    with tarfile.open(sdist, "r:gz") as archive:
        raw_members = [member for member in archive.getmembers() if member.isfile()]
        roots = {PurePosixPath(member.name).parts[0] for member in raw_members}
        if len(roots) != 1:
            raise RuntimeError(f"Expected one sdist root directory, found {sorted(roots)}")
        root = next(iter(roots))
        members = {
            PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
            for member in raw_members
        }
        metadata_member = next((member for member in raw_members if member.name == f"{root}/PKG-INFO"), None)
        if metadata_member is None:
            raise RuntimeError("Source distribution has no root PKG-INFO")
        extracted = archive.extractfile(metadata_member)
        if extracted is None:
            raise RuntimeError("Could not read source distribution PKG-INFO")
        version = _metadata_version(extracted.read().decode("utf-8"))

    leaked = sorted(name for name in members if FORBIDDEN_PARTS.intersection(PurePosixPath(name).parts))
    if leaked:
        raise RuntimeError(f"Source distribution contains forbidden workspace content: {leaked[:10]}")

    expected_package_members = _source_package_files(source_dir)
    package_members = {name for name in members if name.startswith("openamundsen_da/")}
    missing = sorted(expected_package_members - package_members)
    unexpected = sorted(package_members - expected_package_members)
    if missing or unexpected:
        raise RuntimeError(
            "Source distribution runtime package mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    for name in sorted(members - package_members):
        path = PurePosixPath(name)
        if len(path.parts) == 1 and path.name in SDIST_ROOT_FILES:
            continue
        if len(path.parts) == 2 and path.parts[0].endswith(".egg-info") and path.name in EGG_INFO_FILES:
            continue
        raise RuntimeError(f"Unexpected source distribution content: {name}")

    return version


def main() -> int:
    args = _arguments()
    dist_dir = args.dist_dir.resolve()
    source_dir = args.source_dir.resolve()
    wheels = sorted(dist_dir.glob("openamundsen_da-*.whl"))
    sdists = sorted(dist_dir.glob("openamundsen_da-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(f"Expected one wheel and one sdist, found wheels={wheels}, sdists={sdists}")

    wheel_version = _validate_wheel(wheels[0], source_dir=source_dir)
    sdist_version = _validate_sdist(sdists[0], source_dir=source_dir)
    if wheel_version != sdist_version:
        raise RuntimeError(f"Wheel/sdist version mismatch: {wheel_version!r} != {sdist_version!r}")
    if args.expected_version and wheel_version != args.expected_version:
        raise RuntimeError(
            f"Distribution version {wheel_version!r} does not match expected {args.expected_version!r}"
        )

    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"version={wheel_version}\n")

    print(f"Distribution validation passed: version={wheel_version}, wheel={wheels[0].name}, sdist={sdists[0].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
