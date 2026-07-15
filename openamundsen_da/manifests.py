"""Versioned, hash-bound and atomically persisted workflow manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: object) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_inventory(*, root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    """Return a deterministic content inventory with root-relative paths."""
    root = Path(root).resolve()
    unique: dict[str, Path] = {}
    for raw in files:
        path = Path(raw)
        if not path.is_file() or path.is_symlink():
            continue
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Manifest input is outside {root}: {resolved}") from exc
        unique[relative] = resolved
    return [
        {
            "path": relative,
            "size": unique[relative].stat().st_size,
            "sha256": sha256_file(unique[relative]),
        }
        for relative in sorted(unique)
    ]


def inventory_digest(inventory: list[dict[str, Any]]) -> str:
    """Return the stable digest for a file inventory."""
    return hash_json(inventory)


def recursive_files(path: Path) -> list[Path]:
    """List regular non-symlink files below a path in deterministic order."""
    path = Path(path)
    if path.is_file() and not path.is_symlink():
        return [path]
    if not path.is_dir():
        return []
    return [item for item in sorted(path.rglob("*")) if item.is_file() and not item.is_symlink()]


def load_manifest(path: Path) -> dict[str, Any] | None:
    """Load a JSON manifest when it exists and validate its root/version."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Manifest root must be an object: {path}")
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version in {path}: {data.get('schema_version')!r}"
        )
    return data


def write_manifest_atomic(path: Path, data: dict[str, Any]) -> Path:
    """Durably replace a JSON manifest without exposing partial content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["schema_version"] = MANIFEST_SCHEMA_VERSION
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path.resolve()


def workflow_manifest_path(project_dir: Path, name: str) -> Path:
    """Return a package-owned preparation/observation manifest path."""
    return Path(project_dir) / ".openamundsen-da" / "manifests" / f"{name}.json"


def project_run_manifest_path(project_dir: Path) -> Path:
    """Return the canonical atomic project-run manifest path."""
    return Path(project_dir) / "results" / "run_manifest.json"


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "file_inventory",
    "hash_json",
    "inventory_digest",
    "load_manifest",
    "project_run_manifest_path",
    "recursive_files",
    "sha256_file",
    "workflow_manifest_path",
    "write_manifest_atomic",
]
