"""Versioned, hash-bound and atomically persisted workflow manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

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


def file_inventory(
    *,
    root: Path,
    files: Iterable[Path],
    hash_file: Callable[[Path], str] = sha256_file,
) -> list[dict[str, Any]]:
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
            "sha256": hash_file(unique[relative]),
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


def project_scientific_input_inventory(
    config,
    preparation: dict | None = None,
    *,
    identity_root: Path | None = None,
    hash_file: Callable[[Path], str] = sha256_file,
) -> tuple[list[dict], str]:
    """Return the canonical prepared-run scientific input inventory."""
    from openamundsen_da.exceptions import ProjectRunError

    def setup_path(raw: object) -> Path:
        return (config.setup_dir / str(raw)).resolve()

    def logical_setup_path(raw: object) -> Path:
        return config.setup_dir / str(raw)

    domain = str(config.setup.get("domain", "")).strip()
    resolution_raw = config.setup.get("resolution")
    input_data = config.setup.get("input_data")
    grids = input_data.get("grids") if isinstance(input_data, dict) else None
    generated_roi_paths: set[Path] = set()
    if domain and resolution_raw is not None and isinstance(grids, dict) and grids.get("dir"):
        try:
            resolution_float = float(resolution_raw)
        except (TypeError, ValueError):
            resolution = str(resolution_raw).strip()
        else:
            resolution = (
                str(int(resolution_float))
                if resolution_float.is_integer()
                else str(resolution_raw).strip()
            )
        base = setup_path(grids["dir"]) / f"roi_{domain}_{resolution}"
        generated_roi_paths = {base.with_suffix(".asc"), base.with_suffix(".prj")}

    files = [config.setup_yaml, config.project_yaml, *sorted(config.project_dir.glob("*.yml"))]
    scientific_roots: set[Path] = {config.setup_dir / "env"}
    files = [
        path
        for path in files
        if "/obs/" in path.as_posix()
        or not "/ensembles/" in path.as_posix()
        and not "/assim/" in path.as_posix()
        and not "/results/" in path.as_posix()
    ]
    if isinstance(input_data, dict):
        for name in ("grids", "meteo"):
            section = input_data.get(name)
            if isinstance(section, dict) and section.get("dir"):
                scientific_roots.add(logical_setup_path(section["dir"]))
                files.extend(
                    path
                    for path in recursive_files(setup_path(section["dir"]))
                    if path.resolve() not in {item.resolve() for item in generated_roi_paths}
                )
    obs = config.project.get("obs")
    if isinstance(obs, dict):
        for section in obs.values():
            if isinstance(section, dict) and section.get("dir"):
                scientific_roots.add(logical_setup_path(section["dir"]))
                files.extend(recursive_files(setup_path(section["dir"])))
            if isinstance(section, dict):
                for key in (
                    "summary_csv",
                    "wet_snow_line_diagnostics_csv",
                    "acquisition_manifest",
                ):
                    if section.get(key):
                        files.extend(recursive_files(setup_path(section[key])))
    files.extend(recursive_files(config.setup_dir / "env"))
    if preparation is not None:
        preparation_path = workflow_manifest_path(config.project_dir, "preparation")
        files.append(preparation_path)
        recorded = preparation.get("outputs")
        if not isinstance(recorded, list) or not isinstance(preparation.get("output_digest"), str):
            raise ProjectRunError("Preparation manifest is missing its output inventory")
        output_paths = [
            config.setup_dir / str(entry.get("path", ""))
            for entry in recorded
            if isinstance(entry, dict)
        ]
        current = file_inventory(
            root=config.setup_dir,
            files=output_paths,
            hash_file=hash_file,
        )
        if inventory_digest(current) != preparation["output_digest"]:
            raise ProjectRunError("Preparation outputs differ from the completed preparation manifest")
        files.extend(output_paths)
    inventory = file_inventory(
        root=config.setup_dir,
        files=files,
        hash_file=hash_file,
    )
    allowed_root = Path(identity_root or config.setup_dir).resolve()
    symlink_records: dict[str, dict[str, object]] = {}
    scan_roots = scientific_roots | {
        Path(path).parent if Path(path).is_file() else Path(path)
        for path in files
    }
    for scan_root in sorted(scan_roots):
        if scan_root.is_symlink():
            raise ProjectRunError(
                f"Scientific input directory symlinks are unsupported: {scan_root}"
            )
        if not scan_root.is_dir():
            continue
        for directory, dir_names, file_names in os.walk(scan_root, followlinks=False):
            directory_symlinks = [
                Path(directory) / name
                for name in dir_names
                if (Path(directory) / name).is_symlink()
            ]
            if directory_symlinks:
                raise ProjectRunError(
                    "Scientific input directory symlinks are unsupported: "
                    + ", ".join(str(path) for path in sorted(directory_symlinks))
                )
            dir_names[:] = [
                name
                for name in dir_names
                if not (Path(directory) / name).is_symlink()
            ]
            for name in sorted(file_names):
                logical = Path(directory) / name
                if not logical.is_symlink():
                    continue
                target = logical.resolve(strict=True)
                try:
                    target_relative = target.relative_to(allowed_root).as_posix()
                except ValueError as exc:
                    raise ProjectRunError(
                        f"Scientific input symlink escapes {allowed_root}: {logical} -> {target}"
                    ) from exc
                if not target.is_file():
                    raise ProjectRunError(
                        f"Scientific input symlink target is not a file: {logical} -> {target}"
                    )
                try:
                    logical_relative = logical.relative_to(config.setup_dir).as_posix()
                except ValueError as exc:
                    raise ProjectRunError(
                        f"Scientific input symlink is outside {config.setup_dir}: {logical}"
                    ) from exc
                symlink_records[logical_relative] = {
                    "logical_path": logical_relative,
                    "target_relative": target_relative,
                    "size": target.stat().st_size,
                    "sha256": hash_file(target),
                }
    digest = inventory_digest(inventory)
    if symlink_records:
        digest = hash_json(
            {
                "regular_inventory_sha256": digest,
                "symlinks": [symlink_records[key] for key in sorted(symlink_records)],
            }
        )
    return inventory, digest


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "file_inventory",
    "hash_json",
    "inventory_digest",
    "load_manifest",
    "project_run_manifest_path",
    "project_scientific_input_inventory",
    "recursive_files",
    "sha256_file",
    "workflow_manifest_path",
    "write_manifest_atomic",
]
