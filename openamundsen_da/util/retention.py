"""Atomic, path-contained ledger for deliberately cleaned artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from openamundsen_da.exceptions import CleanupSafetyError
from openamundsen_da.manifests import file_inventory, inventory_digest, load_manifest, write_manifest_atomic


RETENTION_SCHEMA_VERSION = 1
RETENTION_MANIFEST = "retention_manifest.json"


def retention_manifest_path(project_dir: str | Path) -> Path:
    return Path(project_dir).resolve() / "results" / RETENTION_MANIFEST


def _contained_files(project_dir: Path, paths: Iterable[Path]) -> list[Path]:
    contained: dict[str, Path] = {}
    for raw in paths:
        path = Path(raw)
        resolved = path.resolve()
        try:
            rel = resolved.relative_to(project_dir)
        except ValueError as exc:
            raise CleanupSafetyError(f"Cleanup path escapes project root: {path}") from exc
        if path.is_symlink():
            raise CleanupSafetyError(f"Cleanup refuses symlinks: {path}")
        if resolved.exists() and not resolved.is_file():
            raise CleanupSafetyError(f"Cleanup candidate is not a regular file: {path}")
        contained[rel.as_posix()] = resolved
    return [contained[key] for key in sorted(contained)]


def _empty_manifest() -> dict:
    return {
        "retention_schema_version": RETENTION_SCHEMA_VERSION,
        "status": "active",
        "batches": [],
    }


def _load_ledger(project_dir: Path) -> dict:
    path = retention_manifest_path(project_dir)
    ledger = load_manifest(path)
    if ledger is None:
        return _empty_manifest()
    if ledger.get("retention_schema_version") != RETENTION_SCHEMA_VERSION:
        raise CleanupSafetyError(f"Unsupported retention manifest: {path}")
    if not isinstance(ledger.get("batches"), list):
        raise CleanupSafetyError(f"Invalid retention batches in {path}")
    return ledger


def _write_ledger(project_dir: Path, ledger: dict) -> None:
    """Persist a validated ledger and its aggregate lifecycle state."""
    batches = ledger.get("batches") or []
    ledger["status"] = (
        "complete"
        if batches and all(batch.get("status") == "complete" for batch in batches)
        else "active"
    )
    write_manifest_atomic(retention_manifest_path(project_dir), ledger)


def _unlink_path(path: Path) -> None:
    path.unlink()


def apply_retention_batch(
    project_dir: str | Path,
    *,
    artifact_class: str,
    paths: Iterable[Path],
    final_consumer: str,
    regeneration_recipe: str,
    producer_digest: str | None = None,
) -> dict:
    """Record, delete and complete one idempotent cleanup batch."""
    project_dir = Path(project_dir).resolve()
    candidates = _contained_files(project_dir, paths)
    ledger_path = retention_manifest_path(project_dir)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(project_dir)
    relative_paths = [path.relative_to(project_dir).as_posix() for path in candidates]

    # A retry may find a planned batch with some paths already absent. Reuse
    # that exact plan rather than creating a second provenance record.
    existing = next(
        (
            batch
            for batch in ledger["batches"]
            if batch.get("artifact_class") == artifact_class
            and (
                batch.get("paths") == relative_paths
                or (
                    batch.get("status") == "planned"
                    and set(relative_paths)
                    == {
                        str(rel)
                        for rel in batch.get("paths", [])
                        if (project_dir / str(rel)).exists()
                    }
                )
            )
            and batch.get("status") in {"planned", "complete"}
        ),
        None,
    )
    if existing is not None and existing.get("status") == "complete":
        return existing

    if existing is None:
        inventory = file_inventory(root=project_dir, files=[path for path in candidates if path.is_file()])
        source_digest = inventory_digest(inventory)
        batch = {
            "batch_id": f"{len(ledger['batches']) + 1:04d}",
            "artifact_class": str(artifact_class),
            "status": "planned",
            "path_count": len(relative_paths),
            "bytes": sum(int(row["size"]) for row in inventory),
            "paths": relative_paths,
            "inventory": inventory,
            "inventory_sha256": source_digest,
            "producer_digest": producer_digest or source_digest,
            "final_consumer": str(final_consumer),
            "regeneration_recipe": str(regeneration_recipe),
            "planned_at": datetime.now(timezone.utc).isoformat(),
        }
        ledger["batches"].append(batch)
        _write_ledger(project_dir, ledger)
    else:
        batch = existing

    failures: list[str] = []
    for rel in batch["paths"]:
        path = (project_dir / rel).resolve()
        try:
            path.relative_to(project_dir)
        except ValueError as exc:
            raise CleanupSafetyError(f"Recorded cleanup path escapes project root: {rel}") from exc
        if not path.exists():
            continue
        try:
            _unlink_path(path)
        except OSError as exc:
            failures.append(f"{rel}: {exc}")
    if failures:
        batch["failures"] = failures
        _write_ledger(project_dir, ledger)
        raise CleanupSafetyError(
            f"Retention cleanup failed for {len(failures)} path(s); see {ledger_path}"
        )
    batch["status"] = "complete"
    batch["completed_at"] = datetime.now(timezone.utc).isoformat()
    batch.pop("failures", None)
    _write_ledger(project_dir, ledger)
    return batch


def reconcile_retention_ledger(project_dir: str | Path) -> tuple[str, ...]:
    """Complete interrupted batches whose recorded files are all absent.

    A batch with remaining files stays planned and is resumed by the next
    class-specific cleanup call. This function never deletes a path.
    """
    project_dir = Path(project_dir).resolve()
    ledger = _load_ledger(project_dir)
    completed: list[str] = []
    changed = False
    for batch in ledger["batches"]:
        if batch.get("status") != "planned":
            continue
        remaining = [
            str(rel)
            for rel in batch.get("paths", [])
            if (project_dir / str(rel)).exists()
        ]
        if remaining:
            continue
        batch["status"] = "complete"
        batch["completed_at"] = datetime.now(timezone.utc).isoformat()
        batch.pop("failures", None)
        completed.append(str(batch.get("batch_id", "")))
        changed = True
    if changed:
        _write_ledger(project_dir, ledger)
    return tuple(completed)


def completed_retention_paths(project_dir: str | Path) -> set[str]:
    """Return project-relative paths deliberately removed by complete batches."""
    project_dir = Path(project_dir).resolve()
    ledger = _load_ledger(project_dir)
    return {
        str(rel)
        for batch in ledger["batches"]
        if batch.get("status") == "complete"
        for rel in batch.get("paths", [])
    }


def planned_retention_paths(
    project_dir: str | Path,
    *,
    artifact_class: str,
) -> tuple[Path, ...]:
    """Return existing paths from the matching interrupted cleanup batch."""
    project_dir = Path(project_dir).resolve()
    ledger = _load_ledger(project_dir)
    return tuple(
        project_dir / str(rel)
        for batch in ledger["batches"]
        if batch.get("status") == "planned"
        and batch.get("artifact_class") == artifact_class
        for rel in batch.get("paths", [])
        if (project_dir / str(rel)).is_file()
    )


__all__ = [
    "RETENTION_MANIFEST",
    "RETENTION_SCHEMA_VERSION",
    "apply_retention_batch",
    "completed_retention_paths",
    "planned_retention_paths",
    "reconcile_retention_ledger",
    "retention_manifest_path",
]
