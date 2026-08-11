"""Atomic, path-contained ledger for deliberately cleaned artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from openamundsen_da.exceptions import CleanupSafetyError
from openamundsen_da.manifests import (
    file_inventory,
    inventory_digest,
    hash_json,
    load_manifest,
    sha256_file,
    write_manifest_atomic,
)


RETENTION_SCHEMA_VERSION = 4
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
        "active_generation": None,
        "generations": [],
        "batches": [],
    }


def _load_ledger(project_dir: Path) -> dict:
    path = retention_manifest_path(project_dir)
    ledger = load_manifest(path)
    if ledger is None:
        return _empty_manifest()
    version = ledger.get("retention_schema_version")
    if version == 3:
        batches = list(ledger.get("batches") or [])
        generation_status = (
            "complete"
            if batches and all(batch.get("status") == "complete" for batch in batches)
            else "planned"
        )
        for batch in batches:
            batch["generation"] = 1
        ledger.update(
            {
                "retention_schema_version": RETENTION_SCHEMA_VERSION,
                "active_generation": 1 if batches else None,
                "generations": (
                    [
                        {
                            "generation": 1,
                            "status": generation_status,
                            "identity_sha256": "legacy-v3",
                            "started_at": "legacy-v3",
                        }
                    ]
                    if batches
                    else []
                ),
                "batches": batches,
            }
        )
    elif version != RETENTION_SCHEMA_VERSION:
        raise CleanupSafetyError(f"Unsupported retention manifest: {path}")
    if not isinstance(ledger.get("batches"), list):
        raise CleanupSafetyError(f"Invalid retention batches in {path}")
    if not isinstance(ledger.get("generations"), list):
        raise CleanupSafetyError(f"Invalid retention generations in {path}")
    return ledger


def _write_ledger(project_dir: Path, ledger: dict) -> None:
    """Persist a validated ledger and its aggregate lifecycle state."""
    active = ledger.get("active_generation")
    generation = next(
        (
            item
            for item in ledger.get("generations") or []
            if item.get("generation") == active
        ),
        None,
    )
    ledger["status"] = (
        "complete" if generation is not None and generation.get("status") == "complete" else "active"
    )
    write_manifest_atomic(retention_manifest_path(project_dir), ledger)


def _active_generation_record(ledger: dict) -> dict | None:
    active = ledger.get("active_generation")
    return next(
        (
            generation
            for generation in ledger.get("generations") or []
            if generation.get("generation") == active
        ),
        None,
    )


def active_retention_generation(project_dir: str | Path) -> tuple[int, str] | None:
    """Return the active generation number and status, if any."""
    ledger = _load_ledger(Path(project_dir).resolve())
    record = _active_generation_record(ledger)
    if record is None:
        return None
    return int(record["generation"]), str(record.get("status", ""))


def start_retention_generation(
    project_dir: str | Path,
    *,
    source_paths: Iterable[Path],
    producer_manifests: Iterable[Path] = (),
    producer_manifest_payload: object | None = None,
) -> int:
    """Start or resume one explicit cleanup generation atomically.

    A planned generation must be resumed before another can start. Starting a
    verified new generation supersedes only the prior generation's validation
    authority; its batches and inventories remain as immutable audit history.
    """
    project_dir = Path(project_dir).resolve()
    sources = _contained_files(project_dir, source_paths)
    producers = _contained_files(project_dir, producer_manifests)
    if not producers and producer_manifest_payload is None:
        raise CleanupSafetyError("Retention generation requires producer evidence")
    ledger = _load_ledger(project_dir)
    active = _active_generation_record(ledger)
    if active is not None and active.get("status") == "planned":
        return int(active["generation"])

    source_inventory = file_inventory(
        root=project_dir,
        files=[path for path in sources if path.is_file()],
    )
    producer_inventory = file_inventory(root=project_dir, files=producers)
    if len(producer_inventory) != len(producers):
        raise CleanupSafetyError("A generation producer manifest is missing or invalid")
    producer_payload = (
        {"file_inventory": producer_inventory}
        if producer_inventory
        else {"manifest_payload": producer_manifest_payload}
    )
    identity = {
        "source_inventory_sha256": inventory_digest(source_inventory),
        "producer_digest": hash_json(producer_payload),
    }
    generation_number = max(
        [int(item.get("generation", 0)) for item in ledger.get("generations") or []],
        default=0,
    ) + 1
    if active is not None:
        active["status"] = "superseded"
        active["superseded_by_generation"] = generation_number
        active["superseded_at"] = datetime.now(timezone.utc).isoformat()
        for batch in ledger.get("batches") or []:
            if batch.get("generation") == active.get("generation"):
                batch["superseded_by_generation"] = generation_number
    record = {
        "generation": generation_number,
        "status": "planned",
        "identity_sha256": hash_json(identity),
        "source_inventory_sha256": identity["source_inventory_sha256"],
        "producer_digest": identity["producer_digest"],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    ledger["generations"].append(record)
    ledger["active_generation"] = generation_number
    _write_ledger(project_dir, ledger)
    return generation_number


def complete_retention_generation(
    project_dir: str | Path,
    *,
    generation: int,
) -> None:
    """Mark the active generation complete after every batch completes."""
    project_dir = Path(project_dir).resolve()
    ledger = _load_ledger(project_dir)
    active = _active_generation_record(ledger)
    if active is None or int(active.get("generation", -1)) != int(generation):
        raise CleanupSafetyError(f"Retention generation is no longer active: {generation}")
    batches = [
        batch
        for batch in ledger.get("batches") or []
        if int(batch.get("generation", -1)) == int(generation)
    ]
    if not batches or any(batch.get("status") != "complete" for batch in batches):
        raise CleanupSafetyError(
            f"Retention generation {generation} has incomplete or missing batches"
        )
    active["status"] = "complete"
    active["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_ledger(project_dir, ledger)


def _unlink_path(path: Path) -> None:
    path.unlink()


def _validate_recorded_file(project_dir: Path, batch: dict, rel: str) -> Path:
    """Bind a planned deletion to the exact generation recorded in its inventory."""
    path = (project_dir / rel).resolve()
    try:
        path.relative_to(project_dir)
    except ValueError as exc:
        raise CleanupSafetyError(f"Recorded cleanup path escapes project root: {rel}") from exc
    if not path.exists():
        return path
    if not path.is_file() or path.is_symlink():
        raise CleanupSafetyError(f"Recorded cleanup candidate is not a regular file: {rel}")
    inventory = {
        str(row.get("path")): row
        for row in batch.get("inventory", [])
        if isinstance(row, dict)
    }
    row = inventory.get(rel)
    if row is None:
        raise CleanupSafetyError(f"Retention plan has no inventory entry for {rel}")
    size = path.stat().st_size
    if size != int(row.get("size", -1)) or sha256_file(path) != str(row.get("sha256", "")):
        raise CleanupSafetyError(
            f"Retention candidate changed after planning; refusing recreated or modified file: {rel}"
        )
    return path


def _validate_inventory_files(
    project_dir: Path,
    *,
    inventory: list[dict],
    purpose: str,
) -> None:
    """Require every recorded dependency to remain byte-identical."""
    if not inventory:
        raise CleanupSafetyError(f"Retention plan has no recorded {purpose}")
    for row in inventory:
        rel = str(row.get("path", ""))
        path = (project_dir / rel).resolve()
        try:
            path.relative_to(project_dir)
        except ValueError as exc:
            raise CleanupSafetyError(f"Recorded {purpose} escapes project root: {rel}") from exc
        if not path.is_file() or path.is_symlink():
            raise CleanupSafetyError(f"Recorded {purpose} is missing or invalid: {rel}")
        if (
            path.stat().st_size != int(row.get("size", -1))
            or sha256_file(path) != str(row.get("sha256", ""))
        ):
            raise CleanupSafetyError(f"Recorded {purpose} changed after planning: {rel}")


def apply_retention_batch(
    project_dir: str | Path,
    *,
    artifact_class: str,
    paths: Iterable[Path],
    final_consumer: str,
    regeneration_recipe: str,
    retained_consumers: Iterable[Path],
    producer_manifests: Iterable[Path] = (),
    producer_manifest_payload: object | None = None,
    generation: int | None = None,
) -> dict:
    """Record, delete and complete one idempotent cleanup batch."""
    project_dir = Path(project_dir).resolve()
    candidates = _contained_files(project_dir, paths)
    consumers = _contained_files(project_dir, retained_consumers)
    producers = _contained_files(project_dir, producer_manifests)
    if not consumers:
        raise CleanupSafetyError("Retention cleanup requires at least one retained consumer")
    if not producers and producer_manifest_payload is None:
        raise CleanupSafetyError("Retention cleanup requires producer manifest evidence")
    ledger_path = retention_manifest_path(project_dir)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(project_dir)
    relative_paths = [path.relative_to(project_dir).as_posix() for path in candidates]

    automatic_generation = generation is None
    if generation is None:
        active = _active_generation_record(ledger)
        active_number = None if active is None else int(active["generation"])
        idempotent = next(
            (
                batch
                for batch in reversed(ledger["batches"])
                if batch.get("generation") == active_number
                and batch.get("artifact_class") == artifact_class
                and batch.get("paths") == relative_paths
                and batch.get("status") == "complete"
                and not any((project_dir / rel).exists() for rel in batch.get("paths", []))
            ),
            None,
        )
        if idempotent is not None:
            return idempotent
        generation = start_retention_generation(
            project_dir,
            source_paths=candidates,
            producer_manifests=producers,
            producer_manifest_payload=producer_manifest_payload,
        )
        ledger = _load_ledger(project_dir)
    active = _active_generation_record(ledger)
    if (
        active is None
        or int(active.get("generation", -1)) != int(generation)
        or active.get("status") != "planned"
    ):
        raise CleanupSafetyError(f"Retention generation is not planned and active: {generation}")

    # A retry may find a planned batch with some paths already absent. Reuse
    # that exact plan rather than creating a second provenance record.
    matching = [
        batch
        for batch in ledger["batches"]
        if batch.get("generation") == generation
        and batch.get("artifact_class") == artifact_class
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
    ]
    # A planned retry always belongs to the recorded generation, even if an
    # older completed generation used the same relative names.
    existing = next(
        (batch for batch in matching if batch.get("status") == "planned"),
        matching[-1] if matching else None,
    )
    if existing is not None and existing.get("status") == "complete":
        # An absent generation is idempotently complete. A path recreated at
        # the same name is a new generation and receives its own batch.
        if not any((project_dir / rel).exists() for rel in existing.get("paths", [])):
            return existing
        existing = None

    if existing is None:
        inventory = file_inventory(root=project_dir, files=[path for path in candidates if path.is_file()])
        source_digest = inventory_digest(inventory)
        consumer_inventory = file_inventory(root=project_dir, files=consumers)
        producer_inventory = file_inventory(root=project_dir, files=producers)
        if len(consumer_inventory) != len(consumers):
            raise CleanupSafetyError("A retained consumer is missing or invalid")
        if len(producer_inventory) != len(producers):
            raise CleanupSafetyError("A producer manifest is missing or invalid")
        producer_payload = (
            {"file_inventory": producer_inventory}
            if producer_inventory
            else {"manifest_payload": producer_manifest_payload}
        )
        batch = {
            "batch_id": f"{len(ledger['batches']) + 1:04d}",
            "generation": generation,
            "artifact_class": str(artifact_class),
            "status": "planned",
            "path_count": len(relative_paths),
            "bytes": sum(int(row["size"]) for row in inventory),
            "paths": relative_paths,
            "inventory": inventory,
            "inventory_sha256": source_digest,
            "consumer_inventory": consumer_inventory,
            "consumer_inventory_sha256": inventory_digest(consumer_inventory),
            "producer_manifest_inventory": producer_inventory,
            "producer_manifest_payload": producer_payload,
            "producer_digest": hash_json(producer_payload),
            "final_consumer": str(final_consumer),
            "regeneration_recipe": str(regeneration_recipe),
            "planned_at": datetime.now(timezone.utc).isoformat(),
        }
        ledger["batches"].append(batch)
        _write_ledger(project_dir, ledger)
    else:
        batch = existing

    producer_inventory = list(batch.get("producer_manifest_inventory") or [])
    if producer_inventory:
        _validate_inventory_files(
            project_dir,
            inventory=producer_inventory,
            purpose="producer manifest",
        )
    elif producer_manifest_payload is None or hash_json(
        {"manifest_payload": producer_manifest_payload}
    ) != str(batch.get("producer_digest", "")):
        raise CleanupSafetyError("Producer manifest payload changed after planning")

    failures: list[str] = []
    for rel in batch["paths"]:
        # This is deliberately repeated before every unlink. If a process was
        # interrupted after deleting an earlier source, a corrupt or replaced
        # retained consumer stops the resumed batch before any further loss.
        _validate_inventory_files(
            project_dir,
            inventory=list(batch.get("consumer_inventory") or []),
            purpose="retained consumer",
        )
        path = _validate_recorded_file(project_dir, batch, str(rel))
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
    if automatic_generation:
        complete_retention_generation(project_dir, generation=int(generation))
        ledger = _load_ledger(project_dir)
        batch = next(
            item
            for item in ledger["batches"]
            if item.get("batch_id") == batch.get("batch_id")
        )
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
    active = ledger.get("active_generation")
    for batch in ledger["batches"]:
        if batch.get("generation") != active:
            continue
        if batch.get("status") != "planned":
            continue
        remaining = [
            str(rel)
            for rel in batch.get("paths", [])
            if (project_dir / str(rel)).exists()
        ]
        if remaining:
            continue
        _validate_inventory_files(
            project_dir,
            inventory=list(batch.get("consumer_inventory") or []),
            purpose="retained consumer",
        )
        producer_inventory = list(batch.get("producer_manifest_inventory") or [])
        if producer_inventory:
            _validate_inventory_files(
                project_dir,
                inventory=producer_inventory,
                purpose="producer manifest",
            )
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
        if not (project_dir / str(rel)).exists()
    }


def validate_retained_consumers(
    project_dir: str | Path,
    *,
    require_complete: bool = False,
) -> tuple[str, ...]:
    """Validate every consumer and producer bound to the retention ledger.

    This is the resume-side counterpart to the validation performed before
    every deletion.  It lets a coordinator accept an already finalized leaf
    without recreating the raw artifacts that the ledger deliberately removed.
    """
    project_dir = Path(project_dir).resolve()
    ledger = _load_ledger(project_dir)
    active = _active_generation_record(ledger)
    if active is None:
        return ()
    if require_complete and active.get("status") != "complete":
        raise CleanupSafetyError(
            f"Retention generation is not complete: {active.get('generation')}"
        )
    batch_ids: list[str] = []
    for batch in ledger["batches"]:
        if batch.get("generation") != active.get("generation"):
            continue
        status = str(batch.get("status", ""))
        if require_complete and status != "complete":
            raise CleanupSafetyError(
                "Retention batch is not complete: "
                f"{batch.get('batch_id', '<unknown>')} ({status or 'missing status'})"
            )
        _validate_inventory_files(
            project_dir,
            inventory=list(batch.get("consumer_inventory") or []),
            purpose="retained consumer",
        )
        producer_inventory = list(batch.get("producer_manifest_inventory") or [])
        if producer_inventory:
            _validate_inventory_files(
                project_dir,
                inventory=producer_inventory,
                purpose="producer manifest",
            )
        batch_ids.append(str(batch.get("batch_id", "")))
    return tuple(batch_ids)


def planned_retention_paths(
    project_dir: str | Path,
    *,
    artifact_class: str,
) -> tuple[Path, ...]:
    """Return existing paths from the matching interrupted cleanup batch."""
    project_dir = Path(project_dir).resolve()
    ledger = _load_ledger(project_dir)
    active = ledger.get("active_generation")
    paths = tuple(
        _validate_recorded_file(project_dir, batch, str(rel))
        for batch in ledger["batches"]
        if batch.get("generation") == active
        and batch.get("status") == "planned"
        and batch.get("artifact_class") == artifact_class
        for rel in batch.get("paths", [])
        if (project_dir / str(rel)).is_file()
    )
    return paths


__all__ = [
    "RETENTION_MANIFEST",
    "RETENTION_SCHEMA_VERSION",
    "active_retention_generation",
    "apply_retention_batch",
    "complete_retention_generation",
    "completed_retention_paths",
    "planned_retention_paths",
    "reconcile_retention_ledger",
    "retention_manifest_path",
    "start_retention_generation",
    "validate_retained_consumers",
]
