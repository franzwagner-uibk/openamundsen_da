"""Atomic, path-contained ledger for deliberately cleaned artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
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


RETENTION_SCHEMA_VERSION = 5
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
    if version in {3, 4}:
        batches = list(ledger.get("batches") or [])
        if version == 3:
            for batch in batches:
                batch["generation"] = 1
        legacy_generations = (
            list(ledger.get("generations") or [])
            if version == 4
            else ([{"generation": 1, "status": ledger.get("status")}] if batches else [])
        )
        active_generation = ledger.get(
            "active_generation",
            int(legacy_generations[-1]["generation"]) if legacy_generations else None,
        )
        migrated_generations: list[dict] = []
        for legacy in legacy_generations:
            generation = int(legacy.get("generation", -1))
            generation_batches = [
                batch for batch in batches if int(batch.get("generation", -2)) == generation
            ]
            if not generation_batches or any(
                batch.get("status") != "complete" for batch in generation_batches
            ):
                raise CleanupSafetyError(
                    "Cannot safely resume a planned legacy retention generation; "
                    f"inspect or restore {path} before cleanup"
                )
            source_inventory = _merge_inventory_rows(
                *(list(batch.get("inventory") or []) for batch in generation_batches)
            )
            consumer_inventory = _merge_inventory_rows(
                *(list(batch.get("consumer_inventory") or []) for batch in generation_batches)
            )
            producer_inventory = _merge_inventory_rows(
                *(
                    list(batch.get("producer_manifest_inventory") or [])
                    for batch in generation_batches
                )
            )
            if producer_inventory:
                producer_payload: object = {"file_inventory": producer_inventory}
            else:
                payloads = [
                    batch.get("producer_manifest_payload")
                    for batch in generation_batches
                ]
                producer_payload = (
                    payloads[0]
                    if len(payloads) == 1
                    else {"batch_payloads": payloads}
                )
            identity = _generation_identity(
                generation=generation,
                source_inventory=source_inventory,
                consumer_inventory=consumer_inventory,
                producer_payload=producer_payload,
            )
            migrated_generations.append(
                {
                    **legacy,
                    "generation": generation,
                    "status": str(legacy.get("status") or "complete"),
                    "identity_sha256": hash_json(identity),
                    "source_inventory": source_inventory,
                    "source_inventory_sha256": inventory_digest(source_inventory),
                    "consumer_inventory": consumer_inventory,
                    "consumer_inventory_sha256": inventory_digest(consumer_inventory),
                    "producer_manifest_inventory": producer_inventory,
                    "producer_manifest_payload": producer_payload,
                    "producer_digest": hash_json(producer_payload),
                    "started_at": legacy.get("started_at", f"legacy-v{version}"),
                    "completed_at": legacy.get("completed_at", f"legacy-v{version}"),
                }
            )
        ledger.update(
            {
                "retention_schema_version": RETENTION_SCHEMA_VERSION,
                "active_generation": active_generation,
                "generations": migrated_generations,
                "batches": batches,
            }
        )
    elif version != RETENTION_SCHEMA_VERSION:
        raise CleanupSafetyError(f"Unsupported retention manifest: {path}")
    if not isinstance(ledger.get("batches"), list):
        raise CleanupSafetyError(f"Invalid retention batches in {path}")
    if not isinstance(ledger.get("generations"), list):
        raise CleanupSafetyError(f"Invalid retention generations in {path}")
    try:
        generation_numbers = [
            int(record["generation"])
            for record in ledger["generations"]
            if isinstance(record, dict)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise CleanupSafetyError(f"Invalid retention generation identity in {path}") from exc
    if len(generation_numbers) != len(ledger["generations"]) or len(
        set(generation_numbers)
    ) != len(generation_numbers):
        raise CleanupSafetyError(f"Invalid or duplicate retention generations in {path}")
    active_generation = ledger.get("active_generation")
    if generation_numbers:
        if active_generation not in generation_numbers:
            raise CleanupSafetyError(f"Retention manifest has no valid active generation: {path}")
    elif active_generation is not None or ledger["batches"]:
        raise CleanupSafetyError(f"Retention manifest generation structure is invalid: {path}")
    unknown_batches = [
        batch
        for batch in ledger["batches"]
        if not isinstance(batch, dict) or batch.get("generation") not in generation_numbers
    ]
    if unknown_batches:
        raise CleanupSafetyError(f"Retention batch references an unknown generation: {path}")
    return ledger


def _merge_inventory_rows(*inventories: list[dict]) -> list[dict]:
    """Merge deterministic inventories and reject conflicting path identities."""
    rows: dict[str, dict] = {}
    for inventory in inventories:
        for raw in inventory:
            if not isinstance(raw, dict):
                raise CleanupSafetyError("Retention inventory row is invalid")
            row = dict(raw)
            rel = str(row.get("path", ""))
            if not rel:
                raise CleanupSafetyError("Retention inventory path is missing")
            if rel in rows and rows[rel] != row:
                raise CleanupSafetyError(
                    f"Retention inventories disagree for the same path: {rel}"
                )
            rows[rel] = row
    return [rows[key] for key in sorted(rows)]


def _generation_identity(
    *,
    generation: int,
    source_inventory: list[dict],
    consumer_inventory: list[dict],
    producer_payload: object,
) -> dict:
    return {
        "generation": int(generation),
        "source_inventory": source_inventory,
        "consumer_inventory": consumer_inventory,
        "producer_manifest_payload": producer_payload,
    }


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


def planned_retention_generation_dependencies(
    project_dir: str | Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return every recorded consumer and producer path for planned resume."""
    project_dir = Path(project_dir).resolve()
    record = _active_generation_record(_load_ledger(project_dir))
    if record is None or record.get("status") != "planned":
        return (), ()
    consumers = tuple(
        project_dir / str(row.get("path", ""))
        for row in record.get("consumer_inventory") or []
    )
    producers = tuple(
        project_dir / str(row.get("path", ""))
        for row in record.get("producer_manifest_inventory") or []
    )
    return consumers, producers


def start_retention_generation(
    project_dir: str | Path,
    *,
    source_paths: Iterable[Path],
    retained_consumers: Iterable[Path],
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
    consumers = _contained_files(project_dir, retained_consumers)
    producers = _contained_files(project_dir, producer_manifests)
    if not sources:
        raise CleanupSafetyError("Retention generation requires at least one source")
    if not consumers:
        raise CleanupSafetyError("Retention generation requires retained consumers")
    if not producers and producer_manifest_payload is None:
        raise CleanupSafetyError("Retention generation requires producer evidence")
    source_inventory = file_inventory(
        root=project_dir,
        files=[path for path in sources if path.is_file()],
    )
    consumer_inventory = file_inventory(root=project_dir, files=consumers)
    producer_inventory = file_inventory(root=project_dir, files=producers)
    if len(consumer_inventory) != len(consumers):
        raise CleanupSafetyError("A generation retained consumer is missing or invalid")
    if len(producer_inventory) != len(producers):
        raise CleanupSafetyError("A generation producer manifest is missing or invalid")
    producer_payload = (
        {"file_inventory": producer_inventory}
        if producer_inventory
        else {"manifest_payload": producer_manifest_payload}
    )
    ledger = _load_ledger(project_dir)
    active = _active_generation_record(ledger)
    if active is not None and active.get("status") == "planned":
        _validate_planned_generation_resume(
            project_dir,
            ledger=ledger,
            generation_record=active,
            current_sources=source_inventory,
            current_consumers=consumer_inventory,
            current_producer_payload=producer_payload,
        )
        return int(active["generation"])
    if len(source_inventory) != len(sources):
        raise CleanupSafetyError("A generation source is missing or invalid")

    generation_number = max(
        [int(item.get("generation", 0)) for item in ledger.get("generations") or []],
        default=0,
    ) + 1
    identity = _generation_identity(
        generation=generation_number,
        source_inventory=source_inventory,
        consumer_inventory=consumer_inventory,
        producer_payload=producer_payload,
    )
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
        "source_inventory": source_inventory,
        "source_inventory_sha256": inventory_digest(source_inventory),
        "consumer_inventory": consumer_inventory,
        "consumer_inventory_sha256": inventory_digest(consumer_inventory),
        "producer_manifest_inventory": producer_inventory,
        "producer_manifest_payload": producer_payload,
        "producer_digest": hash_json(producer_payload),
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
    _validate_generation_for_completion(
        project_dir,
        ledger=ledger,
        generation_record=active,
    )
    active["status"] = "complete"
    active["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_ledger(project_dir, ledger)


def _unlink_path(path: Path) -> None:
    path.unlink()


def _resolve_recorded_path(project_dir: Path, rel: str, *, purpose: str) -> Path:
    raw = project_dir / rel
    path = raw.resolve()
    try:
        path.relative_to(project_dir)
    except ValueError as exc:
        raise CleanupSafetyError(f"Recorded {purpose} escapes project root: {rel}") from exc
    if raw.is_symlink():
        raise CleanupSafetyError(f"Recorded {purpose} is a symlink: {rel}")
    return path


def _validate_recorded_file(project_dir: Path, batch: dict, rel: str) -> Path:
    """Bind a planned deletion to the exact generation recorded in its inventory."""
    path = _resolve_recorded_path(project_dir, rel, purpose="cleanup path")
    if not path.exists():
        return path
    if not path.is_file():
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
        path = _resolve_recorded_path(project_dir, rel, purpose=purpose)
        if not path.is_file():
            raise CleanupSafetyError(f"Recorded {purpose} is missing or invalid: {rel}")
        if (
            path.stat().st_size != int(row.get("size", -1))
            or sha256_file(path) != str(row.get("sha256", ""))
        ):
            raise CleanupSafetyError(f"Recorded {purpose} changed after planning: {rel}")


def _sha256_fd(fd: int, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash an open descriptor without changing its shared file offset."""
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(fd, chunk_size, offset):
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _stat_identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return the mutation-sensitive identity used by a live cleanup guard."""
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(stat.S_IFMT(file_stat.st_mode)),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_ctime_ns),
    )


@dataclass(frozen=True)
class _GuardedConsumer:
    rel: str
    path: Path
    fd: int
    identity: tuple[int, int, int, int, int, int]
    sha256: str


class _RetainedConsumerGuard:
    """Pin byte-validated consumers during one destructive cleanup batch."""

    def __init__(self, project_dir: Path, inventory: list[dict]) -> None:
        if not inventory:
            raise CleanupSafetyError("Retention plan has no recorded retained consumer")
        self._project_dir = project_dir
        self._files: list[_GuardedConsumer] = []
        try:
            for row in inventory:
                self._files.append(self._open(row))
        except BaseException:
            self.close()
            raise

    def _open(self, row: dict) -> _GuardedConsumer:
        rel = str(row.get("path", ""))
        path = _resolve_recorded_path(
            self._project_dir,
            rel,
            purpose="retained consumer",
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise CleanupSafetyError(
                f"Recorded retained consumer is missing or invalid: {rel}"
            ) from exc
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise CleanupSafetyError(
                    f"Recorded retained consumer is missing or invalid: {rel}"
                )
            try:
                digest = _sha256_fd(fd)
            except OSError as exc:
                raise CleanupSafetyError(
                    f"Cannot read recorded retained consumer: {rel}"
                ) from exc
            after = os.fstat(fd)
            identity = _stat_identity(before)
            if (
                identity != _stat_identity(after)
                or int(row.get("size", -1)) != before.st_size
                or str(row.get("sha256", "")) != digest
            ):
                raise CleanupSafetyError(
                    f"Recorded retained consumer changed after planning: {rel}"
                )
            guarded = _GuardedConsumer(
                rel=rel,
                path=path,
                fd=fd,
                identity=identity,
                sha256=digest,
            )
            self._validate_one(guarded)
            return guarded
        except BaseException:
            os.close(fd)
            raise

    def _validate_one(self, guarded: _GuardedConsumer) -> None:
        try:
            current_path = _resolve_recorded_path(
                self._project_dir,
                guarded.rel,
                purpose="retained consumer",
            )
            descriptor_stat = os.fstat(guarded.fd)
            path_stat = os.stat(current_path, follow_symlinks=False)
        except (CleanupSafetyError, OSError) as exc:
            raise CleanupSafetyError(
                f"Recorded retained consumer changed after planning: {guarded.rel}"
            ) from exc
        if (
            current_path != guarded.path
            or not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or _stat_identity(descriptor_stat) != guarded.identity
            or _stat_identity(path_stat) != guarded.identity
        ):
            raise CleanupSafetyError(
                f"Recorded retained consumer changed after planning: {guarded.rel}"
            )

    def validate_fast(self) -> None:
        """Reject replacement or metadata-visible mutation without rereading content."""
        for guarded in self._files:
            self._validate_one(guarded)

    def validate_full(self) -> None:
        """Revalidate byte identity before the batch is accepted."""
        self.validate_fast()
        for guarded in self._files:
            try:
                digest = _sha256_fd(guarded.fd)
            except OSError as exc:
                raise CleanupSafetyError(
                    f"Cannot read recorded retained consumer: {guarded.rel}"
                ) from exc
            if digest != guarded.sha256:
                raise CleanupSafetyError(
                    f"Recorded retained consumer changed after planning: {guarded.rel}"
                )
        self.validate_fast()

    def close(self) -> None:
        while self._files:
            guarded = self._files.pop()
            os.close(guarded.fd)

    def __enter__(self) -> _RetainedConsumerGuard:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def _validate_generation_record_identity(generation_record: dict) -> None:
    """Require the stored inventories to reproduce the generation identity."""
    try:
        generation = int(generation_record["generation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CleanupSafetyError("Retention generation identity is invalid") from exc
    source_inventory = list(generation_record.get("source_inventory") or [])
    consumer_inventory = list(generation_record.get("consumer_inventory") or [])
    producer_payload = generation_record.get("producer_manifest_payload")
    if not source_inventory or not consumer_inventory or not isinstance(producer_payload, dict):
        raise CleanupSafetyError(
            f"Retention generation {generation} has incomplete identity evidence"
        )
    if inventory_digest(source_inventory) != str(
        generation_record.get("source_inventory_sha256", "")
    ):
        raise CleanupSafetyError(f"Retention generation {generation} source identity changed")
    if inventory_digest(consumer_inventory) != str(
        generation_record.get("consumer_inventory_sha256", "")
    ):
        raise CleanupSafetyError(f"Retention generation {generation} consumer identity changed")
    if hash_json(producer_payload) != str(generation_record.get("producer_digest", "")):
        raise CleanupSafetyError(f"Retention generation {generation} producer identity changed")
    identity = _generation_identity(
        generation=generation,
        source_inventory=source_inventory,
        consumer_inventory=consumer_inventory,
        producer_payload=producer_payload,
    )
    if hash_json(identity) != str(generation_record.get("identity_sha256", "")):
        raise CleanupSafetyError(f"Retention generation {generation} identity changed")


def _active_generation_batches(ledger: dict, generation: int) -> list[dict]:
    return [
        batch
        for batch in ledger.get("batches") or []
        if int(batch.get("generation", -1)) == int(generation)
    ]


def _validate_planned_generation_resume(
    project_dir: Path,
    *,
    ledger: dict,
    generation_record: dict,
    current_sources: list[dict],
    current_consumers: list[dict],
    current_producer_payload: object,
) -> None:
    """Accept only the exact surviving portion of one recorded generation."""
    _validate_generation_record_identity(generation_record)
    generation = int(generation_record["generation"])
    recorded_sources = list(generation_record.get("source_inventory") or [])
    recorded_consumers = list(generation_record.get("consumer_inventory") or [])
    recorded_producer_payload = generation_record.get("producer_manifest_payload")
    _validate_inventory_files(
        project_dir,
        inventory=recorded_consumers,
        purpose="generation retained consumer",
    )
    recorded_producers = list(generation_record.get("producer_manifest_inventory") or [])
    if recorded_producers:
        if recorded_producer_payload != {"file_inventory": recorded_producers}:
            raise CleanupSafetyError(
                f"Retention generation {generation} producer inventory is inconsistent"
            )
        _validate_inventory_files(
            project_dir,
            inventory=recorded_producers,
            purpose="generation producer manifest",
        )

    surviving_sources = [
        row
        for row in recorded_sources
        if _resolve_recorded_path(
            project_dir,
            str(row.get("path", "")),
            purpose="generation source",
        ).is_file()
    ]
    if current_sources != surviving_sources:
        raise CleanupSafetyError(
            f"Retention generation {generation} source identity does not match resume"
        )
    for row in surviving_sources:
        _validate_inventory_files(
            project_dir,
            inventory=[row],
            purpose="generation source",
        )
    authorized_paths = {
        str(rel)
        for batch in _active_generation_batches(ledger, generation)
        for rel in batch.get("paths", [])
    }
    missing_unplanned = [
        str(row.get("path", ""))
        for row in recorded_sources
        if not _resolve_recorded_path(
            project_dir,
            str(row.get("path", "")),
            purpose="generation source",
        ).exists()
        and str(row.get("path", "")) not in authorized_paths
    ]
    if missing_unplanned:
        raise CleanupSafetyError(
            "Retention generation source disappeared before it was planned: "
            f"{missing_unplanned[0]}"
        )
    if current_consumers != recorded_consumers:
        raise CleanupSafetyError(
            f"Retention generation {generation} consumer identity does not match resume"
        )
    if isinstance(current_producer_payload, dict) and "file_inventory" in current_producer_payload:
        current_producers = list(current_producer_payload.get("file_inventory") or [])
        if current_producers != recorded_producers:
            raise CleanupSafetyError(
                f"Retention generation {generation} producer identity does not match resume"
            )
    elif current_producer_payload != recorded_producer_payload:
        raise CleanupSafetyError(
            f"Retention generation {generation} producer identity does not match resume"
        )


def _validate_batch_evidence(project_dir: Path, batch: dict) -> None:
    """Revalidate one complete batch's immutable evidence and removed sources."""
    inventory = list(batch.get("inventory") or [])
    if not inventory or inventory_digest(inventory) != str(batch.get("inventory_sha256", "")):
        raise CleanupSafetyError(
            f"Retention batch source identity changed: {batch.get('batch_id', '<unknown>')}"
        )
    recreated = []
    for row in inventory:
        rel = str(row.get("path", ""))
        if _resolve_recorded_path(
            project_dir,
            rel,
            purpose="batch source",
        ).exists():
            recreated.append(rel)
    if recreated:
        raise CleanupSafetyError(
            f"Retention batch source was recreated after deletion: {recreated[0]}"
        )
    consumers = list(batch.get("consumer_inventory") or [])
    if inventory_digest(consumers) != str(batch.get("consumer_inventory_sha256", "")):
        raise CleanupSafetyError(
            f"Retention batch consumer identity changed: {batch.get('batch_id', '<unknown>')}"
        )
    _validate_inventory_files(
        project_dir,
        inventory=consumers,
        purpose="retained consumer",
    )
    producers = list(batch.get("producer_manifest_inventory") or [])
    payload = batch.get("producer_manifest_payload")
    if not isinstance(payload, dict) or hash_json(payload) != str(batch.get("producer_digest", "")):
        raise CleanupSafetyError(
            f"Retention batch producer identity changed: {batch.get('batch_id', '<unknown>')}"
        )
    if producers:
        if payload != {"file_inventory": producers}:
            raise CleanupSafetyError(
                f"Retention batch producer inventory changed: {batch.get('batch_id', '<unknown>')}"
            )
        _validate_inventory_files(
            project_dir,
            inventory=producers,
            purpose="producer manifest",
        )


def _validate_generation_for_completion(
    project_dir: Path,
    *,
    ledger: dict,
    generation_record: dict,
) -> None:
    """Revalidate all evidence and coverage before accepting a generation."""
    _validate_generation_record_identity(generation_record)
    generation = int(generation_record["generation"])
    batches = _active_generation_batches(ledger, generation)
    if not batches or any(batch.get("status") != "complete" for batch in batches):
        raise CleanupSafetyError(
            f"Retention generation {generation} has incomplete or missing batches"
        )
    for batch in batches:
        _validate_batch_evidence(project_dir, batch)
    recorded_sources = {
        str(row.get("path", "")): row
        for row in generation_record.get("source_inventory") or []
    }
    batch_sources = _merge_inventory_rows(
        *(list(batch.get("inventory") or []) for batch in batches)
    )
    if batch_sources != [recorded_sources[key] for key in sorted(recorded_sources)]:
        raise CleanupSafetyError(
            f"Retention generation {generation} batches do not cover its exact source identity"
        )
    _validate_inventory_files(
        project_dir,
        inventory=list(generation_record.get("consumer_inventory") or []),
        purpose="generation retained consumer",
    )
    producers = list(generation_record.get("producer_manifest_inventory") or [])
    if producers:
        if generation_record.get("producer_manifest_payload") != {
            "file_inventory": producers
        }:
            raise CleanupSafetyError(
                f"Retention generation {generation} producer inventory is inconsistent"
            )
        _validate_inventory_files(
            project_dir,
            inventory=producers,
            purpose="generation producer manifest",
        )


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
            if active is None:
                raise CleanupSafetyError("Completed retention batch has no active generation")
            _validate_generation_for_completion(
                project_dir,
                ledger=ledger,
                generation_record=active,
            )
            return idempotent
        generation = start_retention_generation(
            project_dir,
            source_paths=candidates,
            retained_consumers=consumers,
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
    with _RetainedConsumerGuard(
        project_dir,
        list(batch.get("consumer_inventory") or []),
    ) as consumer_guard:
        for rel in batch["paths"]:
            # The full byte identity was verified when the guard opened. The
            # descriptor and path metadata checks remain O(1) per unlink while
            # still stopping replacement or ordinary in-place mutation before
            # another source is removed.
            consumer_guard.validate_fast()
            path = _validate_recorded_file(project_dir, batch, str(rel))
            if not path.exists():
                continue
            try:
                _unlink_path(path)
            except OSError as exc:
                failures.append(f"{rel}: {exc}")
        consumer_guard.validate_full()
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
    _validate_generation_record_identity(active)
    if require_complete and active.get("status") != "complete":
        raise CleanupSafetyError(
            f"Retention generation is not complete: {active.get('generation')}"
        )
    if require_complete:
        _validate_generation_for_completion(
            project_dir,
            ledger=ledger,
            generation_record=active,
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
    "planned_retention_generation_dependencies",
    "planned_retention_paths",
    "reconcile_retention_ledger",
    "retention_manifest_path",
    "start_retention_generation",
    "validate_retained_consumers",
]
