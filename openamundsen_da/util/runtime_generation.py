"""Project-owned disposable runtime generations for compact retention."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from openamundsen_da.exceptions import CleanupSafetyError
from openamundsen_da.manifests import (
    file_inventory,
    hash_json,
    inventory_digest,
    load_manifest,
    workflow_manifest_path,
    write_manifest_atomic,
)


RUNTIME_GENERATION_SCHEMA_VERSION = 1
RUNTIME_GENERATION_MANIFEST = "runtime_generation"
RUNTIME_CONSUMER_VALIDATION_MANIFEST = "runtime_consumer_validation"
RUNTIME_LAYOUT = "generation_tree"
LEGACY_LAYOUT = "legacy_member_tree"
_RUNTIME_STATUSES = {"open", "sealed", "quarantined", "deleting", "complete"}
_RUNTIME_LAYOUT_CACHE: dict[Path, dict] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_generation_manifest_path(project_dir: str | Path) -> Path:
    """Return the durable project runtime-generation manifest path."""
    return workflow_manifest_path(Path(project_dir).resolve(), RUNTIME_GENERATION_MANIFEST)


def runtime_consumer_validation_path(project_dir: str | Path) -> Path:
    """Return durable compact-consumer validation evidence."""
    return workflow_manifest_path(
        Path(project_dir).resolve(),
        RUNTIME_CONSUMER_VALIDATION_MANIFEST,
    )


def _contained_relative(project_dir: Path, raw: object, *, purpose: str) -> Path:
    relative = Path(str(raw))
    if relative.is_absolute():
        raise CleanupSafetyError(f"{purpose} must be project-relative: {relative}")
    candidate = project_dir / relative
    resolved_parent = candidate.parent.resolve()
    try:
        resolved_parent.relative_to(project_dir)
    except ValueError as exc:
        raise CleanupSafetyError(f"{purpose} escapes the project: {relative}") from exc
    if candidate.is_symlink():
        raise CleanupSafetyError(f"{purpose} is a symlink: {candidate}")
    return candidate


def _validate_manifest(project_dir: Path, manifest: Mapping[str, object]) -> dict:
    if int(manifest.get("runtime_generation_schema_version", -1)) != RUNTIME_GENERATION_SCHEMA_VERSION:
        raise CleanupSafetyError(
            f"Unsupported runtime generation manifest: {runtime_generation_manifest_path(project_dir)}"
        )
    layout = str(manifest.get("layout", ""))
    if layout not in {RUNTIME_LAYOUT, LEGACY_LAYOUT}:
        raise CleanupSafetyError(f"Invalid runtime generation layout: {layout!r}")
    if Path(str(manifest.get("project_dir", ""))).resolve() != project_dir:
        raise CleanupSafetyError("Runtime generation project identity changed")
    if layout == LEGACY_LAYOUT:
        if manifest.get("runtime_root") is not None:
            raise CleanupSafetyError("Legacy runtime layout cannot declare a runtime root")
        return dict(manifest)
    generation = str(manifest.get("generation_id", "")).strip()
    if not generation:
        raise CleanupSafetyError("Runtime generation identity is missing")
    status = str(manifest.get("status", ""))
    if status not in _RUNTIME_STATUSES:
        raise CleanupSafetyError(f"Invalid runtime generation status: {status!r}")
    root = _contained_relative(
        project_dir,
        manifest.get("runtime_root"),
        purpose="runtime generation root",
    )
    expected = project_dir / ".openamundsen-da" / "runtime" / generation
    if root != expected:
        raise CleanupSafetyError(f"Runtime generation root is not canonical: {root}")
    quarantine_raw = manifest.get("quarantine_root")
    if quarantine_raw is not None:
        quarantine = _contained_relative(
            project_dir,
            quarantine_raw,
            purpose="runtime quarantine root",
        )
        expected_quarantine = project_dir / ".openamundsen-da" / "quarantine" / generation
        if quarantine != expected_quarantine:
            raise CleanupSafetyError(
                f"Runtime generation quarantine is not canonical: {quarantine}"
            )
    return dict(manifest)


def load_runtime_generation(project_dir: str | Path) -> dict | None:
    """Load and validate the project's runtime layout authority."""
    project_dir = Path(project_dir).resolve()
    path = runtime_generation_manifest_path(project_dir)
    manifest = load_manifest(path)
    if manifest is None:
        return None
    if not isinstance(manifest, Mapping):
        raise CleanupSafetyError(f"Runtime generation manifest root is invalid: {path}")
    return _validate_manifest(project_dir, manifest)


def _has_legacy_runtime_artifacts(project_dir: Path) -> bool:
    steps = project_dir / "steps"
    if not steps.is_dir():
        return False
    patterns = (
        "step_*/ensembles/*/*/meteo",
        "step_*/ensembles/*/*/results",
        "step_*/plots/forcing",
    )
    return any(path.exists() for pattern in patterns for path in steps.glob(pattern))


def ensure_runtime_generation(
    project_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Create or reuse one compact runtime layout after storage admission.

    A project that already contains the historical member-local layout stays
    explicitly on that layout. This keeps partial pre-v6 runs restartable and
    routes only fresh compact work into a generation tree.
    """
    project_dir = Path(project_dir).resolve()
    manifest = load_runtime_generation(project_dir)
    if manifest is not None:
        if manifest["layout"] == LEGACY_LAYOUT:
            _RUNTIME_LAYOUT_CACHE[project_dir] = dict(manifest)
            return manifest
        status = str(manifest["status"])
        if status != "complete" or not overwrite:
            _RUNTIME_LAYOUT_CACHE[project_dir] = dict(manifest)
            return manifest

    if manifest is None and _has_legacy_runtime_artifacts(project_dir):
        legacy = {
            "runtime_generation_schema_version": RUNTIME_GENERATION_SCHEMA_VERSION,
            "layout": LEGACY_LAYOUT,
            "status": "open",
            "project_dir": str(project_dir),
            "runtime_root": None,
            "created_at": _utc_now(),
        }
        write_manifest_atomic(runtime_generation_manifest_path(project_dir), legacy)
        _RUNTIME_LAYOUT_CACHE[project_dir] = dict(legacy)
        return legacy

    generation_number = int(manifest.get("generation", 0)) + 1 if manifest else 1
    generation_id = f"generation-{generation_number:04d}-{uuid4().hex}"
    runtime_root = project_dir / ".openamundsen-da" / "runtime" / generation_id
    runtime_root.mkdir(parents=True, exist_ok=False)
    root_stat = runtime_root.stat()
    created = {
        "runtime_generation_schema_version": RUNTIME_GENERATION_SCHEMA_VERSION,
        "layout": RUNTIME_LAYOUT,
        "generation": generation_number,
        "generation_id": generation_id,
        "status": "open",
        "project_dir": str(project_dir),
        "runtime_root": runtime_root.relative_to(project_dir).as_posix(),
        "quarantine_root": None,
        "root_device": int(root_stat.st_dev),
        "root_inode": int(root_stat.st_ino),
        "step_accounting": {},
        "rolling_removed_paths": [],
        "rolling_removed_bytes": 0,
        "rolling_removed_files": 0,
        "created_at": _utc_now(),
    }
    created["identity_sha256"] = hash_json(
        {
            "generation": generation_number,
            "generation_id": generation_id,
            "project_dir": str(project_dir),
            "runtime_root": created["runtime_root"],
            "root_device": created["root_device"],
            "root_inode": created["root_inode"],
        }
    )
    write_manifest_atomic(runtime_generation_manifest_path(project_dir), created)
    validated = _validate_manifest(project_dir, created)
    _RUNTIME_LAYOUT_CACHE[project_dir] = dict(validated)
    return validated


def _find_runtime_project(path: Path) -> tuple[Path, dict] | None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        cached = _RUNTIME_LAYOUT_CACHE.get(candidate)
        if cached is not None:
            return candidate, dict(cached)
        manifest_path = workflow_manifest_path(candidate, RUNTIME_GENERATION_MANIFEST)
        if not manifest_path.is_file():
            continue
        manifest = load_runtime_generation(candidate)
        if manifest is not None:
            _RUNTIME_LAYOUT_CACHE[candidate] = dict(manifest)
            return candidate.resolve(), manifest
    return None


def mapped_runtime_path(owner_dir: str | Path, artifact_dir: str) -> Path | None:
    """Map an owned member/step directory below the active runtime tree."""
    owner = Path(owner_dir).absolute()
    found = _find_runtime_project(owner)
    if found is None:
        return None
    project_dir, manifest = found
    if manifest["layout"] != RUNTIME_LAYOUT:
        return None
    try:
        relative = owner.relative_to(project_dir)
    except ValueError as exc:
        raise CleanupSafetyError(f"Runtime artifact owner escapes the project: {owner}") from exc
    root = _contained_relative(
        project_dir,
        manifest["runtime_root"],
        purpose="runtime generation root",
    )
    mapped = root / relative / artifact_dir
    try:
        mapped.parent.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise CleanupSafetyError(f"Runtime artifact path escapes its generation: {mapped}") from exc
    return mapped


def runtime_generation_root(project_dir: str | Path) -> Path | None:
    """Return the active generation root, or ``None`` for legacy layout."""
    project_dir = Path(project_dir).resolve()
    manifest = load_runtime_generation(project_dir)
    if manifest is None or manifest["layout"] != RUNTIME_LAYOUT:
        return None
    return _contained_relative(
        project_dir,
        manifest["runtime_root"],
        purpose="runtime generation root",
    )


def record_runtime_step_accounting(
    project_dir: str | Path,
    *,
    step_name: str,
    component_bytes: Mapping[str, int],
    file_counts: Mapping[str, int],
) -> None:
    """Persist monotonic producer accounting for one runtime step."""
    project_dir = Path(project_dir).resolve()
    manifest = load_runtime_generation(project_dir)
    if manifest is None or manifest["layout"] != RUNTIME_LAYOUT:
        return
    if manifest["status"] != "open":
        raise CleanupSafetyError("Runtime accounting cannot change after generation seal")
    disposable_components = {
        "forcing_bytes",
        "member_grid_bytes",
        "point_bytes",
        "restart_baseline_bytes",
        "derived_forcing_plot_bytes",
    }
    recorded_bytes = sum(
        max(0, int(component_bytes.get(component, 0)))
        for component in disposable_components
    )
    recorded_files = sum(
        max(0, int(file_counts.get(component, 0)))
        for component in disposable_components
    )
    accounting = dict(manifest.get("step_accounting") or {})
    previous = dict(accounting.get(str(step_name)) or {})
    accounting[str(step_name)] = {
        "bytes": max(int(previous.get("bytes", 0)), recorded_bytes),
        "files": max(int(previous.get("files", 0)), recorded_files),
        "updated_at": _utc_now(),
    }
    manifest["step_accounting"] = accounting
    write_manifest_atomic(runtime_generation_manifest_path(project_dir), manifest)
    _RUNTIME_LAYOUT_CACHE[project_dir] = dict(manifest)


def runtime_accounted_totals(project_dir: str | Path) -> tuple[int, int]:
    """Return accounted live runtime bytes and files after rolling removal."""
    manifest = load_runtime_generation(project_dir)
    if manifest is None or manifest["layout"] != RUNTIME_LAYOUT:
        return 0, 0
    accounting = manifest.get("step_accounting") or {}
    total_bytes = sum(int(row.get("bytes", 0)) for row in accounting.values())
    total_files = sum(int(row.get("files", 0)) for row in accounting.values())
    return (
        max(0, total_bytes - int(manifest.get("rolling_removed_bytes", 0))),
        max(0, total_files - int(manifest.get("rolling_removed_files", 0))),
    )


def record_runtime_consumer_validation(
    project_dir: str | Path,
    *,
    consumers: list[Path],
) -> Path | None:
    """Bind validated compact consumers to the active runtime generation."""
    project_dir = Path(project_dir).resolve()
    runtime = load_runtime_generation(project_dir)
    if runtime is None or runtime["layout"] != RUNTIME_LAYOUT:
        return None
    inventory = file_inventory(root=project_dir, files=consumers)
    if len(inventory) != len(consumers) or not consumers:
        raise CleanupSafetyError("Runtime consumer validation evidence is incomplete")
    payload = {
        "contract": "compact-runtime-consumers-v1",
        "status": "success",
        "project_dir": str(project_dir),
        "runtime_generation_id": str(runtime["generation_id"]),
        "consumer_inventory": inventory,
        "consumer_inventory_sha256": inventory_digest(inventory),
        "validated_at": _utc_now(),
    }
    return write_manifest_atomic(runtime_consumer_validation_path(project_dir), payload)


def runtime_consumer_validation_evidence(
    project_dir: str | Path,
) -> tuple[list[Path], list[dict]]:
    """Load generation-bound retained-consumer evidence without rehashing it."""
    project_dir = Path(project_dir).resolve()
    runtime = load_runtime_generation(project_dir)
    path = runtime_consumer_validation_path(project_dir)
    payload = load_manifest(path)
    if runtime is None or runtime["layout"] != RUNTIME_LAYOUT:
        raise CleanupSafetyError("Project has no generation-owned compact runtime")
    if (
        payload is None
        or payload.get("contract") != "compact-runtime-consumers-v1"
        or payload.get("status") != "success"
        or Path(str(payload.get("project_dir", ""))).resolve() != project_dir
        or payload.get("runtime_generation_id") != runtime.get("generation_id")
    ):
        raise CleanupSafetyError(f"Runtime consumer validation is missing or stale: {path}")
    inventory = list(payload.get("consumer_inventory") or [])
    if inventory_digest(inventory) != str(payload.get("consumer_inventory_sha256", "")):
        raise CleanupSafetyError("Runtime consumer validation identity changed")
    consumers = [project_dir / str(row.get("path", "")) for row in inventory]
    return consumers, inventory


def validate_runtime_consumer_validation(project_dir: str | Path) -> list[Path]:
    """Revalidate retained consumers without reopening disposable raw sources."""
    project_dir = Path(project_dir).resolve()
    consumers, inventory = runtime_consumer_validation_evidence(project_dir)
    actual = file_inventory(root=project_dir, files=consumers)
    if actual != inventory:
        raise CleanupSafetyError("A validated runtime consumer changed before cleanup")
    return consumers


def update_runtime_generation(
    project_dir: str | Path,
    *,
    status: str,
    quarantine_root: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict:
    """Durably advance runtime-generation lifecycle metadata."""
    project_dir = Path(project_dir).resolve()
    manifest = load_runtime_generation(project_dir)
    if manifest is None or manifest["layout"] != RUNTIME_LAYOUT:
        raise CleanupSafetyError("Project has no generation-owned runtime tree")
    if status not in _RUNTIME_STATUSES:
        raise CleanupSafetyError(f"Invalid runtime generation status: {status!r}")
    manifest["status"] = status
    manifest["quarantine_root"] = quarantine_root
    manifest["updated_at"] = _utc_now()
    if extra:
        manifest.update(dict(extra))
    write_manifest_atomic(runtime_generation_manifest_path(project_dir), manifest)
    validated = _validate_manifest(project_dir, manifest)
    _RUNTIME_LAYOUT_CACHE[project_dir] = dict(validated)
    return validated


def record_runtime_rolling_removal(
    project_dir: str | Path,
    *,
    path_sizes: Mapping[Path, int],
) -> None:
    """Record known predecessor files removed during compact propagation."""
    project_dir = Path(project_dir).resolve()
    manifest = load_runtime_generation(project_dir)
    if manifest is None or manifest["layout"] != RUNTIME_LAYOUT:
        return
    root = runtime_generation_root(project_dir)
    if root is None:
        raise CleanupSafetyError("Runtime generation root is unavailable")
    recorded = {
        str(item.get("path")): int(item.get("bytes", 0))
        for item in manifest.get("rolling_removed_paths") or []
        if isinstance(item, Mapping) and item.get("path")
    }
    for path, size in path_sizes.items():
        try:
            relative = Path(path).resolve(strict=False).relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise CleanupSafetyError(
                f"Rolling cleanup path escapes the runtime generation: {path}"
            ) from exc
        recorded[relative] = max(recorded.get(relative, 0), max(0, int(size)))
    manifest["rolling_removed_paths"] = [
        {"path": relative, "bytes": recorded[relative]}
        for relative in sorted(recorded)
    ]
    manifest["rolling_removed_files"] = len(recorded)
    manifest["rolling_removed_bytes"] = sum(recorded.values())
    manifest["updated_at"] = _utc_now()
    write_manifest_atomic(runtime_generation_manifest_path(project_dir), manifest)
    _RUNTIME_LAYOUT_CACHE[project_dir] = dict(manifest)


__all__ = [
    "LEGACY_LAYOUT",
    "RUNTIME_GENERATION_MANIFEST",
    "RUNTIME_CONSUMER_VALIDATION_MANIFEST",
    "RUNTIME_GENERATION_SCHEMA_VERSION",
    "RUNTIME_LAYOUT",
    "ensure_runtime_generation",
    "load_runtime_generation",
    "mapped_runtime_path",
    "record_runtime_step_accounting",
    "record_runtime_consumer_validation",
    "record_runtime_rolling_removal",
    "runtime_accounted_totals",
    "runtime_consumer_validation_evidence",
    "runtime_generation_manifest_path",
    "runtime_consumer_validation_path",
    "runtime_generation_root",
    "update_runtime_generation",
    "validate_runtime_consumer_validation",
]
