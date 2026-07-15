"""Shared stage-state recording for resumable subdomain workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from openamundsen_da.subdomain.manifest import SubdomainManifest


STAGE_NAMES = frozenset({"prepare", "run", "merge", "render", "cleanup"})
STAGE_STATUSES = frozenset({"pending", "running", "completed", "failed", "interrupted", "skipped"})


def record_stage(
    manifest: SubdomainManifest,
    stage: str,
    status: str,
    *,
    outputs: Iterable[Path] = (),
    error: str | None = None,
) -> None:
    """Record one stage transition on an in-memory manifest."""
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unknown subdomain stage: {stage}")
    if status not in STAGE_STATUSES:
        raise ValueError(f"Unknown subdomain stage status: {status}")

    entry: dict[str, object] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_paths = [str(Path(path).resolve()) for path in outputs]
    if output_paths:
        entry["outputs"] = output_paths
    if error:
        entry["error"] = str(error)
    manifest.stages[stage] = entry


def save_stage(
    manifest: SubdomainManifest,
    manifest_path: Path,
    stage: str,
    status: str,
    *,
    outputs: Iterable[Path] = (),
    error: str | None = None,
) -> None:
    """Record and atomically save one stage transition."""
    record_stage(manifest, stage, status, outputs=outputs, error=error)
    manifest.save(manifest_path)


def terminal_status(exc: BaseException) -> str:
    """Return the persisted terminal state for an escaping exception."""
    return "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"


__all__ = ["record_stage", "save_stage", "terminal_status"]
