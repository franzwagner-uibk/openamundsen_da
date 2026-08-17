"""Immutable result values returned by the public Python interface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class WorkflowStatus(str, Enum):
    """Stable operation outcomes shared by API and JSON CLI output."""

    APPLIED = "applied"
    COMPLETED = "completed"
    PREVIEW = "preview"
    REUSED = "reused"


class ObservationProduct(str, Enum):
    """Supported public observation-preprocessing products."""

    SNOW_COVER = "snow-cover"
    WET_SNOW = "wet-snow"


@dataclass(frozen=True)
class ObservationPreprocessingResult:
    project_dir: Path
    product: ObservationProduct
    status: WorkflowStatus
    summary_path: Path
    diagnostic_paths: tuple[Path, ...]
    processed_count: int
    rejected_count: int
    manifest_path: Path


@dataclass(frozen=True)
class PreparationResult:
    setup_dir: Path
    project_dir: Path
    status: WorkflowStatus
    step_dirs: tuple[Path, ...]
    observation_paths: tuple[Path, ...]
    manifest_path: Path


@dataclass(frozen=True)
class RenderResult:
    project_dir: Path
    status: WorkflowStatus
    plot_paths: tuple[Path, ...]
    map_paths: tuple[Path, ...]
    report_paths: tuple[Path, ...]


@dataclass(frozen=True)
class CleanupFailure:
    path: Path
    error: str


@dataclass(frozen=True)
class CleanupResult:
    project_dir: Path
    status: WorkflowStatus
    applied: bool
    eligible_paths: tuple[Path, ...]
    deleted_paths: tuple[Path, ...]
    failures: tuple[CleanupFailure, ...]
    eligible_bytes: int
    freed_bytes: int
    eligible_count: int = 0
    deleted_count: int = 0


@dataclass(frozen=True)
class RunResult:
    setup_dir: Path
    project_dir: Path
    results_dir: Path
    status: WorkflowStatus
    manifest_path: Path
    step_dirs: tuple[Path, ...]
    completed_members: int
    skipped_members: int
    compact_output_path: Path
    benchmark_manifest_path: Path
    render_result: RenderResult
    cleanup_result: CleanupResult
    duration_seconds: float


__all__ = [
    "CleanupFailure",
    "CleanupResult",
    "ObservationPreprocessingResult",
    "ObservationProduct",
    "PreparationResult",
    "RenderResult",
    "RunResult",
    "WorkflowStatus",
]
