"""Durable, coordinator-owned incremental storage admission.

The expensive storage estimator remains the source of conservative planning
bounds.  This module turns one such plan into a small mutable ledger so normal
step boundaries need only update a fixed-size accounting record and inspect the
current filesystem usage once.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from loguru import logger

from openamundsen_da.manifests import hash_json, sha256_file, write_manifest_atomic
from openamundsen_da.io.paths import (
    find_project_yaml,
    find_setup_yaml,
    list_member_dirs,
    list_steps_sorted,
    open_loop_dir,
)
from openamundsen_da.util.storage_budget import (
    DiskBudgetSnapshot,
    ProjectStorageEstimate,
    StorageReservationProject,
    check_step_admission,
    estimate_coordinated_storage_reserve,
    estimate_project_storage_components,
)


STORAGE_RESERVATION_SCHEMA_VERSION = 1
STORAGE_RESERVATION_RELATIVE_PATH = Path("results/storage/storage_reservation.json")
STORAGE_ADMISSION_REQUEST_TIMEOUT_SECONDS = 1.0
STORAGE_ADMISSION_REQUEST_ATTEMPTS = 3
STORAGE_ADMISSION_REQUEST_HISTORY_LIMIT = 64

_ESTIMATE_COMPONENTS = (
    "forcing_bytes",
    "member_grid_bytes",
    "point_bytes",
    "restart_baseline_bytes",
    "restart_transition_bytes",
    "compact_timeseries_bytes",
    "compact_grid_bytes",
    "map_support_bytes",
    "derived_forcing_plot_bytes",
    "retained_diagnostics_bytes",
)

# These classes are materialized once per propagated step.  Compact exports and
# parent products are released only by explicit lifecycle transitions.
STEP_MATERIALIZATION_COMPONENTS = (
    "forcing_bytes",
    "member_grid_bytes",
    "point_bytes",
    "restart_baseline_bytes",
    "derived_forcing_plot_bytes",
    "retained_diagnostics_bytes",
)


def accounting_summary_from_inventory(
    *,
    completed_step: str,
    inventory: list[Mapping[str, object]],
    source: str,
) -> StorageAccountingSummary:
    """Classify one producer inventory into fixed storage components."""
    materialized = {name: 0 for name in STEP_MATERIALIZATION_COMPONENTS}
    counts = {name: 0 for name in STEP_MATERIALIZATION_COMPONENTS}
    for item in inventory:
        relative = str(item.get("path") or "").replace("\\", "/")
        size = int(item.get("size") or 0)
        if not relative or size < 0:
            raise ValueError("Producer storage inventory contains an invalid file entry")
        name = Path(relative).name
        parts = Path(relative).parts
        if "meteo" in parts and name != "stations.csv":
            component = "forcing_bytes"
        elif name.startswith("output_grids") and name.endswith(".nc"):
            component = "member_grid_bytes"
        elif name.startswith("point_") and name.endswith(".csv"):
            component = "point_bytes"
        elif name.endswith((".pickle.gz", ".pickle", ".pkl.gz", ".pkl")):
            component = "restart_baseline_bytes"
        elif "plots" in parts and "forcing" in parts:
            component = "derived_forcing_plot_bytes"
        else:
            component = "retained_diagnostics_bytes"
        materialized[component] += size
        counts[component] += 1
    return StorageAccountingSummary(
        completed_step=completed_step,
        materialized_bytes=materialized,
        observed_bytes=materialized,
        file_counts=counts,
        source=source,
    )


def accounting_summary_delta(
    *,
    before: StorageAccountingSummary,
    after: StorageAccountingSummary,
    source: str,
) -> StorageAccountingSummary:
    """Report only producer-local growth while retaining gross observations."""
    if before.completed_step != after.completed_step:
        raise ValueError("Storage accounting delta requires one completed step")
    materialized = {
        component: max(
            0,
            int(after.observed_bytes.get(component, 0))
            - int(before.observed_bytes.get(component, 0)),
        )
        for component in STEP_MATERIALIZATION_COMPONENTS
    }
    return StorageAccountingSummary(
        completed_step=after.completed_step,
        materialized_bytes=materialized,
        observed_bytes=dict(after.observed_bytes),
        file_counts=dict(after.file_counts),
        source=source,
    )


def reused_accounting_summary(
    summary: StorageAccountingSummary,
    *,
    source: str,
) -> StorageAccountingSummary:
    """Preserve gross observations without releasing preflight-credited bytes."""
    return StorageAccountingSummary(
        completed_step=summary.completed_step,
        materialized_bytes={name: 0 for name in STEP_MATERIALIZATION_COMPONENTS},
        observed_bytes=dict(summary.observed_bytes),
        file_counts=dict(summary.file_counts),
        source=source,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def storage_reservation_path(project_dir: str | Path) -> Path:
    """Return the retained storage reservation ledger path."""
    return Path(project_dir).resolve() / STORAGE_RESERVATION_RELATIVE_PATH


def admit_storage_transition(
    project_dir: str | Path,
    *,
    phase: str,
    estimated_growth_bytes: int,
    allow_existing_step_drain: bool = True,
) -> DiskBudgetSnapshot:
    """Fully reconcile and durably admit a standalone parent lifecycle stage."""
    project_dir = Path(project_dir).resolve()
    started = time.perf_counter()
    snapshot = check_step_admission(
        project_dir,
        estimated_growth_bytes=estimated_growth_bytes,
        allow_existing_step_drain=allow_existing_step_drain,
    )
    duration = time.perf_counter() - started
    path = storage_reservation_path(project_dir)
    previous: dict[str, object] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, json.JSONDecodeError):
            previous = {}
    device = int(project_dir.stat().st_dev)
    if previous and int(previous.get("filesystem_device", -1)) != device:
        _archive_superseded_ledger(path, previous)
        previous = {}
    ledger = dict(previous)
    ledger.update(
        {
            "storage_reservation_schema_version": STORAGE_RESERVATION_SCHEMA_VERSION,
            "generation": str(ledger.get("generation") or uuid.uuid4()),
            "transition_sequence": int(ledger.get("transition_sequence", 0)) + 1,
            "filesystem_device": device,
            "filesystem_capacity_bytes": snapshot.total_bytes,
            "phase": str(phase),
            "status": "admitted",
            "updated_at": _utc_now(),
            "remaining_peak_growth_bytes": int(estimated_growth_bytes),
            "latest_filesystem_snapshot": {
                "path": str(snapshot.filesystem_path),
                "total_bytes": snapshot.total_bytes,
                "used_bytes": snapshot.used_bytes,
                "free_bytes": snapshot.free_bytes,
                "used_fraction": snapshot.used_fraction,
                "projected_used_fraction": snapshot.projected_used_fraction,
                "checked_at": _utc_now(),
            },
            "latest_projected_headroom_bytes": (
                snapshot.total_bytes
                - snapshot.used_bytes
                - snapshot.estimated_growth_bytes
                - snapshot.operational_reserve_bytes
            ),
            "full_estimate_count": int(ledger.get("full_estimate_count", 0)) + 1,
            "full_estimate_duration_seconds": float(
                ledger.get("full_estimate_duration_seconds", 0.0)
            ) + duration,
        }
    )
    write_manifest_atomic(path, ledger)
    return snapshot


@dataclass(frozen=True)
class StorageAccountingSummary:
    """Small producer-owned accounting evidence for one completed step."""

    completed_step: str
    materialized_bytes: Mapping[str, int]
    observed_bytes: Mapping[str, int] = field(default_factory=dict)
    file_counts: Mapping[str, int] = field(default_factory=dict)
    cleanup_freed_bytes: int = 0
    source: str = "producer"

    def __post_init__(self) -> None:
        completed_step = str(self.completed_step).strip()
        if not completed_step:
            raise ValueError("completed_step is required for storage accounting")
        materialized = {str(key): int(value) for key, value in self.materialized_bytes.items()}
        observed = {str(key): int(value) for key, value in self.observed_bytes.items()}
        counts = {str(key): int(value) for key, value in self.file_counts.items()}
        unknown = sorted(
            (set(materialized) | set(observed)) - set(STEP_MATERIALIZATION_COMPONENTS)
        )
        if unknown:
            raise ValueError(
                "Unknown storage accounting component(s): " + ", ".join(unknown)
            )
        if any(value < 0 for value in materialized.values()):
            raise ValueError("Materialized storage bytes must be non-negative")
        if any(value < 0 for value in observed.values()):
            raise ValueError("Observed storage bytes must be non-negative")
        if any(value < 0 for value in counts.values()):
            raise ValueError("Storage accounting file counts must be non-negative")
        if int(self.cleanup_freed_bytes) < 0:
            raise ValueError("cleanup_freed_bytes must be non-negative")
        object.__setattr__(self, "completed_step", completed_step)
        object.__setattr__(self, "materialized_bytes", materialized)
        object.__setattr__(self, "observed_bytes", observed or dict(materialized))
        object.__setattr__(self, "file_counts", counts)
        object.__setattr__(self, "cleanup_freed_bytes", int(self.cleanup_freed_bytes))

    def as_dict(self) -> dict[str, object]:
        return {
            "completed_step": self.completed_step,
            "materialized_bytes": dict(self.materialized_bytes),
            "observed_bytes": dict(self.observed_bytes),
            "file_counts": dict(self.file_counts),
            "cleanup_freed_bytes": self.cleanup_freed_bytes,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "StorageAccountingSummary":
        materialized = value.get("materialized_bytes")
        counts = value.get("file_counts") or {}
        observed = value.get("observed_bytes") or materialized
        if not isinstance(materialized, Mapping) or not isinstance(counts, Mapping):
            raise ValueError("Malformed storage accounting summary")
        return cls(
            completed_step=str(value.get("completed_step") or ""),
            materialized_bytes={str(key): int(item) for key, item in materialized.items()},
            observed_bytes={str(key): int(item) for key, item in observed.items()},
            file_counts={str(key): int(item) for key, item in counts.items()},
            cleanup_freed_bytes=int(value.get("cleanup_freed_bytes") or 0),
            source=str(value.get("source") or "producer"),
        )

    def merged(self, other: "StorageAccountingSummary") -> "StorageAccountingSummary":
        if self.completed_step != other.completed_step:
            raise ValueError(
                "Cannot merge storage summaries for different steps: "
                f"{self.completed_step!r} != {other.completed_step!r}"
            )
        materialized = {
            key: int(self.materialized_bytes.get(key, 0))
            + int(other.materialized_bytes.get(key, 0))
            for key in set(self.materialized_bytes) | set(other.materialized_bytes)
        }
        observed = {
            key: int(self.observed_bytes.get(key, 0))
            + int(other.observed_bytes.get(key, 0))
            for key in set(self.observed_bytes) | set(other.observed_bytes)
        }
        counts = {
            key: int(self.file_counts.get(key, 0)) + int(other.file_counts.get(key, 0))
            for key in set(self.file_counts) | set(other.file_counts)
        }
        return StorageAccountingSummary(
            completed_step=self.completed_step,
            materialized_bytes=materialized,
            observed_bytes=observed,
            file_counts=counts,
            cleanup_freed_bytes=self.cleanup_freed_bytes + other.cleanup_freed_bytes,
            source=f"{self.source}+{other.source}",
        )


@dataclass(frozen=True)
class StorageLeafPlan:
    leaf_id: str
    setup_dir: Path
    project_dir: Path
    step_names: tuple[str, ...]
    obligations: Mapping[str, int]
    step_obligations: Mapping[str, Mapping[str, int]]
    queued_retained_bytes: int
    identity: str
    grid_cell_count: int = 0

    @property
    def total_bytes(self) -> int:
        return sum(int(value) for value in self.obligations.values())


@dataclass(frozen=True)
class StoragePlan:
    root_project_dir: Path
    leaves: Mapping[str, StorageLeafPlan]
    waves: tuple[tuple[str, ...], ...]
    wave_growth_bytes: tuple[int, ...]
    outer_workers: int
    parent_finalization_reserve_bytes: int
    estimated_growth_bytes: int
    overwrite: bool
    filesystem_device: int
    filesystem_capacity_bytes: int
    identity: str
    estimate_duration_seconds: float


def _estimate_obligations(estimate: ProjectStorageEstimate) -> dict[str, int]:
    obligations = {name: int(getattr(estimate, name)) for name in _ESTIMATE_COMPONENTS}
    if any(value < 0 for value in obligations.values()):
        raise ValueError("Storage estimator returned a negative component obligation")
    return obligations


def _validate_partial_member_manifests(
    project: StorageReservationProject,
    step_names: tuple[str, ...],
) -> None:
    """Validate present member manifests without inferring step completion."""
    for step_name in step_names:
        step_dir = project.project_dir / "steps" / step_name
        expected_members = list_member_dirs(step_dir / "ensembles", "prior")
        expected_open_loop = open_loop_dir(step_dir)
        if expected_open_loop.is_dir():
            expected_members = [expected_open_loop, *expected_members]
        member_manifests = [
            member_dir / "results" / "member_run.json"
            for member_dir in expected_members
        ]
        statuses: list[str] = []
        for manifest_path in member_manifests:
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            status = str(payload.get("status") or "").lower()
            if manifest_path.is_file() and not status:
                raise RuntimeError(
                    f"Malformed authoritative member manifest: {manifest_path}"
                )
            accounting = payload.get("storage_accounting")
            if status == "success" and accounting is not None:
                if not isinstance(accounting, Mapping):
                    raise RuntimeError(
                        f"Malformed member storage accounting: {manifest_path}"
                    )
                try:
                    StorageAccountingSummary.from_dict(accounting)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Malformed member storage accounting: {manifest_path}"
                    ) from exc
            statuses.append(status)


def _step_obligations(
    *,
    project: StorageReservationProject,
    step_names: tuple[str, ...],
    obligations: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    """Record completion slots without inventing unproven per-step shares.

    The estimator provides exact conservative aggregate components, not exact
    heterogeneous step shares. The aggregate remains reserved until validated
    finalization, so step-local values deliberately stay zero in this increment.
    """
    del project
    del obligations
    return {
        step_name: {component: 0 for component in STEP_MATERIALIZATION_COMPONENTS}
        for step_name in step_names
    }


def _path_identity(paths: list[Path]) -> str:
    records: list[dict[str, object]] = []
    for path in sorted({Path(item).resolve() for item in paths}):
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": sha256_file(path),
            }
        )
    return hash_json(records)


def _project_identity(project: StorageReservationProject, step_names: tuple[str, ...]) -> str:
    inputs = [find_setup_yaml(project.setup_dir), find_project_yaml(project.project_dir)]
    for step_name in step_names:
        step_dir = project.project_dir / "steps" / step_name
        inputs.extend(sorted(step_dir.glob("*.yml")))
        inputs.extend(sorted(step_dir.glob("*.yaml")))
    meteo_dir = project.setup_dir / "meteo"
    if meteo_dir.is_dir():
        inputs.extend(
            path
            for path in sorted(meteo_dir.glob("*.csv"))
            if path.is_file() and not path.is_symlink()
        )
    return _path_identity(inputs)


def build_storage_plan(
    *,
    root_project_dir: str | Path,
    projects: tuple[StorageReservationProject, ...],
    outer_workers: int,
    parent_finalization_reserve_bytes: int = 0,
    overwrite: bool = False,
    leaf_ids: tuple[str, ...] | None = None,
    estimated_growth_override: int | None = None,
    waves: tuple[tuple[str, ...], ...] | None = None,
    queued_retained_by_id: Mapping[str, int] | None = None,
) -> StoragePlan:
    """Build one immutable conservative plan using the expensive estimator."""
    started = time.perf_counter()
    root_project_dir = Path(root_project_dir).resolve()
    if not projects:
        raise ValueError("At least one project is required for storage planning")
    if outer_workers < 1:
        raise ValueError("outer_workers must be positive")
    if parent_finalization_reserve_bytes < 0:
        raise ValueError("parent_finalization_reserve_bytes must be non-negative")
    if leaf_ids is None:
        leaf_ids = tuple(project.project_dir.resolve().name for project in projects)
    if len(leaf_ids) != len(projects) or len(set(leaf_ids)) != len(leaf_ids):
        raise ValueError("leaf_ids must uniquely identify every storage reservation project")
    if waves is None:
        waves = (leaf_ids,)
    flattened = tuple(leaf_id for wave in waves for leaf_id in wave)
    if flattened != leaf_ids or any(not wave for wave in waves):
        raise ValueError("waves must contain every leaf exactly once in leaf_ids order")
    queued_retained_by_id = dict(queued_retained_by_id or {})

    usage = os.statvfs(root_project_dir)
    capacity = int(usage.f_frsize * usage.f_blocks)
    device = int(root_project_dir.stat().st_dev)
    for project in projects:
        if int(project.project_dir.resolve().stat().st_dev) != device:
            raise ValueError(
                "Bounded storage admission requires every project to use the coordinator filesystem"
            )

    estimated_growth, estimates = estimate_coordinated_storage_reserve(
        projects,
        outer_workers=outer_workers,
        parent_finalization_reserve_bytes=parent_finalization_reserve_bytes,
        overwrite=overwrite,
    )
    leaves: dict[str, StorageLeafPlan] = {}
    for leaf_id, project in zip(leaf_ids, projects, strict=True):
        project_dir = project.project_dir.resolve()
        estimate = estimates.get(str(project_dir))
        if estimate is None:
            # A durably completed project contributes no future growth.
            obligations = {name: 0 for name in _ESTIMATE_COMPONENTS}
        else:
            obligations = _estimate_obligations(estimate)
        step_names = tuple(path.name for path in list_steps_sorted(project_dir))
        if not step_names:
            raise FileNotFoundError(f"Prepared project has no storage-plannable steps: {project_dir}")
        _validate_partial_member_manifests(project, step_names)
        leaves[leaf_id] = StorageLeafPlan(
            leaf_id=leaf_id,
            setup_dir=project.setup_dir.resolve(),
            project_dir=project_dir,
            step_names=step_names,
            obligations=obligations,
            step_obligations=_step_obligations(
                project=project,
                step_names=step_names,
                obligations=obligations,
            ),
            queued_retained_bytes=int(
                queued_retained_by_id.get(leaf_id, sum(obligations.values()))
            ),
            identity=_project_identity(project, step_names),
            grid_cell_count=int(project.grid_cell_count),
        )
    wave_growth_bytes = tuple(
        _wave_growth(
            leaves=leaves,
            waves=waves,
            wave_index=wave_index,
            parent_finalization_reserve_bytes=parent_finalization_reserve_bytes,
        )
        for wave_index in range(len(waves))
    )
    estimated_growth = max(wave_growth_bytes)
    if estimated_growth_override is not None:
        if int(estimated_growth_override) < estimated_growth:
            raise ValueError(
                "estimated_growth_override cannot weaken the coordinated estimator bound"
            )
        estimated_growth = int(estimated_growth_override)
    # This identity deliberately excludes estimator outputs.  Those outputs are
    # net-additional and therefore shrink as a run materializes files.  A
    # resumed coordinator validates the immutable run topology here, then keeps
    # using the original obligations persisted in its durable ledger.
    identity_payload = {
        "root_project_dir": str(root_project_dir),
        "device": device,
        "outer_workers": int(outer_workers),
        "waves": [list(wave) for wave in waves],
        "leaves": {
            leaf_id: {
                "setup_dir": str(leaf.setup_dir),
                "project_dir": str(leaf.project_dir),
                "steps": list(leaf.step_names),
                "identity": leaf.identity,
            }
            for leaf_id, leaf in leaves.items()
        },
    }
    return StoragePlan(
        root_project_dir=root_project_dir,
        leaves=leaves,
        waves=waves,
        wave_growth_bytes=wave_growth_bytes,
        outer_workers=int(outer_workers),
        parent_finalization_reserve_bytes=int(parent_finalization_reserve_bytes),
        estimated_growth_bytes=int(estimated_growth),
        overwrite=bool(overwrite),
        filesystem_device=device,
        filesystem_capacity_bytes=capacity,
        identity=hash_json(identity_payload),
        estimate_duration_seconds=time.perf_counter() - started,
    )


def _wave_growth(
    *,
    leaves: Mapping[str, StorageLeafPlan],
    waves: tuple[tuple[str, ...], ...],
    wave_index: int,
    parent_finalization_reserve_bytes: int,
) -> int:
    active = [leaves[leaf_id] for leaf_id in waves[wave_index]]
    active_non_transition = sum(
        sum(
            int(value)
            for name, value in leaf.obligations.items()
            if name != "restart_transition_bytes"
        )
        for leaf in active
    )
    active_transition = sum(
        int(leaf.obligations.get("restart_transition_bytes", 0))
        for leaf in active
    )
    retained_inactive = sum(
        leaves[leaf_id].queued_retained_bytes
        for index, wave in enumerate(waves)
        if index != wave_index
        for leaf_id in wave
    )
    return (
        active_non_transition
        + active_transition
        + retained_inactive
        + int(parent_finalization_reserve_bytes)
    )


def _archive_superseded_ledger(path: Path, ledger: Mapping[str, object]) -> None:
    generation = str(ledger.get("generation") or "unknown")
    archive = path.with_name(f"storage_reservation.{generation}.json")
    if archive.exists():
        archive = path.with_name(
            f"storage_reservation.{generation}.{uuid.uuid4().hex[:8]}.json"
        )
    write_manifest_atomic(archive, dict(ledger))


def _has_partial_run_evidence(plan: StoragePlan) -> bool:
    patterns = (
        "steps/step_*/assim/prior_forcing_manifest.json",
        "steps/step_*/assim/rejuvenate_manifest.json",
        "steps/step_*/ensembles/*/*/results/member_run.json",
    )
    return any(
        next(leaf.project_dir.glob(pattern), None) is not None
        for leaf in plan.leaves.values()
        for pattern in patterns
    ) or any(
        (leaf.setup_dir / "leaf_finalization_manifest.json").is_file()
        for leaf in plan.leaves.values()
    )


class StorageAdmissionCoordinator:
    """Sole mutable owner of one storage reservation ledger."""

    def __init__(
        self,
        plan: StoragePlan,
        *,
        ledger_path: Path | None = None,
        disk_usage: Callable[[Path], object] | None = None,
    ) -> None:
        self.plan = plan
        self.ledger_path = (
            Path(ledger_path).resolve()
            if ledger_path is not None
            else storage_reservation_path(plan.root_project_dir)
        )
        self._disk_usage = disk_usage
        self._lock = threading.Lock()
        self._ledger = self._initialize_ledger()

    @property
    def generation(self) -> str:
        return str(self._ledger["generation"])

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return copy.deepcopy(self._ledger)

    def active_leaf_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(str(item) for item in self._ledger["active_leaf_ids"])

    def reconcile_full_plan(self, fresh_plan: StoragePlan) -> None:
        """Raise durable obligations from a serialized phase-transition plan."""
        if fresh_plan.identity != self.plan.identity:
            raise RuntimeError(
                "Storage phase reconciliation changed immutable run identity"
            )
        with self._lock:
            ledger = copy.deepcopy(self._ledger)
            for leaf_id, fresh_leaf in fresh_plan.leaves.items():
                state = ledger["leaves"][leaf_id]
                if state["phase"] == "finalized":
                    continue
                for component, value in fresh_leaf.obligations.items():
                    state["planned_by_component"][component] = max(
                        int(state["planned_by_component"].get(component, 0)),
                        int(value),
                    )
                for step_name, components in fresh_leaf.step_obligations.items():
                    for component, value in components.items():
                        state["planned_by_step"][step_name][component] = max(
                            int(state["planned_by_step"][step_name].get(component, 0)),
                            int(value),
                        )
                if leaf_id in set(ledger["active_leaf_ids"]):
                    completed_index = int(state["last_completed_step_index"])
                    for component in STEP_MATERIALIZATION_COMPONENTS:
                        future = sum(
                            int(state["planned_by_step"][step].get(component, 0))
                            for step in state["step_names"][completed_index + 1 :]
                        )
                        state["remaining_by_component"][component] = max(
                            int(state["remaining_by_component"].get(component, 0)),
                            future,
                        )
                    for component in set(fresh_leaf.obligations) - set(
                        STEP_MATERIALIZATION_COMPONENTS
                    ):
                        state["remaining_by_component"][component] = max(
                            int(state["remaining_by_component"].get(component, 0)),
                            int(fresh_leaf.obligations.get(component, 0)),
                        )
                ledger["queued_retained_by_leaf"][leaf_id] = max(
                    int(ledger["queued_retained_by_leaf"].get(leaf_id, 0)),
                    int(fresh_leaf.queued_retained_bytes),
                )
            ledger["wave_growth_bytes"] = [
                max(int(old), int(fresh))
                for old, fresh in zip(
                    ledger["wave_growth_bytes"],
                    fresh_plan.wave_growth_bytes,
                    strict=True,
                )
            ]
            ledger["full_estimate_count"] = int(ledger["full_estimate_count"]) + 1
            ledger["full_estimate_duration_seconds"] = float(
                ledger["full_estimate_duration_seconds"]
            ) + float(fresh_plan.estimate_duration_seconds)
            ledger["updated_at"] = _utc_now()
            self._recompute_remaining_peak(ledger)
            write_manifest_atomic(self.ledger_path, ledger)
            self._ledger = ledger

    def reconcile_leaf_finalization(
        self,
        *,
        leaf_id: str,
        request_id: str | None = None,
        allow_existing_step_drain: bool = True,
    ) -> DiskBudgetSnapshot:
        """Run one serialized authoritative estimate before compact finalization."""
        request_id = request_id or f"{leaf_id}:reconcile_finalization"
        with self._lock:
            if leaf_id not in self.plan.leaves:
                raise KeyError(f"Unknown storage-admission leaf: {leaf_id}")
            ledger = copy.deepcopy(self._ledger)
            request = {
                "kind": "reconcile",
                "leaf_id": leaf_id,
                "phase": "project_finalizing",
                "request_id": request_id,
            }
            if not self._is_duplicate_request(
                ledger,
                request_id=request_id,
                request=request,
            ):
                leaf_plan = self.plan.leaves[leaf_id]
                started = time.perf_counter()
                estimate = estimate_project_storage_components(
                    setup_dir=leaf_plan.setup_dir,
                    project_dir=leaf_plan.project_dir,
                    overwrite=self.plan.overwrite,
                    grid_cell_count=(leaf_plan.grid_cell_count or None),
                )
                obligations = _estimate_obligations(estimate)
                state = ledger["leaves"][leaf_id]
                for component, value in obligations.items():
                    state["planned_by_component"][component] = max(
                        int(state["planned_by_component"].get(component, 0)),
                        int(value),
                    )
                    state["remaining_by_component"][component] = max(
                        int(state["remaining_by_component"].get(component, 0)),
                        int(value),
                    )
                ledger["full_estimate_count"] = int(ledger["full_estimate_count"]) + 1
                ledger["full_estimate_duration_seconds"] = float(
                    ledger["full_estimate_duration_seconds"]
                ) + (time.perf_counter() - started)
                ledger["phase"] = "finalization_reconciliation"
                self._recompute_remaining_peak(ledger)
            return self._check_and_commit(
                ledger,
                allow_existing_step_drain=allow_existing_step_drain,
                request_id=request_id,
                request=request,
            )

    def _new_ledger(self) -> dict[str, object]:
        active_ids = set(self.plan.waves[0])
        leaf_states = {
            leaf_id: {
                "project_dir": str(leaf.project_dir),
                "phase": "prepared",
                "step_names": list(leaf.step_names),
                "last_admitted_step_index": -1,
                "last_completed_step_index": -1,
                "remaining_by_component": (
                    dict(leaf.obligations)
                    if leaf_id in active_ids
                    else {name: 0 for name in leaf.obligations}
                ),
                "planned_by_component": dict(leaf.obligations),
                "planned_by_step": {
                    step: dict(components)
                    for step, components in leaf.step_obligations.items()
                },
                "cumulative_materialized_bytes": {
                    name: 0 for name in STEP_MATERIALIZATION_COMPONENTS
                },
                "observed_step_high_water_bytes": {
                    name: 0 for name in STEP_MATERIALIZATION_COMPONENTS
                },
                "last_accounting_summary": None,
            }
            for leaf_id, leaf in self.plan.leaves.items()
        }
        non_transition = sum(
            sum(
                int(value)
                for name, value in leaf.obligations.items()
                if name != "restart_transition_bytes"
            )
            for leaf_id, leaf in self.plan.leaves.items()
            if leaf_id in active_ids
        )
        active_transition = sum(
            sorted(
                (
                    int(leaf.obligations.get("restart_transition_bytes", 0))
                    for leaf_id, leaf in self.plan.leaves.items()
                    if leaf_id in active_ids
                ),
                reverse=True,
            )[: self.plan.outer_workers]
        )
        calculated_growth = (
            non_transition
            + active_transition
            + sum(
                self.plan.leaves[leaf_id].queued_retained_bytes
                for wave in self.plan.waves[1:]
                for leaf_id in wave
            )
            + self.plan.parent_finalization_reserve_bytes
        )
        return {
            "storage_reservation_schema_version": STORAGE_RESERVATION_SCHEMA_VERSION,
            "generation": str(uuid.uuid4()),
            "transition_sequence": 0,
            "plan_identity": self.plan.identity,
            "filesystem_device": self.plan.filesystem_device,
            "filesystem_capacity_bytes": self.plan.filesystem_capacity_bytes,
            "overwrite_generation": bool(self.plan.overwrite),
            "outer_workers": self.plan.outer_workers,
            "waves": [list(wave) for wave in self.plan.waves],
            "wave_growth_bytes": list(self.plan.wave_growth_bytes),
            "queued_retained_by_leaf": {
                leaf_id: int(leaf.queued_retained_bytes)
                for leaf_id, leaf in self.plan.leaves.items()
            },
            "active_wave_index": 0,
            "active_leaf_ids": list(self.plan.waves[0]),
            "phase": "preflight",
            "status": "planned",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "immutable_component_obligations": {
                leaf_id: dict(leaf.obligations)
                for leaf_id, leaf in self.plan.leaves.items()
            },
            "leaves": leaf_states,
            "parent_finalization_reserve_bytes": (
                self.plan.parent_finalization_reserve_bytes
            ),
            "queued_retained_reserve_bytes": sum(
                self.plan.leaves[leaf_id].queued_retained_bytes
                for wave in self.plan.waves[1:]
                for leaf_id in wave
            ),
            "fixed_conservative_padding_bytes": max(
                0,
                int(self.plan.estimated_growth_bytes) - calculated_growth,
            ),
            "remaining_peak_growth_bytes": self.plan.estimated_growth_bytes,
            "latest_filesystem_snapshot": None,
            "latest_projected_headroom_bytes": None,
            "full_estimate_count": 1,
            "full_estimate_duration_seconds": self.plan.estimate_duration_seconds,
            "lightweight_check_count": 0,
            "lightweight_check_duration_seconds": 0.0,
            "targeted_reconciliation_count": 0,
            "requests": {},
            "idempotence": {"steps": {}, "transitions": {}, "waves": {}},
        }

    @staticmethod
    def _base_remaining_growth(ledger: Mapping[str, object]) -> int:
        active_ids = set(ledger["active_leaf_ids"])
        leaves = [
            leaf
            for leaf_id, leaf in ledger["leaves"].items()
            if leaf_id in active_ids
        ]
        non_transition = sum(
            sum(
                int(value)
                for name, value in leaf["remaining_by_component"].items()
                if name != "restart_transition_bytes"
            )
            for leaf in leaves
        )
        active_transition = sum(
            sorted(
                (
                    int(leaf["remaining_by_component"].get("restart_transition_bytes", 0))
                    for leaf in leaves
                ),
                reverse=True,
            )[: int(ledger["outer_workers"])]
        )
        return (
            non_transition
            + active_transition
            + int(ledger["queued_retained_reserve_bytes"])
            + int(ledger["parent_finalization_reserve_bytes"])
        )

    @staticmethod
    def _refresh_future_padding(ledger: dict[str, object]) -> None:
        active_wave_index = int(ledger["active_wave_index"])
        finalized_retained_credit = sum(
            int(ledger["queued_retained_by_leaf"].get(leaf_id, 0))
            for wave in ledger["waves"][: active_wave_index + 1]
            for leaf_id in wave
            if ledger["leaves"][leaf_id]["phase"] == "finalized"
        )
        future_waves = ledger["wave_growth_bytes"][active_wave_index + 1 :]
        live_future_peak = (
            max(0, max(int(value) for value in future_waves) - finalized_retained_credit)
            if future_waves
            else 0
        )
        base = StorageAdmissionCoordinator._base_remaining_growth(ledger)
        ledger["fixed_conservative_padding_bytes"] = max(0, live_future_peak - base)

    def _recompute_remaining_peak(self, ledger: dict[str, object]) -> None:
        self._refresh_future_padding(ledger)
        ledger["remaining_peak_growth_bytes"] = self._base_remaining_growth(
            ledger
        ) + int(ledger["fixed_conservative_padding_bytes"])

    def admit_wave(
        self,
        wave_index: int,
        *,
        request_id: str | None = None,
        allow_existing_step_drain: bool = False,
    ) -> DiskBudgetSnapshot:
        """Activate the next immutable wave without changing generations."""
        if not 0 <= wave_index < len(self.plan.waves):
            raise ValueError(f"Invalid storage admission wave index: {wave_index}")
        request_id = request_id or f"wave:{wave_index}"
        with self._lock:
            ledger = copy.deepcopy(self._ledger)
            request = {
                "kind": "wave",
                "wave_index": wave_index,
                "request_id": request_id,
            }
            if not self._is_duplicate_request(
                ledger, request_id=request_id, request=request
            ):
                expected = int(ledger["active_wave_index"])
                if wave_index not in {expected, expected + 1}:
                    raise ValueError(
                        f"Storage wave admission is stale or out of order: {wave_index}"
                    )
                if wave_index == expected + 1:
                    unfinished = [
                        leaf_id
                        for leaf_id in self.plan.waves[expected]
                        if ledger["leaves"][leaf_id]["phase"] != "finalized"
                    ]
                    if unfinished:
                        raise ValueError(
                            "Cannot admit the next storage wave before finalization: "
                            + ", ".join(unfinished)
                        )
                    ledger["active_wave_index"] = wave_index
                    ledger["active_leaf_ids"] = list(self.plan.waves[wave_index])
                    for leaf_id in self.plan.waves[wave_index]:
                        leaf = ledger["leaves"][leaf_id]
                        leaf["remaining_by_component"] = dict(
                            leaf["planned_by_component"]
                        )
                        leaf["phase"] = "prepared"
                    ledger["queued_retained_reserve_bytes"] = sum(
                        int(ledger["queued_retained_by_leaf"][leaf_id])
                        for wave in self.plan.waves[wave_index + 1 :]
                        for leaf_id in wave
                    )
                    self._recompute_remaining_peak(ledger)
                ledger["phase"] = "wave_preflight"
            return self._check_and_commit(
                ledger,
                allow_existing_step_drain=allow_existing_step_drain,
                request_id=request_id,
                request=request,
            )

    def _initialize_ledger(self) -> dict[str, object]:
        existing: dict[str, object] | None = None
        ledger_exists = self.ledger_path.is_file()
        if ledger_exists:
            try:
                loaded = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except (OSError, json.JSONDecodeError):
                existing = None
        can_resume = (
            existing is not None
            and not self.plan.overwrite
            and existing.get("storage_reservation_schema_version")
            == STORAGE_RESERVATION_SCHEMA_VERSION
            and existing.get("plan_identity") == self.plan.identity
            and int(existing.get("filesystem_device", -1))
            == self.plan.filesystem_device
        )
        if can_resume:
            ledger = copy.deepcopy(existing)
            ledger.setdefault("idempotence", {"steps": {}, "transitions": {}, "waves": {}})
            ledger.setdefault(
                "queued_retained_by_leaf",
                {
                    leaf_id: int(leaf.queued_retained_bytes)
                    for leaf_id, leaf in self.plan.leaves.items()
                },
            )
            self._reconcile_finalization_manifests(ledger)
            self._recompute_remaining_peak(ledger)
            ledger["full_estimate_count"] = int(ledger.get("full_estimate_count", 0)) + 1
            ledger["full_estimate_duration_seconds"] = float(
                ledger.get("full_estimate_duration_seconds", 0.0)
            ) + self.plan.estimate_duration_seconds
            ledger["status"] = "reconciled"
            ledger["updated_at"] = _utc_now()
            write_manifest_atomic(self.ledger_path, ledger)
            return ledger
        if not self.plan.overwrite and ledger_exists:
            raise RuntimeError(
                "Storage reservation ledger is invalid or does not match the current "
                f"plan; refusing ambiguous resume: {self.ledger_path}"
            )
        reconciled_partial = not self.plan.overwrite and _has_partial_run_evidence(self.plan)
        if existing is not None:
            _archive_superseded_ledger(self.ledger_path, existing)
        ledger = self._new_ledger()
        if reconciled_partial:
            ledger["targeted_reconciliation_count"] = 1
            ledger["status"] = "reconciled_legacy_partial"
            self._reconcile_finalization_manifests(ledger)
            self._recompute_remaining_peak(ledger)
        write_manifest_atomic(self.ledger_path, ledger)
        return ledger

    def _reconcile_finalization_manifests(self, ledger: dict[str, object]) -> None:
        """Reconcile the only lifecycle state that may commit after the ledger.

        Leaf finalization writes and validates its own durable acceptance
        manifest before notifying this coordinator.  A crash in that narrow
        window is therefore safely recoverable without rescanning producer
        trees or weakening the original plan.
        """
        reconciled = 0
        for leaf_id, leaf_plan in self.plan.leaves.items():
            if leaf_id == "project":
                continue
            leaf_state = ledger["leaves"][leaf_id]
            manifest_path = leaf_plan.setup_dir / "leaf_finalization_manifest.json"
            if not manifest_path.is_file():
                if leaf_state["phase"] == "finalized":
                    raise RuntimeError(
                        "Storage ledger claims a finalized leaf without its "
                        f"authoritative manifest: {manifest_path}"
                    )
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Invalid authoritative leaf finalization manifest: {manifest_path}"
                ) from exc
            if str(payload.get("status") or "").lower() != "success":
                continue
            recorded_project = payload.get("project_dir")
            if recorded_project is None or Path(str(recorded_project)).resolve() != leaf_plan.project_dir:
                raise RuntimeError(
                    f"Leaf finalization project identity changed: {manifest_path}"
                )
            final_index = len(leaf_plan.step_names) - 1
            if leaf_state["phase"] != "finalized":
                leaf_state["last_admitted_step_index"] = final_index
                leaf_state["last_completed_step_index"] = final_index
                leaf_state["remaining_by_component"] = {
                    name: 0 for name in leaf_state["remaining_by_component"]
                }
                leaf_state["phase"] = "finalized"
                reconciled += 1
        if reconciled:
            ledger["targeted_reconciliation_count"] = int(
                ledger.get("targeted_reconciliation_count", 0)
            ) + reconciled

    def _apply_summary(
        self,
        ledger: dict[str, object],
        *,
        leaf_id: str,
        summary: StorageAccountingSummary,
    ) -> None:
        leaf_plan = self.plan.leaves[leaf_id]
        leaf_state = ledger["leaves"][leaf_id]
        admitted_index = int(leaf_state["last_admitted_step_index"])
        if admitted_index < 0:
            raise ValueError(
                f"Cannot account completed step before initial admission for {leaf_id}"
            )
        expected_step = leaf_plan.step_names[admitted_index]
        if summary.completed_step != expected_step:
            raise ValueError(
                "Storage accounting summary is stale or out of order for "
                f"{leaf_id}: expected {expected_step!r}, got {summary.completed_step!r}"
            )
        completed_index = int(leaf_state["last_completed_step_index"])
        if completed_index >= admitted_index:
            raise ValueError(
                f"Storage accounting summary for {leaf_id}/{expected_step} was already applied"
            )

        for component in STEP_MATERIALIZATION_COMPONENTS:
            actual = int(summary.materialized_bytes.get(component, 0))
            observed = int(summary.observed_bytes.get(component, actual))
            high_water = max(
                int(leaf_state["observed_step_high_water_bytes"][component]),
                observed,
            )
            leaf_state["cumulative_materialized_bytes"][component] = int(
                leaf_state["cumulative_materialized_bytes"][component]
            ) + actual
            leaf_state["observed_step_high_water_bytes"][component] = high_water
            # Per-step shares cannot be proven from a net aggregate estimate.
            # Keep the full aggregate obligation until authoritative leaf
            # finalization; summaries can only raise, never release, it.
            leaf_state["remaining_by_component"][component] = max(
                int(leaf_state["remaining_by_component"][component]),
                observed,
            )
        leaf_state["last_completed_step_index"] = admitted_index
        leaf_state["last_accounting_summary"] = summary.as_dict()
        self._recompute_remaining_peak(ledger)

    def _record_filesystem_snapshot(
        self,
        ledger: dict[str, object],
        snapshot: DiskBudgetSnapshot,
        *,
        duration: float,
    ) -> None:
        projected_used = (
            snapshot.used_bytes
            + snapshot.estimated_growth_bytes
            + snapshot.operational_reserve_bytes
        )
        ledger["latest_filesystem_snapshot"] = {
            "path": str(snapshot.filesystem_path),
            "total_bytes": snapshot.total_bytes,
            "used_bytes": snapshot.used_bytes,
            "free_bytes": snapshot.free_bytes,
            "used_fraction": snapshot.used_fraction,
            "projected_used_fraction": snapshot.projected_used_fraction,
            "checked_at": _utc_now(),
        }
        ledger["latest_projected_headroom_bytes"] = snapshot.total_bytes - projected_used
        ledger["lightweight_check_count"] = int(ledger["lightweight_check_count"]) + 1
        ledger["lightweight_check_duration_seconds"] = float(
            ledger["lightweight_check_duration_seconds"]
        ) + duration

    def record_preflight(
        self,
        snapshot: DiskBudgetSnapshot,
        *,
        phase: str = "wave_preflight",
    ) -> None:
        """Commit a full-plan admission snapshot already checked by the caller."""
        with self._lock:
            ledger = copy.deepcopy(self._ledger)
            ledger["phase"] = phase
            ledger["status"] = "admitted"
            ledger["updated_at"] = _utc_now()
            ledger["transition_sequence"] = int(ledger["transition_sequence"]) + 1
            self._record_filesystem_snapshot(ledger, snapshot, duration=0.0)
            write_manifest_atomic(self.ledger_path, ledger)
            self._ledger = ledger

    @staticmethod
    def _idempotence_slot(request: Mapping[str, object]) -> tuple[str, str, str]:
        kind = str(request.get("kind") or "")
        if kind == "step":
            return "steps", str(request.get("leaf_id") or ""), str(
                request.get("step_name") or ""
            )
        if kind == "transition":
            return "transitions", str(request.get("leaf_id") or "parent"), str(
                request.get("phase") or ""
            )
        if kind == "reconcile":
            return "transitions", str(request.get("leaf_id") or ""), str(
                request.get("phase") or "reconcile"
            )
        if kind == "wave":
            return "waves", "coordinator", str(request.get("wave_index"))
        raise ValueError(f"Unknown storage request kind: {kind!r}")

    @staticmethod
    def _is_duplicate_request(
        ledger: dict[str, object],
        *,
        request_id: str,
        request: Mapping[str, object],
    ) -> bool:
        existing = ledger["requests"].get(request_id)
        payload_sha256 = hash_json(dict(request))
        if existing is not None:
            if existing.get("payload_sha256") != payload_sha256:
                raise ValueError(
                    f"Storage request ID was reused with a different payload: {request_id}"
                )
            return True
        category, owner, slot = StorageAdmissionCoordinator._idempotence_slot(request)
        accepted = (
            ledger.get("idempotence", {})
            .get(category, {})
            .get(owner, {})
            .get(slot)
        )
        if accepted is None:
            return False
        if accepted.get("request_id") != request_id or accepted.get(
            "payload_sha256"
        ) != payload_sha256:
            raise ValueError(
                "Storage lifecycle slot was replayed with a different request: "
                f"{category}/{owner}/{slot}"
            )
        return True

    @staticmethod
    def _record_request(
        ledger: dict[str, object],
        *,
        request_id: str,
        request: Mapping[str, object],
        status: str,
    ) -> None:
        requests = ledger["requests"]
        payload_sha256 = hash_json(dict(request))
        category, owner, slot = StorageAdmissionCoordinator._idempotence_slot(request)
        category_state = ledger["idempotence"].setdefault(category, {})
        owner_state = category_state.setdefault(owner, {})
        accepted = owner_state.get(slot)
        if accepted is not None and (
            accepted.get("request_id") != request_id
            or accepted.get("payload_sha256") != payload_sha256
        ):
            raise ValueError(
                "Storage lifecycle slot was committed with a different request: "
                f"{category}/{owner}/{slot}"
            )
        owner_state[slot] = {
            "request_id": request_id,
            "payload_sha256": payload_sha256,
            "status": status,
        }
        requests[request_id] = {
            "payload_sha256": payload_sha256,
            "status": status,
            "sequence": ledger["transition_sequence"],
            "updated_at": ledger["updated_at"],
        }
        while len(requests) > STORAGE_ADMISSION_REQUEST_HISTORY_LIMIT:
            requests.pop(next(iter(requests)))

    def _check_and_commit(
        self,
        ledger: dict[str, object],
        *,
        allow_existing_step_drain: bool,
        request_id: str,
        request: Mapping[str, object],
    ) -> DiskBudgetSnapshot:
        started = time.perf_counter()
        try:
            usage = self._disk_usage(self.plan.root_project_dir) if self._disk_usage else None
            snapshot = check_step_admission(
                self.plan.root_project_dir,
                estimated_growth_bytes=int(ledger["remaining_peak_growth_bytes"]),
                allow_existing_step_drain=allow_existing_step_drain,
                usage=usage,
            )
        except Exception as exc:
            duration = time.perf_counter() - started
            ledger["status"] = "paused_low_disk" if exc.__class__.__name__.startswith("LowDisk") else "failed"
            ledger["last_error"] = str(exc)
            ledger["updated_at"] = _utc_now()
            ledger["transition_sequence"] = int(ledger["transition_sequence"]) + 1
            self._record_request(
                ledger,
                request_id=request_id,
                request=request,
                status=str(ledger["status"]),
            )
            write_manifest_atomic(self.ledger_path, ledger)
            self._ledger = ledger
            raise
        duration = time.perf_counter() - started
        self._record_filesystem_snapshot(ledger, snapshot, duration=duration)
        ledger["status"] = "admitted"
        ledger.pop("last_error", None)
        ledger["updated_at"] = _utc_now()
        ledger["transition_sequence"] = int(ledger["transition_sequence"]) + 1
        self._record_request(
            ledger,
            request_id=request_id,
            request=request,
            status="admitted",
        )
        write_manifest_atomic(self.ledger_path, ledger)
        self._ledger = ledger
        logger.info(
            "Storage admission generation={} phase={} leaf={} step={} used={:.1%} "
            "future_reserve_gib={:.2f} operational_reserve_gib={:.2f} "
            "projected_headroom_gib={:.2f} latency_ms={:.1f}",
            ledger["generation"],
            ledger["phase"],
            request.get("leaf_id", "-"),
            request.get("step_name", "-"),
            snapshot.used_fraction,
            snapshot.estimated_growth_bytes / (1024**3),
            snapshot.operational_reserve_bytes / (1024**3),
            int(ledger["latest_projected_headroom_bytes"]) / (1024**3),
            duration * 1000.0,
        )
        return snapshot

    def admit_step(
        self,
        *,
        leaf_id: str,
        step_name: str,
        summary: StorageAccountingSummary | None = None,
        request_id: str | None = None,
        allow_existing_step_drain: bool = False,
    ) -> DiskBudgetSnapshot:
        request_id = request_id or str(uuid.uuid4())
        with self._lock:
            if leaf_id not in self.plan.leaves:
                raise KeyError(f"Unknown storage-admission leaf: {leaf_id}")
            leaf_plan = self.plan.leaves[leaf_id]
            try:
                step_index = leaf_plan.step_names.index(step_name)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown storage-admission step for {leaf_id}: {step_name}"
                ) from exc
            ledger = copy.deepcopy(self._ledger)
            request = {
                "kind": "step",
                "leaf_id": leaf_id,
                "step_name": step_name,
                "request_id": request_id,
                "summary_sha256": (
                    hash_json(summary.as_dict()) if summary is not None else None
                ),
            }
            duplicate = self._is_duplicate_request(
                ledger, request_id=request_id, request=request
            )
            if not duplicate:
                if leaf_id not in set(ledger["active_leaf_ids"]):
                    raise ValueError(
                        f"Storage admission leaf is not in the active wave: {leaf_id}"
                    )
                leaf_state = ledger["leaves"][leaf_id]
                expected_index = int(leaf_state["last_admitted_step_index"]) + 1
                if step_index != expected_index:
                    raise ValueError(
                        "Storage admission request is stale or out of order for "
                        f"{leaf_id}: expected index {expected_index}, got {step_index}"
                    )
                if expected_index > 0:
                    if summary is None:
                        raise ValueError(
                            "A producer accounting summary is required before admitting "
                            f"{leaf_id}/{step_name}"
                        )
                    try:
                        summary_index = leaf_plan.step_names.index(
                            summary.completed_step
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "Storage accounting summary is stale or out of order for "
                            f"{leaf_id}: unknown step {summary.completed_step!r}"
                        ) from exc
                    if summary_index > int(leaf_state["last_completed_step_index"]):
                        self._apply_summary(ledger, leaf_id=leaf_id, summary=summary)
                elif summary is not None:
                    raise ValueError("Initial step admission cannot include a completed-step summary")
                leaf_state["last_admitted_step_index"] = step_index
                leaf_state["phase"] = "running"
                ledger["phase"] = "running"
            return self._check_and_commit(
                ledger,
                allow_existing_step_drain=allow_existing_step_drain,
                request_id=request_id,
                request=request,
            )

    def transition(
        self,
        *,
        phase: str,
        leaf_id: str | None = None,
        summary: StorageAccountingSummary | None = None,
        release_bytes: int = 0,
        request_id: str | None = None,
        allow_existing_step_drain: bool = True,
    ) -> DiskBudgetSnapshot:
        """Apply an authoritative lifecycle release and admit the next phase."""
        phase = str(phase).strip()
        if not phase:
            raise ValueError("phase is required")
        if release_bytes < 0:
            raise ValueError("release_bytes must be non-negative")
        request_id = request_id or str(uuid.uuid4())
        with self._lock:
            ledger = copy.deepcopy(self._ledger)
            request = {
                "kind": "transition",
                "phase": phase,
                "leaf_id": leaf_id,
                "request_id": request_id,
                "release_bytes": int(release_bytes),
                "summary_sha256": (
                    hash_json(summary.as_dict()) if summary is not None else None
                ),
            }
            duplicate = self._is_duplicate_request(
                ledger, request_id=request_id, request=request
            )
            if not duplicate:
                if summary is not None:
                    if leaf_id is None or leaf_id not in self.plan.leaves:
                        raise ValueError("A valid leaf_id is required with a transition summary")
                    leaf_state = ledger["leaves"][leaf_id]
                    try:
                        summary_index = self.plan.leaves[leaf_id].step_names.index(
                            summary.completed_step
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"Unknown completed step in storage summary: {summary.completed_step}"
                        ) from exc
                    if summary_index > int(leaf_state["last_completed_step_index"]):
                        self._apply_summary(ledger, leaf_id=leaf_id, summary=summary)
                if phase == "leaf_finalized":
                    if leaf_id is None or leaf_id not in self.plan.leaves:
                        raise ValueError("leaf_finalized requires a valid leaf_id")
                    if leaf_id not in set(ledger["active_leaf_ids"]):
                        raise ValueError(
                            f"Cannot finalize inactive storage-admission leaf: {leaf_id}"
                        )
                    leaf_state = ledger["leaves"][leaf_id]
                    final_index = len(self.plan.leaves[leaf_id].step_names) - 1
                    if (
                        int(leaf_state["last_admitted_step_index"]) != final_index
                        or int(leaf_state["last_completed_step_index"]) != final_index
                    ):
                        raise ValueError(
                            f"Cannot release unfinished storage-admission leaf: {leaf_id}"
                        )
                    if leaf_id != "project":
                        finalization = self.plan.leaves[leaf_id].setup_dir / (
                            "leaf_finalization_manifest.json"
                        )
                        try:
                            finalization_payload = json.loads(
                                finalization.read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError) as exc:
                            raise ValueError(
                                "Authoritative leaf finalization manifest is missing or "
                                f"invalid: {finalization}"
                            ) from exc
                        if str(finalization_payload.get("status") or "").lower() != "success":
                            raise ValueError(
                                f"Authoritative leaf finalization is not successful: {finalization}"
                            )
                        recorded_project = finalization_payload.get("project_dir")
                        if recorded_project is None or Path(
                            str(recorded_project)
                        ).resolve() != self.plan.leaves[leaf_id].project_dir:
                            raise ValueError(
                                f"Authoritative leaf finalization project identity changed: {finalization}"
                            )
                    leaf_state["remaining_by_component"] = {
                        key: 0 for key in leaf_state["remaining_by_component"]
                    }
                    leaf_state["phase"] = "finalized"
                    self._recompute_remaining_peak(ledger)
                if release_bytes:
                    remaining_release = int(release_bytes)
                    for field_name in (
                        "parent_finalization_reserve_bytes",
                        "fixed_conservative_padding_bytes",
                    ):
                        available = int(ledger[field_name])
                        applied = min(available, remaining_release)
                        ledger[field_name] = available - applied
                        remaining_release -= applied
                        if remaining_release == 0:
                            break
                    if remaining_release:
                        raise ValueError(
                            "Lifecycle release exceeds coordinator-owned parent and padding reserves"
                        )
                    self._recompute_remaining_peak(ledger)
                ledger["phase"] = phase
            return self._check_and_commit(
                ledger,
                allow_existing_step_drain=allow_existing_step_drain,
                request_id=request_id,
                request=request,
            )


@dataclass(frozen=True)
class StorageAdmissionClient:
    """Spawn-safe client for a parent-owned storage admission coordinator."""

    ipc_dir: Path | None = None
    generation: str | None = None
    leaf_id: str = "project"
    _coordinator: StorageAdmissionCoordinator | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def in_process(
        cls,
        coordinator: StorageAdmissionCoordinator,
        *,
        leaf_id: str,
    ) -> "StorageAdmissionClient":
        return cls(leaf_id=leaf_id, _coordinator=coordinator)

    def for_leaf(self, leaf_id: str) -> "StorageAdmissionClient":
        return StorageAdmissionClient(
            ipc_dir=self.ipc_dir,
            generation=self.generation,
            leaf_id=leaf_id,
            _coordinator=self._coordinator,
        )

    def _request(self, payload: dict[str, object]) -> DiskBudgetSnapshot:
        if self._coordinator is not None:
            return _dispatch_coordinator_request(self._coordinator, payload)
        if self.ipc_dir is None or self.generation is None:
            raise RuntimeError("Storage admission coordinator is unavailable")
        route_id = "_coordinator" if payload.get("kind") == "wave" else self.leaf_id
        leaf_token = hashlib.sha256(route_id.encode("utf-8")).hexdigest()[:16]
        leaf_dir = self.ipc_dir / self.generation / leaf_token
        leaf_dir.mkdir(parents=True, exist_ok=True)
        request_id = str(payload["request_id"])
        request_sha256 = hash_json(payload)
        request_path = leaf_dir / "request.json"
        response_path = leaf_dir / (
            "response."
            + hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
            + ".json"
        )
        envelope = {
            "generation": self.generation,
            "leaf_id": self.leaf_id,
            "route_id": route_id,
            "request_id": request_id,
            "payload_sha256": request_sha256,
            "payload": payload,
        }
        if request_path.is_file():
            try:
                pending = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Storage IPC request is invalid: {request_path}") from exc
            if pending != envelope:
                raise RuntimeError(
                    f"Another storage request is pending for leaf {self.leaf_id}"
                )
        else:
            write_manifest_atomic(request_path, envelope)
        deadline = time.monotonic() + (
            STORAGE_ADMISSION_REQUEST_TIMEOUT_SECONDS
            * STORAGE_ADMISSION_REQUEST_ATTEMPTS
        )
        response: object | None = None
        while time.monotonic() < deadline:
            if response_path.is_file():
                try:
                    candidate = json.loads(response_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.005)
                    continue
                if (
                    candidate.get("generation") == self.generation
                    and candidate.get("request_id") == request_id
                    and candidate.get("payload_sha256") == request_sha256
                ):
                    response = candidate.get("response")
                    try:
                        response_path.unlink()
                    except FileNotFoundError:
                        pass
                    break
            time.sleep(0.005)
        if response is None:
            raise RuntimeError(
                "Storage admission coordinator is unavailable or timed out; "
                "refusing the next boundary"
            )
        if not isinstance(response, dict):
            raise RuntimeError("Storage admission coordinator returned an invalid response")
        if response.get("ok") is not True:
            error_type = str(response.get("error_type") or "RuntimeError")
            message = str(response.get("error") or "Storage admission failed")
            from openamundsen_da.exceptions import LowDiskEmergencyError, LowDiskPauseError

            if error_type == "LowDiskEmergencyError":
                raise LowDiskEmergencyError(message)
            if error_type == "LowDiskPauseError":
                raise LowDiskPauseError(message)
            raise RuntimeError(message)
        snapshot = response.get("snapshot")
        if not isinstance(snapshot, dict):
            raise RuntimeError("Storage admission coordinator omitted its filesystem snapshot")
        return DiskBudgetSnapshot(
            filesystem_path=Path(str(snapshot["filesystem_path"])),
            total_bytes=int(snapshot["total_bytes"]),
            used_bytes=int(snapshot["used_bytes"]),
            free_bytes=int(snapshot["free_bytes"]),
            estimated_growth_bytes=int(snapshot["estimated_growth_bytes"]),
            operational_reserve_bytes=int(snapshot["operational_reserve_bytes"]),
        )

    def admit_step(
        self,
        step_name: str,
        *,
        summary: StorageAccountingSummary | None = None,
        request_id: str | None = None,
        allow_existing_step_drain: bool = False,
    ) -> DiskBudgetSnapshot:
        request_id = request_id or str(uuid.uuid4())
        return self._request(
            {
                "kind": "step",
                "leaf_id": self.leaf_id,
                "step_name": step_name,
                "summary": summary.as_dict() if summary is not None else None,
                "request_id": request_id,
                "allow_existing_step_drain": allow_existing_step_drain,
            }
        )

    def reconcile_finalization(
        self,
        *,
        request_id: str | None = None,
        allow_existing_step_drain: bool = True,
    ) -> DiskBudgetSnapshot:
        request_id = request_id or f"{self.leaf_id}:reconcile_finalization"
        return self._request(
            {
                "kind": "reconcile",
                "leaf_id": self.leaf_id,
                "phase": "project_finalizing",
                "request_id": request_id,
                "allow_existing_step_drain": allow_existing_step_drain,
            }
        )

    def transition(
        self,
        phase: str,
        *,
        summary: StorageAccountingSummary | None = None,
        release_bytes: int = 0,
        request_id: str | None = None,
        allow_existing_step_drain: bool = True,
    ) -> DiskBudgetSnapshot:
        request_id = request_id or str(uuid.uuid4())
        return self._request(
            {
                "kind": "transition",
                "phase": phase,
                "leaf_id": self.leaf_id,
                "summary": summary.as_dict() if summary is not None else None,
                "release_bytes": int(release_bytes),
                "request_id": request_id,
                "allow_existing_step_drain": allow_existing_step_drain,
            }
        )

    def admit_wave(
        self,
        wave_index: int,
        *,
        request_id: str | None = None,
        allow_existing_step_drain: bool = False,
    ) -> DiskBudgetSnapshot:
        request_id = request_id or str(uuid.uuid4())
        return self._request(
            {
                "kind": "wave",
                "wave_index": int(wave_index),
                "request_id": request_id,
                "allow_existing_step_drain": allow_existing_step_drain,
            }
        )


def _snapshot_dict(snapshot: DiskBudgetSnapshot) -> dict[str, object]:
    return {
        "filesystem_path": str(snapshot.filesystem_path),
        "total_bytes": snapshot.total_bytes,
        "used_bytes": snapshot.used_bytes,
        "free_bytes": snapshot.free_bytes,
        "estimated_growth_bytes": snapshot.estimated_growth_bytes,
        "operational_reserve_bytes": snapshot.operational_reserve_bytes,
    }


def _dispatch_coordinator_request(
    coordinator: StorageAdmissionCoordinator,
    payload: Mapping[str, object],
) -> DiskBudgetSnapshot:
    summary_payload = payload.get("summary")
    summary = (
        StorageAccountingSummary.from_dict(summary_payload)
        if isinstance(summary_payload, Mapping)
        else None
    )
    kind = payload.get("kind")
    if kind == "step":
        return coordinator.admit_step(
            leaf_id=str(payload["leaf_id"]),
            step_name=str(payload["step_name"]),
            summary=summary,
            request_id=(str(payload["request_id"]) if payload.get("request_id") else None),
            allow_existing_step_drain=bool(payload.get("allow_existing_step_drain", False)),
        )
    if kind == "transition":
        return coordinator.transition(
            phase=str(payload["phase"]),
            leaf_id=(str(payload["leaf_id"]) if payload.get("leaf_id") else None),
            summary=summary,
            release_bytes=int(payload.get("release_bytes") or 0),
            request_id=(str(payload["request_id"]) if payload.get("request_id") else None),
            allow_existing_step_drain=bool(payload.get("allow_existing_step_drain", True)),
        )
    if kind == "wave":
        return coordinator.admit_wave(
            int(payload["wave_index"]),
            request_id=(str(payload["request_id"]) if payload.get("request_id") else None),
            allow_existing_step_drain=bool(
                payload.get("allow_existing_step_drain", False)
            ),
        )
    if kind == "reconcile":
        return coordinator.reconcile_leaf_finalization(
            leaf_id=str(payload["leaf_id"]),
            request_id=(str(payload["request_id"]) if payload.get("request_id") else None),
            allow_existing_step_drain=bool(
                payload.get("allow_existing_step_drain", True)
            ),
        )
    raise ValueError(f"Unknown storage admission request kind: {kind!r}")


class StorageAdmissionServer:
    """Parent thread serializing durable per-leaf request manifests."""

    def __init__(self, coordinator: StorageAdmissionCoordinator) -> None:
        self.coordinator = coordinator
        self._ipc_dir = coordinator.ledger_path.parent / "ipc"
        self._generation_dir = self._ipc_dir / coordinator.generation
        self._generation_dir.mkdir(parents=True, exist_ok=True)
        self._request_paths = {
            leaf_id: self._generation_dir
            / hashlib.sha256(leaf_id.encode("utf-8")).hexdigest()[:16]
            / "request.json"
            for leaf_id in coordinator.plan.leaves
        }
        self._control_request_path = self._generation_dir / (
            hashlib.sha256(b"_coordinator").hexdigest()[:16]
        ) / "request.json"
        self._control_request_path.parent.mkdir(parents=True, exist_ok=True)
        for path in self._request_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._serve,
            name="storage-admission-coordinator",
            daemon=True,
        )
        self._thread.start()

    def client(self, *, leaf_id: str) -> StorageAdmissionClient:
        if leaf_id not in self._request_paths:
            raise KeyError(f"Unknown storage-admission leaf: {leaf_id}")
        return StorageAdmissionClient(
            ipc_dir=self._ipc_dir,
            generation=self.coordinator.generation,
            leaf_id=leaf_id,
        )

    def _serve(self) -> None:
        while not self._stop.is_set():
            handled = False
            active_paths = [
                (leaf_id, self._request_paths[leaf_id])
                for leaf_id in self.coordinator.active_leaf_ids()
            ]
            for leaf_id, request_path in [
                (None, self._control_request_path),
                *active_paths,
            ]:
                if not request_path.is_file():
                    continue
                handled = True
                envelope: dict[str, object] = {}
                try:
                    envelope = json.loads(request_path.read_text(encoding="utf-8"))
                    payload = envelope.get("payload")
                    if (
                        not isinstance(payload, dict)
                        or envelope.get("generation") != self.coordinator.generation
                        or (
                            leaf_id is not None
                            and envelope.get("leaf_id") != leaf_id
                        )
                        or envelope.get("route_id")
                        != (leaf_id if leaf_id is not None else "_coordinator")
                        or envelope.get("payload_sha256") != hash_json(payload)
                    ):
                        raise ValueError(f"Invalid storage IPC request: {request_path}")
                    snapshot = _dispatch_coordinator_request(self.coordinator, payload)
                except Exception as exc:  # noqa: BLE001 - serialized to worker
                    response = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                else:
                    response = {"ok": True, "snapshot": _snapshot_dict(snapshot)}
                request_id = str(envelope.get("request_id") or "invalid")
                response_path = request_path.parent / (
                    "response."
                    + hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
                    + ".json"
                )
                try:
                    request_path.unlink()
                except FileNotFoundError:
                    pass
                write_manifest_atomic(
                    response_path,
                    {
                        "generation": envelope.get("generation"),
                        "request_id": request_id,
                        "payload_sha256": envelope.get("payload_sha256"),
                        "response": response,
                    },
                )
            if not handled:
                self._stop.wait(0.005)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def __enter__(self) -> "StorageAdmissionServer":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "STEP_MATERIALIZATION_COMPONENTS",
    "STORAGE_RESERVATION_SCHEMA_VERSION",
    "StorageAccountingSummary",
    "StorageAdmissionClient",
    "StorageAdmissionCoordinator",
    "StorageAdmissionServer",
    "StorageLeafPlan",
    "StoragePlan",
    "accounting_summary_from_inventory",
    "admit_storage_transition",
    "build_storage_plan",
    "storage_reservation_path",
]
