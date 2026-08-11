"""Preview-first, ledger-backed cleanup of package-owned artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from openamundsen_da.core.constants import (
    DA_BLOCK,
    RESTART_BLOCK,
    RESTART_STATE_PATTERN,
    STATE_DEFAULT_NAME,
)
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.results import CleanupFailure, CleanupResult, WorkflowStatus
from openamundsen_da.util.da_output import output_retention_mode
from openamundsen_da.util.map_support import validate_map_support
from openamundsen_da.util.point_output import validate_project_ensemble_points
from openamundsen_da.util.forcing_output import validate_project_ensemble_forcing
from openamundsen_da.util.retention import (
    apply_retention_batch,
    planned_retention_paths,
    reconcile_retention_ledger,
)
from openamundsen_da.util.da_events import load_assimilation_events


def _read_restart_config(project_dir: Path) -> dict:
    """Return restart config dict from project YAML (best-effort)."""
    try:
        project_yaml = find_project_yaml(project_dir)
        cfg = _read_yaml_file(project_yaml) or {}
        da_cfg = cfg.get(DA_BLOCK) or {}
        return da_cfg.get(RESTART_BLOCK) or {}
    except Exception:
        return {}


def state_patterns_from_setup(project_dir: Path) -> list[str]:
    """Return configured and default state filename patterns."""
    restart_cfg = _read_restart_config(project_dir)
    patt = restart_cfg.get(RESTART_STATE_PATTERN) or STATE_DEFAULT_NAME
    patterns = [str(patt), STATE_DEFAULT_NAME]
    seen = set()
    unique = []
    for p in patterns:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _single_domain_cleanup_candidates(project_dir: Path, patterns: Sequence[str]) -> list[Path]:
    """Return owned restart artifacts below top-level project steps only."""
    project_dir = Path(project_dir).resolve()
    steps_dir = project_dir / "steps"
    if not steps_dir.is_dir():
        return []
    candidates: dict[str, Path] = {}
    for pattern in patterns:
        for path in steps_dir.glob(f"step_*/ensembles/*/*/results/{pattern}"):
            if path.is_file() and not path.is_symlink():
                resolved = path.resolve()
                resolved.relative_to(project_dir)
                candidates[resolved.as_posix()] = resolved
    state_candidates = set(candidates.values())
    pointer_patterns = (
        "step_*/ensembles/*/*/state_pointer.json",
        "step_*/ensembles/*/*/results/state_pointer.json",
    )
    for pointer in (path for pattern in pointer_patterns for path in steps_dir.glob(pattern)):
        if not pointer.is_file() or pointer.is_symlink():
            continue
        try:
            import json

            data = json.loads(pointer.read_text(encoding="utf-8"))
            target_raw = data.get("path") if isinstance(data, dict) else None
            target = (pointer.parent / str(target_raw)).resolve() if target_raw else None
        except Exception:
            target = None
        if target is None or not target.is_file() or target in state_candidates:
            resolved = pointer.resolve()
            resolved.relative_to(project_dir)
            candidates[resolved.as_posix()] = resolved
    return [candidates[key] for key in sorted(candidates)]


def _restart_checkpoint_candidates(project_dir: Path, step_dir: Path) -> list[Path]:
    """Return one predecessor checkpoint and every pointer targeting it."""
    project_dir = Path(project_dir).resolve()
    step_dir = Path(step_dir).resolve()
    step_dir.relative_to(project_dir / "steps")
    states: set[Path] = set()
    for pattern in state_patterns_from_setup(project_dir):
        states.update(
            path.resolve()
            for path in step_dir.glob(f"ensembles/*/*/results/{pattern}")
            if path.is_file() and not path.is_symlink()
        )
    candidates = set(states)
    for pointer in project_dir.glob("steps/step_*/ensembles/*/*/**/state_pointer.json"):
        if not pointer.is_file() or pointer.is_symlink():
            continue
        try:
            import json

            data = json.loads(pointer.read_text(encoding="utf-8"))
            raw = data.get("path") if isinstance(data, dict) else None
            target = (pointer.parent / str(raw)).resolve() if raw else None
        except Exception:
            target = None
        if target in states:
            candidates.add(pointer.resolve())
    return sorted(candidates)


def clean_predecessor_checkpoint(project_dir: Path, step_dir: Path, *, apply: bool) -> tuple[Path, ...]:
    """Remove a predecessor checkpoint after its successor has validated."""
    project_dir = Path(project_dir).resolve()
    if output_retention_mode(project_dir) != "compact":
        return ()
    candidates = _restart_checkpoint_candidates(project_dir, step_dir)
    if apply and candidates:
        apply_retention_batch(
            project_dir,
            artifact_class=f"restart_checkpoint:{Path(step_dir).name}",
            paths=candidates,
            final_consumer="validated successor member checkpoints",
            regeneration_recipe="rerun the predecessor step from the prior retained checkpoint",
        )
    return tuple(candidates)


def _compact_point_candidates(project_dir: Path) -> list[Path]:
    """Return point CSVs only when their lossless compact store exists."""
    planned = planned_retention_paths(project_dir, artifact_class="member_point_csv")
    if planned:
        return list(planned)
    retained = project_dir / "results" / "points" / "ensemble_points.nc"
    if not retained.is_file():
        return []
    candidates = sorted(
        path.resolve()
        for path in project_dir.glob("steps/step_*/ensembles/*/*/results/point_*.csv")
        if path.is_file() and not path.is_symlink()
    )
    if not candidates:
        return []
    validate_project_ensemble_points(project_dir, output_nc=retained)
    return candidates


def _compact_forcing_candidates(project_dir: Path) -> list[Path]:
    """Return member forcing CSVs only when their lossless compact store exists."""
    planned = planned_retention_paths(project_dir, artifact_class="member_forcing_csv")
    if planned:
        return list(planned)
    retained = project_dir / "results" / "forcing" / "ensemble_forcing.nc"
    if not retained.is_file():
        return []
    candidates = sorted(
        path.resolve()
        for path in project_dir.glob("steps/step_*/ensembles/*/*/meteo/*.csv")
        if path.name != "stations.csv" and path.is_file() and not path.is_symlink()
    )
    if not candidates:
        return []
    validate_project_ensemble_forcing(project_dir, output_nc=retained)
    return candidates


def _compact_grid_candidates(project_dir: Path) -> list[Path]:
    """Return raw member grids only after final metrics and map support exist."""
    planned = planned_retention_paths(project_dir, artifact_class="member_grid")
    if planned:
        return list(planned)
    compact = project_dir / "results" / "grids" / "da_output_grids.nc"
    if not compact.is_file():
        return []
    events = load_assimilation_events(project_dir)
    fraction_events = [
        event
        for event in events
        if event.variable in {"scf", "wet_snow", "wet_snow_line"}
    ]
    fraction_variables = {event.variable for event in fraction_events}
    support = project_dir / "results" / "grids" / "da_map_support.nc"
    if fraction_variables:
        if not support.is_file():
            return []
        required_fields: set[str] = set()
        if "scf" in fraction_variables:
            required_fields.update(
                {
                    "scf_open_loop_binary",
                    "scf_prior_probability",
                    "scf_posterior_probability",
                }
            )
        if fraction_variables & {"wet_snow", "wet_snow_line"}:
            required_fields.update(
                {
                    "wet_snow_open_loop",
                    "wet_snow_prior_probability",
                    "wet_snow_posterior_probability",
                }
            )
        validate_map_support(
            project_dir,
            dates=[event.date for event in events],
            fields=required_fields,
        )
    patterns = (
        "steps/step_*/ensembles/*/*/results/output_grids*.nc",
        "steps/step_*/ensembles/*/*/results/**/*.tif",
        "steps/step_*/ensembles/*/*/results/**/*.tiff",
    )
    return sorted(
        {
            path.resolve()
            for pattern in patterns
            for path in project_dir.glob(pattern)
            if path.is_file() and not path.is_symlink()
        }
    )


def _cleanup_classes(project_dir: Path) -> dict[str, list[Path]]:
    if output_retention_mode(project_dir) != "compact":
        return {}
    return {
        "restart_state": _single_domain_cleanup_candidates(
            project_dir,
            state_patterns_from_setup(project_dir),
        ),
        "member_point_csv": _compact_point_candidates(project_dir),
        "member_forcing_csv": _compact_forcing_candidates(project_dir),
        "member_grid": _compact_grid_candidates(project_dir),
    }


def clean_project_artifacts(project_dir: Path, *, apply: bool) -> CleanupResult:
    """Preview or delete safe single-domain restart artifacts."""
    project_dir = Path(project_dir).resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    reconcile_retention_ledger(project_dir)
    classes = _cleanup_classes(project_dir)
    candidates = sorted({path for paths in classes.values() for path in paths})
    sizes: dict[Path, int] = {}
    for path in candidates:
        try:
            sizes[path] = path.stat().st_size
        except OSError:
            sizes[path] = 0
    if not apply:
        return CleanupResult(
            project_dir=project_dir,
            status=WorkflowStatus.PREVIEW,
            applied=False,
            eligible_paths=tuple(candidates),
            deleted_paths=(),
            failures=(),
            eligible_bytes=sum(sizes.values()),
            freed_bytes=0,
        )

    deleted: list[Path] = []
    failures: list[CleanupFailure] = []
    freed = 0
    for artifact_class, class_paths in classes.items():
        if not class_paths:
            continue
        try:
            apply_retention_batch(
                project_dir,
                artifact_class=artifact_class,
                paths=class_paths,
                final_consumer="validated compact output and configured render",
                regeneration_recipe=(
                    "read results/points/ensemble_points.nc"
                    if artifact_class == "member_point_csv"
                    else (
                        "read results/forcing/ensemble_forcing.nc"
                        if artifact_class == "member_forcing_csv"
                        else (
                            "read retained DA grid and map-support NetCDF outputs"
                            if artifact_class == "member_grid"
                            else "rerun propagation from immutable inputs"
                        )
                    )
                ),
            )
            deleted.extend(path for path in class_paths if not path.exists())
            freed += sum(sizes[path] for path in class_paths if not path.exists())
        except Exception as exc:  # noqa: BLE001
            failures.extend(CleanupFailure(path=path, error=str(exc)) for path in class_paths if path.exists())
    return CleanupResult(
        project_dir=project_dir,
        status=WorkflowStatus.APPLIED,
        applied=True,
        eligible_paths=tuple(candidates),
        deleted_paths=tuple(deleted),
        failures=tuple(failures),
        eligible_bytes=sum(sizes.values()),
        freed_bytes=freed,
    )
