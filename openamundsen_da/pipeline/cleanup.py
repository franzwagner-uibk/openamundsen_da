"""Preview-first, ledger-backed cleanup of package-owned artifacts."""

from __future__ import annotations

import json
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
from openamundsen_da.util.da_output import output_retention_mode, validate_project_da_output_grids
from openamundsen_da.util.map_support import validate_map_support
from openamundsen_da.util.point_output import validate_project_ensemble_points
from openamundsen_da.util.forcing_output import validate_project_ensemble_forcing
from openamundsen_da.util.retention import (
    apply_retention_batch,
    planned_retention_paths,
    reconcile_retention_ledger,
)
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.restart_state import validate_restart_state


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


def _validated_successor_states(project_dir: Path, successor_step: Path) -> tuple[Path, ...]:
    """Require a readable checkpoint for open loop and every successor member."""
    project_cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    try:
        ensemble_size = int(project_cfg[DA_BLOCK]["prior_forcing"]["ensemble_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Cannot validate successor checkpoints without ensemble_size") from exc
    prior_dir = successor_step / "ensembles" / "prior"
    roots = sorted(path for path in prior_dir.iterdir() if path.is_dir()) if prior_dir.is_dir() else []
    expected_names = ["open_loop", *(f"member_{index:03d}" for index in range(1, ensemble_size + 1))]
    root_by_name = {
        path.name: path
        for path in roots
        if path.name == "open_loop" or path.name.startswith("member_")
    }
    actual_names = sorted(root_by_name)
    if actual_names != sorted(expected_names):
        raise RuntimeError(
            "Successor checkpoint membership differs from configured ensemble: "
            f"{actual_names} != {sorted(expected_names)} in {successor_step}"
        )
    member_roots = [root_by_name[name] for name in expected_names]
    pattern = state_patterns_from_setup(project_dir)[0]
    output_name = STATE_DEFAULT_NAME if any(char in pattern for char in "*?[]") else pattern
    states: list[Path] = []
    for root in member_roots:
        state = root / "results" / output_name
        validate_restart_state(state)
        states.append(state.resolve())
    return tuple(states)


def _member_run_manifests(project_dir: Path, candidates: Sequence[Path]) -> tuple[Path, ...]:
    """Resolve immutable package producer manifests for owned member artifacts."""
    manifests: set[Path] = set()
    for candidate in candidates:
        found_member_root = False
        for parent in candidate.resolve().parents:
            if parent == project_dir:
                break
            if parent.name == "open_loop" or parent.name.startswith("member_"):
                found_member_root = True
                manifest = parent / "results" / "member_run.json"
                if not manifest.is_file() or manifest.is_symlink():
                    raise RuntimeError(f"Producer member manifest is missing or invalid: {manifest}")
                try:
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Producer member manifest is unreadable: {manifest}") from exc
                if not isinstance(payload, dict):
                    raise RuntimeError(f"Producer member manifest root is invalid: {manifest}")
                if str(payload.get("status", "")).strip().lower() != "success":
                    raise RuntimeError(f"Producer member manifest is not successful: {manifest}")
                if str(payload.get("member", "")).strip() != parent.name:
                    raise RuntimeError(
                        "Producer member manifest identity differs from its member directory: "
                        f"{payload.get('member')!r} != {parent.name!r} in {manifest}"
                    )
                manifests.add(manifest.resolve())
                break
        if not found_member_root:
            raise RuntimeError(f"Cannot resolve member producer for cleanup candidate: {candidate}")
    return tuple(sorted(manifests))


def _forcing_plot_producer_manifests(
    project_dir: Path,
    candidates: Sequence[Path],
) -> tuple[Path, ...]:
    """Return successful member manifests for every plotted forcing step."""
    manifests: set[Path] = set()
    for candidate in candidates:
        try:
            step_dir = candidate.resolve().parents[2]
            step_dir.relative_to(project_dir / "steps")
        except (IndexError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot resolve forcing-plot producer step: {candidate}"
            ) from exc
        roots = sorted(
            path
            for path in (step_dir / "ensembles" / "prior").iterdir()
            if path.is_dir()
            and (path.name == "open_loop" or path.name.startswith("member_"))
        ) if (step_dir / "ensembles" / "prior").is_dir() else []
        if not roots:
            raise RuntimeError(f"Forcing-plot producer members are missing in {step_dir}")
        for root in roots:
            manifests.update(_member_run_manifests(project_dir, (root / "meteo",)))
    return tuple(sorted(manifests))


def clean_predecessor_checkpoint(
    project_dir: Path,
    step_dir: Path,
    *,
    successor_step: Path | None = None,
    apply: bool,
) -> tuple[Path, ...]:
    """Remove a predecessor checkpoint after its successor has validated."""
    project_dir = Path(project_dir).resolve()
    if output_retention_mode(project_dir) != "compact":
        return ()
    candidates = _restart_checkpoint_candidates(project_dir, step_dir)
    if apply and candidates:
        if successor_step is None:
            raise RuntimeError("Successor step is required before deleting a predecessor checkpoint")
        successor_states = _validated_successor_states(project_dir, Path(successor_step).resolve())
        apply_retention_batch(
            project_dir,
            artifact_class=f"restart_checkpoint:{Path(step_dir).name}",
            paths=candidates,
            final_consumer="validated successor member checkpoints",
            regeneration_recipe="rerun the predecessor step from the prior retained checkpoint",
            retained_consumers=successor_states,
            producer_manifests=_member_run_manifests(project_dir, candidates),
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
    derived_plots = any(project_dir.glob("steps/step_*/plots/forcing/*.png"))
    report = project_dir / "results" / "reports" / "project_report.pdf"
    if derived_plots and not report.is_file():
        # The accepted report is the durable rendered consumer for these
        # diagnostics.  Keep their raw forcing regeneration source until the
        # render succeeds and the derived plot batch can be finalized first.
        return []
    validate_project_ensemble_forcing(project_dir, output_nc=retained)
    return candidates


def _derived_forcing_plot_candidates(project_dir: Path) -> list[Path]:
    """Return step forcing PNGs after their compact source and report exist."""
    planned = planned_retention_paths(project_dir, artifact_class="derived_forcing_plot")
    if planned:
        return list(planned)
    retained = project_dir / "results" / "forcing" / "ensemble_forcing.nc"
    report = project_dir / "results" / "reports" / "project_report.pdf"
    if not retained.is_file() or not report.is_file():
        return []
    candidates = sorted(
        path.resolve()
        for path in project_dir.glob("steps/step_*/plots/forcing/*.png")
        if path.is_file() and not path.is_symlink()
    )
    if not candidates:
        return []
    # Validate against the still-present raw CSVs before the forcing batch is
    # applied.  On an interrupted retry, the planned ledger batch binds both
    # retained consumers byte-for-byte and is revalidated before every unlink.
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
    patterns = (
        "steps/step_*/ensembles/*/*/results/output_grids*.nc",
        "steps/step_*/ensembles/*/*/results/**/*.tif",
        "steps/step_*/ensembles/*/*/results/**/*.tiff",
    )
    candidates = sorted(
        {
            path.resolve()
            for pattern in patterns
            for path in project_dir.glob(pattern)
            if path.is_file() and not path.is_symlink()
        }
    )
    if not candidates:
        return []
    validate_project_da_output_grids(project_dir, output_nc=compact)
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
        from openamundsen_da.methods.viz.maps.panel_renderers import project_da_map_support_fields

        built_support = project_da_map_support_fields(project_dir)
        if built_support is None:
            return []
        _support_dates, source_fields, roi_mask = built_support
        validate_map_support(
            project_dir,
            dates=[event.date for event in events],
            fields=required_fields,
            roi_mask=roi_mask,
            source_fields=source_fields,
        )
    return candidates


def _cleanup_classes(project_dir: Path) -> dict[str, list[Path]]:
    if output_retention_mode(project_dir) != "compact":
        return {}
    return {
        "restart_state": _single_domain_cleanup_candidates(
            project_dir,
            state_patterns_from_setup(project_dir),
        ),
        "member_point_csv": _compact_point_candidates(project_dir),
        # Plan and remove derived plots while their raw forcing sources are
        # still available for exact compact-store validation.  This ordering
        # also makes an interruption before forcing deletion safely resumable.
        "derived_forcing_plot": _derived_forcing_plot_candidates(project_dir),
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
    forcing_plot_failed = False
    for artifact_class, class_paths in classes.items():
        if not class_paths:
            continue
        if artifact_class == "member_forcing_csv" and forcing_plot_failed:
            failures.extend(
                CleanupFailure(
                    path=path,
                    error=(
                        "forcing cleanup withheld because derived forcing-plot "
                        "cleanup did not complete"
                    ),
                )
                for path in class_paths
                if path.exists()
            )
            continue
        try:
            if artifact_class == "member_point_csv":
                consumers = (project_dir / "results" / "points" / "ensemble_points.nc",)
            elif artifact_class == "member_forcing_csv":
                consumers = (project_dir / "results" / "forcing" / "ensemble_forcing.nc",)
            elif artifact_class == "derived_forcing_plot":
                consumers = (
                    project_dir / "results" / "forcing" / "ensemble_forcing.nc",
                    project_dir / "results" / "reports" / "project_report.pdf",
                )
            elif artifact_class == "member_grid":
                consumers = [project_dir / "results" / "grids" / "da_output_grids.nc"]
                support = project_dir / "results" / "grids" / "da_map_support.nc"
                if support.is_file():
                    consumers.append(support)
                consumers = tuple(consumers)
            else:
                consumers = tuple(
                    path
                    for path in (
                        project_dir / "results" / "grids" / "da_output_grids.nc",
                        project_dir / "results" / "points" / "ensemble_points.nc",
                        project_dir / "results" / "forcing" / "ensemble_forcing.nc",
                    )
                    if path.is_file() and not path.is_symlink()
                )
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
                            "rerender project forcing plots from retained "
                            "results/forcing/ensemble_forcing.nc; rerun the project "
                            "to recreate step-local diagnostic PNGs"
                            if artifact_class == "derived_forcing_plot"
                            else (
                                "read retained DA grid and map-support NetCDF outputs"
                                if artifact_class == "member_grid"
                                else "rerun propagation from immutable inputs"
                            )
                        )
                    )
                ),
                retained_consumers=consumers,
                producer_manifests=(
                    _forcing_plot_producer_manifests(project_dir, class_paths)
                    if artifact_class == "derived_forcing_plot"
                    else _member_run_manifests(project_dir, class_paths)
                ),
            )
            deleted.extend(path for path in class_paths if not path.exists())
            freed += sum(sizes[path] for path in class_paths if not path.exists())
        except Exception as exc:  # noqa: BLE001
            if artifact_class == "derived_forcing_plot":
                forcing_plot_failed = True
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
