"""Preview-first, ledger-backed cleanup of package-owned artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

from openamundsen_da.core.constants import (
    DA_BLOCK,
    RESTART_BLOCK,
    RESTART_STATE_PATTERN,
    STATE_DEFAULT_NAME,
    STATE_POINTER_JSON,
)
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    default_results_dir,
    find_project_yaml,
    list_member_dirs,
    list_steps_sorted,
    member_run_manifest_path,
    open_loop_dir,
    state_pointer_path,
)
from openamundsen_da.results import CleanupFailure, CleanupResult, WorkflowStatus
from openamundsen_da.pipeline.rendering import (
    render_completion_manifest_path,
    validate_render_completion,
)
from openamundsen_da.util.da_output import output_retention_mode, validate_project_da_output_grids
from openamundsen_da.util.map_support import validate_map_support
from openamundsen_da.util.point_output import validate_project_ensemble_points
from openamundsen_da.util.forcing_output import validate_project_ensemble_forcing
from openamundsen_da.util.retention import (
    active_retention_generation,
    apply_retention_batch,
    apply_runtime_tree_cleanup,
    complete_retention_generation,
    planned_retention_generation_dependencies,
    planned_retention_paths,
    reconcile_retention_ledger,
    start_retention_generation,
    start_runtime_tree_cleanup,
)
from openamundsen_da.util.runtime_generation import (
    RUNTIME_LAYOUT,
    load_runtime_generation,
    record_runtime_consumer_validation,
    record_runtime_rolling_removal,
    runtime_accounted_totals,
    runtime_consumer_validation_evidence,
    runtime_generation_root,
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
    member_roots: list[Path] = []
    for ensemble in ("prior", "posterior"):
        member_roots.extend(list_member_dirs(step_dir / "ensembles", ensemble))
    prior_open_loop = open_loop_dir(step_dir)
    if prior_open_loop.is_dir():
        member_roots.append(prior_open_loop)
    for pattern in state_patterns_from_setup(project_dir):
        states.update(
            path.resolve()
            for member in member_roots
            for path in default_results_dir(member).glob(pattern)
            if path.is_file() and not path.is_symlink()
        )
    candidates = set(states)
    project_steps = sorted(
        path
        for path in (project_dir / "steps").glob("step_*")
        if path.is_dir() and not path.is_symlink()
    )
    pointers = [
        state_pointer_path(member)
        for step in project_steps
        for ensemble in ("prior", "posterior")
        for member in list_member_dirs(step / "ensembles", ensemble)
    ]
    pointers.extend(
        state_pointer_path(open_loop_dir(step))
        for step in project_steps
        if open_loop_dir(step).is_dir()
    )
    for pointer in pointers:
        if not pointer.is_file() or pointer.is_symlink():
            continue
        target = _resolve_checkpoint_pointer_target(project_dir, pointer)
        if target in states:
            candidates.add(pointer.resolve())
    return sorted(candidates)


def _resolve_checkpoint_pointer_target(project_dir: Path, pointer: Path) -> Path:
    """Resolve one checkpoint pointer to a contained, existing state file."""
    project_dir = Path(project_dir).resolve()
    pointer = Path(pointer).resolve()
    runtime_root = runtime_generation_root(project_dir)

    def contained(path: Path) -> bool:
        roots = [project_dir / "steps"]
        if runtime_root is not None:
            roots.append(runtime_root)
        return any(
            path == root or root in path.parents
            for root in roots
        )

    if not contained(pointer):
        raise RuntimeError(f"Checkpoint pointer escapes project runtime: {pointer}")
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Checkpoint pointer is unreadable: {pointer}") from exc
    raw = payload.get("path") if isinstance(payload, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(f"Checkpoint pointer has no valid path: {pointer}")

    raw_target = Path(raw)
    targets: list[Path] = []
    if raw_target.is_absolute():
        parts = raw_target.parts
        runtime_indices = [
            index for index, part in enumerate(parts) if part == ".openamundsen-da"
        ]
        if runtime_indices:
            targets.append(project_dir.joinpath(*parts[runtime_indices[-1] :]))
        step_indices = [index for index, part in enumerate(parts) if part == "steps"]
        if step_indices:
            targets.append(project_dir.joinpath(*parts[step_indices[-1] :]))
        targets.append(raw_target)
    else:
        targets.append(pointer.parent / raw_target)

    for candidate in targets:
        target = candidate.resolve()
        if not contained(target):
            continue
        if target.is_file() and not target.is_symlink():
            return target
    raise RuntimeError(
        f"Checkpoint pointer target is missing or outside project steps: {pointer} -> {raw}"
    )


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
        state = default_results_dir(root) / output_name
        validate_restart_state(state)
        states.append(state.resolve())
    return tuple(states)


def _member_run_manifests(project_dir: Path, candidates: Sequence[Path]) -> tuple[Path, ...]:
    """Resolve immutable package producer manifests for owned member artifacts."""
    project_dir = Path(project_dir).resolve()
    manifests: set[Path] = set()
    for candidate in candidates:
        producer_artifact = Path(candidate).resolve()
        if producer_artifact.name == STATE_POINTER_JSON:
            producer_artifact = _resolve_checkpoint_pointer_target(
                project_dir,
                producer_artifact,
            )
        found_member_root = False
        for parent in producer_artifact.parents:
            if parent == project_dir:
                break
            if parent.name == "open_loop" or parent.name.startswith("member_"):
                found_member_root = True
                manifest = member_run_manifest_path(parent)
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
            raise RuntimeError(
                f"Cannot resolve member producer for cleanup candidate: {candidate}"
            )
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
        runtime = load_runtime_generation(project_dir)
        if runtime is not None and runtime.get("layout") == RUNTIME_LAYOUT:
            runtime_root = runtime_generation_root(project_dir)
            if runtime_root is None:
                raise RuntimeError("Compact runtime generation root is unavailable")
            producer_manifests = [
                member_run_manifest_path(member)
                for member in [
                    open_loop_dir(step_dir),
                    *list_member_dirs(step_dir / "ensembles", "prior"),
                ]
            ]
            for manifest in producer_manifests:
                if not manifest.is_file() or manifest.is_symlink():
                    raise RuntimeError(
                        f"Predecessor checkpoint producer is missing: {manifest}"
                    )
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                if payload.get("status") != "success":
                    raise RuntimeError(
                        f"Predecessor checkpoint producer is not successful: {manifest}"
                    )
            sizes: dict[Path, int] = {}
            for candidate in candidates:
                resolved = candidate.resolve()
                try:
                    resolved.relative_to(runtime_root.resolve())
                except ValueError as exc:
                    raise RuntimeError(
                        f"Runtime checkpoint cleanup path escapes generation: {candidate}"
                    ) from exc
                sizes[resolved] = resolved.stat().st_size
            for candidate in sizes:
                candidate.unlink()
            for successor in successor_states:
                validate_restart_state(successor)
            record_runtime_rolling_removal(
                project_dir,
                path_sizes=sizes,
            )
            return tuple(candidates)
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
    try:
        validate_render_completion(project_dir)
    except (FileNotFoundError, ValueError):
        render_complete = False
    else:
        render_complete = True
    if derived_plots and not render_complete:
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
    try:
        validate_render_completion(project_dir)
    except (FileNotFoundError, ValueError):
        return []
    if not retained.is_file():
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


def _retained_consumers_for_class(
    project_dir: Path,
    *,
    artifact_class: str,
) -> tuple[Path, ...]:
    """Resolve one cleanup class's stable retained-consumer contract."""
    if artifact_class == "member_point_csv":
        consumers = [project_dir / "results" / "points" / "ensemble_points.nc"]
    elif artifact_class == "member_forcing_csv":
        consumers = [project_dir / "results" / "forcing" / "ensemble_forcing.nc"]
    elif artifact_class == "derived_forcing_plot":
        consumers = [
            project_dir / "results" / "forcing" / "ensemble_forcing.nc",
            render_completion_manifest_path(project_dir),
        ]
    elif artifact_class == "member_grid":
        consumers = [project_dir / "results" / "grids" / "da_output_grids.nc"]
        support = project_dir / "results" / "grids" / "da_map_support.nc"
        if support.is_file():
            consumers.append(support)
    else:
        consumers = [
            path
            for path in (
                project_dir / "results" / "grids" / "da_output_grids.nc",
                project_dir / "results" / "points" / "ensemble_points.nc",
                project_dir / "results" / "forcing" / "ensemble_forcing.nc",
            )
            if path.is_file() and not path.is_symlink()
        ]
    return tuple(consumers)


def runtime_retained_consumers(project_dir: str | Path) -> tuple[Path, ...]:
    """Return the small retained authority set for tree-level cleanup."""
    project_dir = Path(project_dir).resolve()
    consumers = [
        project_dir / "results" / "grids" / "da_output_grids.nc",
        project_dir / "results" / "points" / "ensemble_points.nc",
        project_dir / "results" / "forcing" / "ensemble_forcing.nc",
        validate_render_completion(project_dir),
    ]
    map_support = project_dir / "results" / "grids" / "da_map_support.nc"
    if map_support.is_file():
        consumers.append(map_support)
    missing = [path for path in consumers if not path.is_file() or path.is_symlink()]
    if missing:
        raise RuntimeError(f"Required retained compact consumer is missing: {missing[0]}")
    return tuple(dict.fromkeys(path.resolve() for path in consumers))


def record_runtime_cleanup_authority(project_dir: str | Path) -> Path | None:
    """Record already-validated compact outputs after successful rendering."""
    project_dir = Path(project_dir).resolve()
    return record_runtime_consumer_validation(
        project_dir,
        consumers=list(runtime_retained_consumers(project_dir)),
    )


def _runtime_producer_manifests(project_dir: Path) -> tuple[Path, ...]:
    manifests: list[Path] = []
    for step in list_steps_sorted(project_dir):
        roots = [open_loop_dir(step), *list_member_dirs(step / "ensembles", "prior")]
        for member in roots:
            path = member_run_manifest_path(member)
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"Runtime producer manifest is missing: {path}")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Runtime producer manifest is unreadable: {path}") from exc
            if (
                not isinstance(payload, dict)
                or payload.get("status") != "success"
                or payload.get("member") != member.name
            ):
                raise RuntimeError(f"Runtime producer is not quiescent and successful: {path}")
            manifests.append(path.resolve())
    if not manifests:
        raise RuntimeError(f"No runtime producer manifests found under {project_dir}")
    return tuple(dict.fromkeys(manifests))


def _clean_runtime_generation(
    project_dir: Path,
    *,
    apply: bool,
) -> CleanupResult:
    """Preview or delete one generation-owned compact runtime tree."""
    accounted_bytes, accounted_files = runtime_accounted_totals(project_dir)
    if not apply:
        return CleanupResult(
            project_dir=project_dir,
            status=WorkflowStatus.PREVIEW,
            applied=False,
            eligible_paths=(),
            deleted_paths=(),
            failures=(),
            eligible_bytes=accounted_bytes,
            freed_bytes=0,
            eligible_count=accounted_files,
            deleted_count=0,
        )
    consumers, consumer_inventory = runtime_consumer_validation_evidence(project_dir)
    producers = _runtime_producer_manifests(project_dir)
    start_runtime_tree_cleanup(
        project_dir,
        retained_consumers=consumers,
        producer_manifests=producers,
        retained_consumer_inventory=consumer_inventory,
    )
    workers = int(os.environ.get("OPENAMUNDSEN_DA_CLEANUP_WORKERS", "8"))
    if workers < 1:
        raise ValueError("OPENAMUNDSEN_DA_CLEANUP_WORKERS must be at least one")
    record = apply_runtime_tree_cleanup(project_dir, workers=workers)
    return CleanupResult(
        project_dir=project_dir,
        status=WorkflowStatus.APPLIED,
        applied=True,
        eligible_paths=(),
        deleted_paths=(),
        failures=(),
        eligible_bytes=int(record.get("accounted_bytes", accounted_bytes)),
        freed_bytes=int(record.get("accounted_bytes", accounted_bytes)),
        eligible_count=int(record.get("accounted_files", accounted_files)),
        deleted_count=int(record.get("deleted_files", 0)),
    )


def clean_project_artifacts(project_dir: Path, *, apply: bool) -> CleanupResult:
    """Preview or delete safe single-domain restart artifacts."""
    project_dir = Path(project_dir).resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    runtime = load_runtime_generation(project_dir)
    if runtime is not None and runtime.get("layout") == RUNTIME_LAYOUT:
        return _clean_runtime_generation(project_dir, apply=apply)
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

    generation: int | None = None
    active_generation = active_retention_generation(project_dir)
    if candidates:
        producer_manifests: set[Path] = set()
        retained_consumers: set[Path] = set()
        if active_generation is not None and active_generation[1] == "planned":
            recorded_consumers, recorded_producers = (
                planned_retention_generation_dependencies(project_dir)
            )
            retained_consumers.update(recorded_consumers)
            producer_manifests.update(recorded_producers)
        for artifact_class, class_paths in classes.items():
            if not class_paths:
                continue
            retained_consumers.update(
                _retained_consumers_for_class(
                    project_dir,
                    artifact_class=artifact_class,
                )
            )
            producer_manifests.update(
                _forcing_plot_producer_manifests(project_dir, class_paths)
                if artifact_class == "derived_forcing_plot"
                else _member_run_manifests(project_dir, class_paths)
            )
        generation = start_retention_generation(
            project_dir,
            source_paths=candidates,
            retained_consumers=tuple(sorted(retained_consumers)),
            producer_manifests=tuple(sorted(producer_manifests)),
        )
    elif active_generation is not None and active_generation[1] == "planned":
        generation = active_generation[0]

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
            consumers = _retained_consumers_for_class(
                project_dir,
                artifact_class=artifact_class,
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
                generation=generation,
            )
            deleted.extend(path for path in class_paths if not path.exists())
            freed += sum(sizes[path] for path in class_paths if not path.exists())
        except Exception as exc:  # noqa: BLE001
            if artifact_class == "derived_forcing_plot":
                forcing_plot_failed = True
            failures.extend(CleanupFailure(path=path, error=str(exc)) for path in class_paths if path.exists())
    if generation is not None and not failures:
        complete_retention_generation(project_dir, generation=generation)
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
