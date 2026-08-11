"""Fixed, conservative disk-admission policy for project steps."""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from openamundsen_da.exceptions import LowDiskEmergencyError, LowDiskPauseError
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    find_project_yaml,
    find_setup_yaml,
    list_steps_sorted,
    read_step_config,
)
from openamundsen_da.util.roi_grid import resolve_setup_grid_spec


SOFT_USED_FRACTION = 0.80
EMERGENCY_USED_FRACTION = 0.90
OPERATIONAL_RESERVE_FRACTION = 0.05

# First-run bounds. Observed files can only increase these byte rates; they can
# never make a later estimate less conservative than the documented baseline.
GRID_BYTES_PER_CELL_SAMPLE = 8
STATE_BYTES_PER_CELL_MEMBER = 4096
POINT_BYTES_PER_VALUE = 32
# openAMUNDSEN's current default point-output contract. Keeping the names here
# makes the bound auditable and lets layered model state be counted rather than
# treating the historical total of 40 as 40 scalar columns.
DEFAULT_POINT_VARIABLES = (
    "meteo.temp", "meteo.precip", "meteo.snowfall", "meteo.rainfall",
    "meteo.rel_hum", "meteo.wind_speed", "meteo.sw_in", "meteo.sw_out",
    "meteo.lw_in", "meteo.lw_out", "meteo.sw_in_clearsky", "meteo.dir_in_clearsky",
    "meteo.diff_in_clearsky", "meteo.cloud_factor", "meteo.cloud_fraction",
    "meteo.wet_bulb_temp", "meteo.dew_point_temp", "meteo.atmos_press",
    "meteo.sat_vap_press", "meteo.vap_press", "meteo.spec_hum",
    "surface.temp", "surface.heat_flux", "surface.sens_heat_flux",
    "surface.lat_heat_flux", "surface.advective_heat_flux", "surface.albedo",
    "soil.temp", "soil.heat_flux",
    "snow.swe", "snow.depth", "snow.temp", "snow.thickness", "snow.density",
    "snow.ice_content", "snow.liquid_water_content", "snow.melt", "snow.runoff",
    "snow.sublimation", "snow.refreezing",
)
DEFAULT_POINT_VARIABLE_COUNT = len(DEFAULT_POINT_VARIABLES)
FILE_OVERHEAD_BYTES = 256 * 1024
COMPACT_OUTPUT_MARGIN = 1.10
OBSERVED_REFIT_MARGIN = 1.25


@dataclass(frozen=True)
class DiskBudgetSnapshot:
    filesystem_path: Path
    total_bytes: int
    used_bytes: int
    free_bytes: int
    estimated_growth_bytes: int
    operational_reserve_bytes: int

    @property
    def used_fraction(self) -> float:
        return self.used_bytes / self.total_bytes

    @property
    def projected_used_fraction(self) -> float:
        return (
            self.used_bytes
            + self.estimated_growth_bytes
            + self.operational_reserve_bytes
        ) / self.total_bytes


@dataclass(frozen=True)
class ProjectStorageEstimate:
    """Additional bytes required to finish one prepared project."""

    forcing_bytes: int
    member_grid_bytes: int
    point_bytes: int
    restart_baseline_bytes: int
    restart_transition_bytes: int
    compact_timeseries_bytes: int
    compact_grid_bytes: int
    map_support_bytes: int = 0

    @property
    def non_transition_bytes(self) -> int:
        return (
            self.forcing_bytes
            + self.member_grid_bytes
            + self.point_bytes
            + self.restart_baseline_bytes
            + self.compact_timeseries_bytes
            + self.compact_grid_bytes
            + self.map_support_bytes
        )

    @property
    def total_bytes(self) -> int:
        return self.non_transition_bytes + self.restart_transition_bytes


@dataclass(frozen=True)
class StorageReservationProject:
    """One leaf participating in a shared-filesystem reservation."""

    setup_dir: Path
    project_dir: Path
    grid_cell_count: int
    run_manifest: Path | None = None
    completion_not_before_ns: int = 0


def _parse_csv_timestamp(raw: str) -> datetime | None:
    value = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return None


def estimate_step_forcing_bytes(
    meteo_dir: str | Path,
    *,
    start: datetime,
    end: datetime,
    ensemble_size: int,
) -> int:
    """Estimate generated forcing bytes from source size and temporal coverage."""
    meteo_dir = Path(meteo_dir)
    if ensemble_size < 1 or end < start:
        raise ValueError("Invalid forcing estimate inputs")
    station_files = sorted(
        path for path in meteo_dir.glob("*.csv") if path.name != "stations.csv" and path.is_file()
    )
    if not station_files:
        raise FileNotFoundError(f"No station forcing CSV files found in {meteo_dir}")
    if start.tzinfo is not None:
        start = start.astimezone(timezone.utc).replace(tzinfo=None)
    if end.tzinfo is not None:
        end = end.astimezone(timezone.utc).replace(tzinfo=None)
    per_copy_payload = 0
    for path in station_files:
        per_copy_payload += _selected_forcing_source_bytes(path, start=start, end=end)
    metadata_bytes = (meteo_dir / "stations.csv").stat().st_size if (meteo_dir / "stations.csv").is_file() else 0
    # CSV formatting and uneven station coverage make exact byte prediction
    # impossible before generation. Keep a 35% conservative serialization
    # margin while scaling only the requested step window.
    per_copy = per_copy_payload + metadata_bytes
    return per_copy * (ensemble_size + 1)


def _selected_forcing_source_bytes(path: Path, *, start: datetime, end: datetime) -> int:
    """Count exact selected source-row bytes plus conservative serialization growth."""
    with path.open("rb") as stream:
        header_raw = stream.readline()
        if not header_raw:
            raise ValueError(f"Forcing CSV is empty: {path}")
        header = next(csv.reader([header_raw.decode("utf-8-sig", errors="strict")]))
        try:
            date_idx = header.index("date")
        except ValueError as exc:
            raise ValueError(f"Forcing CSV has no date column: {path}") from exc
        selected_payload = 0
        for line_number, raw_line in enumerate(stream, start=2):
            if not raw_line.strip():
                continue
            row = next(csv.reader([raw_line.decode("utf-8", errors="strict")]))
            if date_idx >= len(row):
                raise ValueError(f"Forcing row {line_number} has no date in {path}")
            timestamp = _parse_csv_timestamp(row[date_idx])
            if timestamp is None:
                raise ValueError(f"Invalid forcing timestamp on row {line_number} in {path}")
            if start <= timestamp <= end:
                selected_payload += len(raw_line)
    # Perturbation formatting can expand numeric fields. Reserve each file's
    # literal header and 35% above the exact selected source row bytes.
    return len(header_raw) + math.ceil(selected_payload * 1.35)


def estimate_compact_timeseries_bytes(project_dir: str | Path) -> int:
    """Conservatively reserve one raw-byte equivalent for compact exports."""
    project_dir = Path(project_dir).resolve()
    patterns = (
        "steps/step_*/ensembles/*/*/results/point_*.csv",
        "steps/step_*/ensembles/*/*/meteo/*.csv",
    )
    paths = {
        path.resolve()
        for pattern in patterns
        for path in project_dir.glob(pattern)
        if path.is_file() and not path.is_symlink()
    }
    # Compression normally makes the NetCDF smaller than the CSV source. Keep
    # ten percent for metadata, temporary files and sparse point variables.
    return int(sum(path.stat().st_size for path in paths) * COMPACT_OUTPUT_MARGIN)


def _owned_file_bytes(paths: list[Path]) -> int:
    """Return allocated payload bytes once per inode."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        identity = (int(stat.st_dev), int(stat.st_ino))
        if identity in seen:
            continue
        seen.add(identity)
        total += int(stat.st_size)
    return total


def _window_sample_count(start: datetime, end: datetime, frequency: object) -> int:
    if end < start:
        raise ValueError(f"Invalid output window: {start.isoformat()}..{end.isoformat()}")
    try:
        seconds = float(pd.to_timedelta(str(frequency)).total_seconds())
    except (TypeError, ValueError):
        try:
            anchored = pd.date_range(start=start, end=end, freq=str(frequency))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid output frequency: {frequency!r}") from exc
        # Include both possibly partial edge intervals conservatively.
        return max(1, len(anchored) + 2)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Invalid output frequency: {frequency!r}")
    # The model dates are inclusive. The extra endpoint also keeps anchored
    # output frequencies conservative without reproducing OA's calendar code.
    return max(1, int(math.ceil((end - start).total_seconds() / seconds)) + 1)


def _project_steps(project_dir: Path) -> list[tuple[Path, datetime, datetime]]:
    windows: list[tuple[Path, datetime, datetime]] = []
    for step in list_steps_sorted(project_dir):
        step_cfg = read_step_config(step) or {}
        try:
            start = datetime.fromisoformat(str(step_cfg["start_date"]))
            end = datetime.fromisoformat(str(step_cfg["end_date"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot estimate storage for invalid step window in {step}") from exc
        windows.append((step, start, end))
    if not windows:
        raise FileNotFoundError(f"No prepared steps found under {project_dir}")
    return windows


def _merged_output_data(setup_cfg: dict, project_cfg: dict) -> dict:
    output_data = dict(setup_cfg.get("output_data") or {})
    output_data.update(project_cfg.get("output_data") or {})
    return output_data


def _configured_grid_samples(
    *,
    output_data: dict,
    setup_cfg: dict,
    start: datetime,
    end: datetime,
    model_timestep: object,
) -> int:
    grids_cfg = output_data.get("grids") or {}
    variables = list(grids_cfg.get("variables") or [])
    if not variables:
        raise ValueError(
            "Cannot budget member-grid storage because setup.output_data.grids.variables is empty"
        )
    samples = 0
    for variable in variables:
        if not isinstance(variable, dict):
            raise ValueError("Grid output variables must be mappings for storage budgeting")
        frequency = variable.get("freq") or model_timestep
        layer_multiplier = _grid_layer_count(variable, setup_cfg)
        samples += _window_sample_count(start, end, frequency) * layer_multiplier
    return samples


def _grid_layer_count(variable: dict, setup_cfg: dict) -> int:
    """Return the configured/default third-dimension length for known fields."""
    var_name = str(variable.get("var") or "").lower()
    if var_name in {"snow.swe", "snow.depth"}:
        return 1
    if var_name.startswith("snow."):
        snow_cfg = setup_cfg.get("snow") or {}
        if str(snow_cfg.get("model") or "multilayer").lower() == "cryolayers":
            return 4
        thickness = list(snow_cfg.get("min_thickness") or [0.1, 0.2, 0.4])
        return max(1, len(thickness))
    if var_name.startswith("soil."):
        thickness = list(((setup_cfg.get("soil") or {}).get("thickness")) or [0.1, 0.2, 0.4, 0.8])
        return max(1, len(thickness))
    return 16 if "layer" in var_name else 1


def _configured_compact_grid_samples(
    *,
    setup_output_data: dict,
    setup_cfg: dict,
    project_cfg: dict,
    start: datetime,
    end: datetime,
    model_timestep: object,
) -> int:
    source_variables = {
        str(variable.get("name") or variable.get("var")): variable
        for variable in ((setup_output_data.get("grids") or {}).get("variables") or [])
        if isinstance(variable, dict)
    }
    compact_variables = list(
        ((((project_cfg.get("data_assimilation") or {}).get("output") or {}).get("grids") or {}).get("variables"))
        or []
    )
    samples = 0
    for variable in compact_variables:
        if not isinstance(variable, dict):
            continue
        source_name = str(variable.get("var") or variable.get("name") or "")
        source = source_variables.get(source_name) or {}
        frequency = source.get("freq") or model_timestep
        metrics = list(variable.get("metrics") or ["open_loop", "ens_mean", "ens_std"])
        samples += (
            _window_sample_count(start, end, frequency)
            * max(1, len(metrics))
            * _grid_layer_count(source, setup_cfg)
        )
    return samples


def _station_count(setup_dir: Path) -> int:
    stations = setup_dir / "meteo" / "stations.csv"
    if not stations.is_file():
        return 0
    with stations.open("r", encoding="utf-8-sig", errors="strict") as stream:
        return max(0, sum(1 for line in stream if line.strip()) - 1)


def _point_storage_bound(
    *,
    setup_dir: Path,
    setup_cfg: dict,
    output_data: dict,
    steps: list[tuple[Path, datetime, datetime]],
    model_timestep: object,
    member_count: int,
    project_dir: Path | None = None,
) -> int:
    timeseries = output_data.get("timeseries") or {}
    explicit_points = list(timeseries.get("points") or [])
    point_count = len(explicit_points)
    if bool(timeseries.get("add_default_points", True)):
        point_count += _station_count(setup_dir)
    if point_count == 0:
        return 0
    variables = list(timeseries.get("variables") or [])
    variable_count = sum(
        _grid_layer_count(variable, setup_cfg) if isinstance(variable, dict) else 1
        for variable in variables
    )
    if bool(timeseries.get("add_default_variables", True)):
        variable_count += sum(
            _grid_layer_count({"var": name}, setup_cfg)
            for name in DEFAULT_POINT_VARIABLES
        )
    variable_count = max(1, variable_count)
    write_frequency = timeseries.get("write_freq") or model_timestep
    time_count = sum(
        _window_sample_count(start, end, write_frequency)
        for _step, start, end in steps
    )
    observed_rate = _observed_point_bytes_per_value(project_dir) if project_dir is not None else 0.0
    byte_rate = max(
        POINT_BYTES_PER_VALUE * OBSERVED_REFIT_MARGIN,
        observed_rate * OBSERVED_REFIT_MARGIN,
    )
    return int(
        point_count
        * variable_count
        * time_count
        * member_count
        * byte_rate
    )


def _observed_point_bytes_per_value(project_dir: Path | None) -> float:
    """Calibrate the point bound upward from already produced CSVs."""
    if project_dir is None:
        return 0.0
    measured = 0.0
    for path in Path(project_dir).glob("steps/step_*/ensembles/*/*/results/point_*.csv"):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration:
                continue
            rows = sum(1 for row in reader if row)
        values = rows * max(1, len(header) - 1)
        if values:
            measured = max(measured, path.stat().st_size / values)
    return measured


def _project_grid_storage_bound(
    *,
    project_dir: Path,
    steps: list[tuple[Path, datetime, datetime]],
    output_data: dict,
    setup_cfg: dict,
    model_timestep: object,
    grid_cell_count: int,
    member_count: int,
) -> tuple[int, int]:
    sample_counts = {
        step.name: _configured_grid_samples(
            output_data=output_data,
            setup_cfg=setup_cfg,
            start=start,
            end=end,
            model_timestep=model_timestep,
        )
        for step, start, end in steps
    }
    existing_paths = [
        path
        for path in project_dir.glob("steps/step_*/ensembles/*/*/results/output_grids*.nc")
        if path.is_file() and not path.is_symlink()
    ]
    measured_rate = 0.0
    for path in existing_paths:
        try:
            step_name = path.relative_to(project_dir / "steps").parts[0]
        except (ValueError, IndexError):
            continue
        samples = sample_counts.get(step_name, 0)
        if samples > 0:
            measured_rate = max(
                measured_rate,
                path.stat().st_size / (grid_cell_count * samples),
            )
    byte_rate = max(
        float(GRID_BYTES_PER_CELL_SAMPLE),
        measured_rate * OBSERVED_REFIT_MARGIN,
    )
    expected_files = len(steps) * member_count
    expected = int(
        grid_cell_count * sum(sample_counts.values()) * member_count * byte_rate
        + expected_files * FILE_OVERHEAD_BYTES
    )
    existing = _owned_file_bytes(existing_paths)
    return max(0, expected - existing), existing


def _restart_storage_bound(
    *,
    project_dir: Path,
    grid_cell_count: int,
    member_count: int,
    step_count: int,
    compact: bool,
    state_pattern: str,
) -> tuple[int, int]:
    state_paths = [
        path
        for path in project_dir.glob(
            f"steps/step_*/ensembles/*/*/results/{state_pattern}"
        )
        if path.is_file() and not path.is_symlink()
    ]
    existing = _owned_file_bytes(state_paths)
    measured_per_checkpoint = 0.0
    if state_paths:
        measured_per_checkpoint = (
            max(path.stat().st_size for path in state_paths)
            * member_count
            * OBSERVED_REFIT_MARGIN
        )
    checkpoint_bound = max(
        grid_cell_count * member_count * STATE_BYTES_PER_CELL_MEMBER,
        int(measured_per_checkpoint),
    )
    if compact:
        baseline_expected = checkpoint_bound if step_count else 0
        transition_expected = checkpoint_bound if step_count > 1 else 0
    else:
        baseline_expected = checkpoint_bound * step_count
        transition_expected = 0
    baseline_additional = max(0, baseline_expected - existing)
    existing_after_baseline = max(0, existing - baseline_expected)
    transition_additional = max(0, transition_expected - existing_after_baseline)
    return baseline_additional, transition_additional


def _compact_grid_storage_bound(
    *,
    project_dir: Path,
    setup_output_data: dict,
    setup_cfg: dict,
    project_cfg: dict,
    steps: list[tuple[Path, datetime, datetime]],
    model_timestep: object,
    grid_cell_count: int,
    overwrite: bool,
) -> int:
    total_samples = sum(
        _configured_compact_grid_samples(
            setup_output_data=setup_output_data,
            setup_cfg=setup_cfg,
            project_cfg=project_cfg,
            start=start,
            end=end,
            model_timestep=model_timestep,
        )
        for _step, start, end in steps
    )
    if total_samples == 0:
        return 0
    expected = int(
        grid_cell_count * total_samples * GRID_BYTES_PER_CELL_SAMPLE * OBSERVED_REFIT_MARGIN
        + FILE_OVERHEAD_BYTES
    )
    output = project_dir / "results" / "grids" / "da_output_grids.nc"
    if output.is_file() and not overwrite:
        return 0
    # The writer uses a same-directory atomic temporary. During overwrite the
    # complete old output and complete replacement coexist.
    return expected


def _map_support_storage_bound(
    *,
    project_dir: Path,
    project_cfg: dict,
    grid_cell_count: int,
    overwrite: bool,
) -> int:
    """Reserve the event-map support atomic temporary only when still needed."""
    events = list((project_cfg.get("data_assimilation") or {}).get("assimilation_events") or [])
    fraction_variables = {
        str(event.get("variable", ""))
        for event in events
        if isinstance(event, dict)
        and str(event.get("variable", "")) in {"scf", "wet_snow", "wet_snow_line"}
    }
    if not fraction_variables:
        return 0
    output = project_dir / "results" / "grids" / "da_map_support.nc"
    if output.is_file() and not overwrite:
        return 0
    event_dates = {
        str(event.get("date", ""))
        for event in events
        if isinstance(event, dict) and event.get("date") is not None
    }
    field_count = 3 * int("scf" in fraction_variables)
    field_count += 3 * int(bool(fraction_variables & {"wet_snow", "wet_snow_line"}))
    return int(
        max(1, len(event_dates))
        * field_count
        * grid_cell_count
        * 4
        * OBSERVED_REFIT_MARGIN
        + FILE_OVERHEAD_BYTES
    )


def estimate_project_storage_components(
    *,
    setup_dir: str | Path,
    project_dir: str | Path,
    overwrite: bool = False,
    grid_cell_count: int | None = None,
) -> ProjectStorageEstimate:
    """Estimate all additional retained and peak-transition project bytes.

    Atomic compact outputs reserve one complete temporary when ``overwrite``
    is requested because the accepted old file remains until replacement.
    """
    setup_dir = Path(setup_dir).resolve()
    project_dir = Path(project_dir).resolve()
    project_yaml = find_project_yaml(project_dir)
    project_cfg = _read_yaml_file(project_yaml) or {}
    setup_cfg = _read_yaml_file(find_setup_yaml(setup_dir)) or {}
    try:
        ensemble_size = int(project_cfg["data_assimilation"]["prior_forcing"]["ensemble_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot estimate project storage reserve: invalid ensemble size in {project_yaml}"
        ) from exc
    member_count = ensemble_size + 1
    if grid_cell_count is None:
        spec = resolve_setup_grid_spec(setup_dir)
        grid_cell_count = int(spec.rows) * int(spec.cols)
    if grid_cell_count < 1:
        raise ValueError("grid_cell_count must be positive")
    steps = _project_steps(project_dir)
    model_timestep = setup_cfg.get("timestep") or "1h"
    output_data = _merged_output_data(setup_cfg, project_cfg)

    forcing_expected = sum(
        estimate_step_forcing_bytes(
            setup_dir / "meteo",
            start=start,
            end=end,
            ensemble_size=ensemble_size,
        )
        for _step, start, end in steps
    )
    forcing_existing = _owned_file_bytes(
        [
            path
            for path in project_dir.glob("steps/step_*/ensembles/*/*/meteo/*.csv")
            if path.name != "stations.csv"
        ]
    )
    forcing_additional = max(0, forcing_expected - forcing_existing)

    grid_additional, _grid_existing = _project_grid_storage_bound(
        project_dir=project_dir,
        steps=steps,
        output_data=output_data,
        setup_cfg=setup_cfg,
        model_timestep=model_timestep,
        grid_cell_count=grid_cell_count,
        member_count=member_count,
    )

    point_expected = _point_storage_bound(
        setup_dir=setup_dir,
        setup_cfg=setup_cfg,
        output_data=output_data,
        steps=steps,
        model_timestep=model_timestep,
        member_count=member_count,
        project_dir=project_dir,
    )
    point_existing = _owned_file_bytes(
        [
            path
            for path in project_dir.glob("steps/step_*/ensembles/*/*/results/point_*.csv")
        ]
    )
    point_additional = max(0, point_expected - point_existing)

    da_cfg = project_cfg.get("data_assimilation") or {}
    output_cfg = da_cfg.get("output") or {}
    default_retention = "full" if str(project_cfg.get("run_mode", "")).lower() == "subdomain" else "compact"
    retention = str(output_cfg.get("retention", default_retention)).strip().lower()
    if retention not in {"compact", "full"}:
        raise ValueError(f"Invalid output retention in {project_yaml}: {retention!r}")
    compact = retention == "compact"
    restart_cfg = da_cfg.get("restart") or {}
    state_pattern = str(restart_cfg.get("state_pattern") or "model_state.pickle.gz")
    if any(character in state_pattern for character in "*?[]"):
        state_pattern = "model_state.pickle.gz"
    restart_baseline, restart_transition = _restart_storage_bound(
        project_dir=project_dir,
        grid_cell_count=grid_cell_count,
        member_count=member_count,
        step_count=len(steps),
        compact=compact,
        state_pattern=state_pattern,
    )

    compact_timeseries = 0
    if compact:
        point_output = project_dir / "results" / "points" / "ensemble_points.nc"
        forcing_output = project_dir / "results" / "forcing" / "ensemble_forcing.nc"
        if overwrite or not point_output.is_file():
            compact_timeseries += int(point_expected * COMPACT_OUTPUT_MARGIN)
        if overwrite or not forcing_output.is_file():
            compact_timeseries += int(forcing_expected * COMPACT_OUTPUT_MARGIN)

    compact_grid = _compact_grid_storage_bound(
        project_dir=project_dir,
        setup_output_data=output_data,
        setup_cfg=setup_cfg,
        project_cfg=project_cfg,
        steps=steps,
        model_timestep=model_timestep,
        grid_cell_count=grid_cell_count,
        overwrite=overwrite,
    )
    map_support = _map_support_storage_bound(
        project_dir=project_dir,
        project_cfg=project_cfg,
        grid_cell_count=grid_cell_count,
        overwrite=overwrite,
    )
    return ProjectStorageEstimate(
        forcing_bytes=forcing_additional,
        member_grid_bytes=grid_additional,
        point_bytes=point_additional,
        restart_baseline_bytes=restart_baseline,
        restart_transition_bytes=restart_transition,
        compact_timeseries_bytes=compact_timeseries,
        compact_grid_bytes=compact_grid,
        map_support_bytes=map_support,
    )


def estimate_project_storage_reserve(
    *,
    setup_dir: str | Path,
    project_dir: str | Path,
    overwrite: bool = False,
    grid_cell_count: int | None = None,
) -> int:
    """Estimate all additional bytes needed to finish one project safely."""
    return estimate_project_storage_components(
        setup_dir=setup_dir,
        project_dir=project_dir,
        overwrite=overwrite,
        grid_cell_count=grid_cell_count,
    ).total_bytes


def estimate_coordinated_storage_reserve(
    projects: tuple[StorageReservationProject, ...],
    *,
    outer_workers: int,
    parent_merge_reserve_bytes: int,
    overwrite: bool = False,
) -> tuple[int, dict[str, ProjectStorageEstimate]]:
    """Reserve retained leaf growth plus rolling concurrent checkpoints.

    Member forcing, grids, point CSVs and compact leaf products accumulate
    until the subdomain render/cleanup gate, so every unfinished leaf is
    included. Only the second rolling checkpoint is concurrency-bound.
    """
    estimates: dict[str, ProjectStorageEstimate] = {}
    for project in projects:
        if project.run_manifest is not None and project.run_manifest.is_file():
            try:
                data = json.loads(project.run_manifest.read_text(encoding="utf-8"))
                completed_during_reservation = (
                    project.completion_not_before_ns <= 0
                    or project.run_manifest.stat().st_mtime_ns
                    >= project.completion_not_before_ns
                )
                if (
                    str(data.get("status", "")).lower() == "success"
                    and (not overwrite or completed_during_reservation)
                ):
                    continue
            except (OSError, ValueError, TypeError):
                pass
        estimate = estimate_project_storage_components(
            setup_dir=project.setup_dir,
            project_dir=project.project_dir,
            overwrite=overwrite,
            grid_cell_count=project.grid_cell_count,
        )
        estimates[str(project.project_dir)] = estimate
    active_count = min(max(1, int(outer_workers)), len(estimates))
    transition = sum(
        sorted(
            (estimate.restart_transition_bytes for estimate in estimates.values()),
            reverse=True,
        )[:active_count]
    )
    total = (
        sum(estimate.non_transition_bytes for estimate in estimates.values())
        + transition
        + max(0, int(parent_merge_reserve_bytes))
    )
    return total, estimates


def estimate_parent_compact_merge_bytes(
    *,
    setup_dir: str | Path,
    project_dir: str | Path,
    grid_cell_count: int,
) -> int:
    """Reserve one complete same-directory atomic parent NetCDF temporary."""
    setup_dir = Path(setup_dir).resolve()
    project_dir = Path(project_dir).resolve()
    setup_cfg = _read_yaml_file(find_setup_yaml(setup_dir)) or {}
    project_cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    output_data = _merged_output_data(setup_cfg, project_cfg)
    steps = _project_steps(project_dir)
    model_timestep = setup_cfg.get("timestep") or "1h"
    total_samples = sum(
        _configured_compact_grid_samples(
            setup_output_data=output_data,
            setup_cfg=setup_cfg,
            project_cfg=project_cfg,
            start=start,
            end=end,
            model_timestep=model_timestep,
        )
        for _step, start, end in steps
    )
    if total_samples == 0:
        raise ValueError(
            "Cannot budget parent compact merge because no DA grid metrics are configured"
        )
    return int(
        grid_cell_count * total_samples * GRID_BYTES_PER_CELL_SAMPLE * OBSERVED_REFIT_MARGIN
        + FILE_OVERHEAD_BYTES
    )


def check_step_admission(
    project_dir: str | Path,
    *,
    estimated_growth_bytes: int = 0,
    allow_existing_step_drain: bool = False,
    usage: shutil._ntuple_diskusage | None = None,
) -> DiskBudgetSnapshot:
    """Refuse a new step when fixed project-filesystem limits are exceeded."""
    project_dir = Path(project_dir).resolve()
    if estimated_growth_bytes < 0:
        raise ValueError("estimated_growth_bytes must be non-negative")
    current = usage if usage is not None else shutil.disk_usage(project_dir)
    if current.total <= 0:
        raise RuntimeError(f"Could not determine filesystem capacity for {project_dir}")
    snapshot = DiskBudgetSnapshot(
        filesystem_path=project_dir,
        total_bytes=int(current.total),
        used_bytes=int(current.used),
        free_bytes=int(current.free),
        estimated_growth_bytes=int(estimated_growth_bytes),
        operational_reserve_bytes=int(current.total * OPERATIONAL_RESERVE_FRACTION),
    )
    if snapshot.used_fraction >= EMERGENCY_USED_FRACTION:
        raise LowDiskEmergencyError(
            f"Project filesystem is at or above the fixed 90% emergency limit "
            f"({snapshot.used_fraction:.1%} used): {project_dir}"
        )
    if snapshot.used_fraction >= SOFT_USED_FRACTION and not allow_existing_step_drain:
        raise LowDiskPauseError(
            f"Project filesystem is at or above the fixed 80% step-admission limit "
            f"({snapshot.used_fraction:.1%} used): {project_dir}"
        )
    if snapshot.projected_used_fraction >= EMERGENCY_USED_FRACTION:
        raise LowDiskPauseError(
            "Step completion estimate would reach the fixed 90% emergency limit: "
            f"current={snapshot.used_fraction:.1%}, projected={snapshot.projected_used_fraction:.1%}, "
            f"estimated_growth={snapshot.estimated_growth_bytes} bytes"
        )
    return snapshot


__all__ = [
    "DiskBudgetSnapshot",
    "EMERGENCY_USED_FRACTION",
    "OPERATIONAL_RESERVE_FRACTION",
    "SOFT_USED_FRACTION",
    "ProjectStorageEstimate",
    "StorageReservationProject",
    "check_step_admission",
    "estimate_compact_timeseries_bytes",
    "estimate_coordinated_storage_reserve",
    "estimate_parent_compact_merge_bytes",
    "estimate_project_storage_components",
    "estimate_project_storage_reserve",
    "estimate_step_forcing_bytes",
]
