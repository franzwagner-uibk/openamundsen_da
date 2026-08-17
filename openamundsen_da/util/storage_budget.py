"""Fixed, conservative disk-admission policy for project steps."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import stat as stat_module
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from ruamel.yaml.error import YAMLError

from openamundsen_da.exceptions import LowDiskEmergencyError, LowDiskPauseError
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    find_project_yaml,
    find_setup_yaml,
    list_steps_sorted,
    read_step_config,
)
from openamundsen_da.pipeline.project_skeleton import plan_project_steps
from openamundsen_da.util.roi_grid import resolve_setup_grid_spec
from openamundsen_da.util.source_catalog import SourceCatalog


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
PARENT_RENDER_MIN_BYTES = 512 * 1024**2
PARENT_RENDER_BYTES_PER_CELL_EVENT = 8
# Calibrated from 136.28 GB for 4,555 station-leaf identities, ES30 and a
# nine-month archived Euregio run.  The 4,400-byte rate projects about 372 GB
# for ES50 over a complete leap hydrological year; OBSERVED_REFIT_MARGIN then
# reserves about 465 GB until each compact leaf is finalized.
FORCING_PLOT_BYTES_PER_STATION_MEMBER_DAY = 4_400
EUREGIO_ES30_RETAINED_DIAGNOSTICS_BYTES = 8_010_000_000
EUREGIO_AUDIT_LEAF_COUNT = 90
EUREGIO_AUDIT_MEMBER_COUNT = 31


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
    derived_forcing_plot_bytes: int = 0
    retained_diagnostics_bytes: int = 0

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
            + self.derived_forcing_plot_bytes
            + self.retained_diagnostics_bytes
        )

    @property
    def total_bytes(self) -> int:
        return self.non_transition_bytes + self.restart_transition_bytes

    @property
    def retained_compact_bytes(self) -> int:
        """Return outputs that remain after successful compact cleanup."""
        return (
            self.compact_timeseries_bytes
            + self.compact_grid_bytes
            + self.map_support_bytes
            + self.retained_diagnostics_bytes
        )


@dataclass(frozen=True)
class StorageReservationProject:
    """One leaf participating in a shared-filesystem reservation."""

    setup_dir: Path
    project_dir: Path
    grid_cell_count: int
    run_manifest: Path | None = None
    completion_not_before_ns: int = 0
    scientific_input_paths: tuple[Path, ...] = ()
    scientific_root: Path | None = None
    preparation_bytes: int = 0
    requires_preparation: bool = False


@dataclass(frozen=True)
class _OwnedFileMetadata:
    """One regular package-owned file captured without following symlinks."""

    path: Path
    size: int
    device: int
    inode: int


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
    source_catalog: SourceCatalog | None = None,
) -> int:
    """Estimate generated forcing bytes from source size and temporal coverage."""
    meteo_dir = Path(meteo_dir)
    if source_catalog is not None:
        return source_catalog.estimate_step_forcing_bytes(
            meteo_dir,
            start=start,
            end=end,
            ensemble_size=ensemble_size,
        )
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
    """Conservatively reserve one raw-byte equivalent for compact exports.

    This standalone helper has no retention context, so mutable-path
    disappearance remains a strict error rather than an implicit fallback.
    """
    project_dir = Path(project_dir).resolve()
    patterns = (
        "steps/step_*/ensembles/*/*/results/point_*.csv",
        "steps/step_*/ensembles/*/*/meteo/*.csv",
    )
    paths = {
        path
        for pattern in patterns
        for path in project_dir.glob(pattern)
    }
    files, _stable = _owned_file_snapshot(list(paths), root=project_dir)
    # Compression normally makes the NetCDF smaller than the CSV source. Keep
    # ten percent for metadata, temporary files and sparse point variables.
    estimate = int(sum(item.size for item in files) * COMPACT_OUTPUT_MARGIN)
    _verify_owned_file_snapshot(
        files,
        root=project_dir,
        tolerate_disappearance=False,
    )
    return estimate


def _owned_file_snapshot(
    paths: list[Path],
    *,
    root: Path,
    tolerate_disappearance: bool = False,
) -> tuple[tuple[_OwnedFileMetadata, ...], bool]:
    """Capture contained regular files once per inode with an explicit race policy."""
    if not paths:
        return (), True
    root = root.resolve(strict=True)
    root_device = int(root.stat().st_dev)
    seen: set[tuple[int, int]] = set()
    files: list[_OwnedFileMetadata] = []
    stable = True
    for path in paths:
        path = Path(path)
        try:
            path.absolute().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Owned file path is outside its project root: {path}") from exc
        try:
            resolved_parent = path.parent.resolve(strict=True)
            resolved_parent.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Owned file path escapes its project root through an ancestor: {path}"
            ) from exc
        except FileNotFoundError:
            if not tolerate_disappearance:
                raise
            stable = False
            continue
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            if not tolerate_disappearance:
                raise
            stable = False
            continue
        if not stat_module.S_ISREG(metadata.st_mode):
            continue
        if int(metadata.st_dev) != root_device:
            raise ValueError(
                f"Owned file path is on a different filesystem than its project root: {path}"
            )
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if identity in seen:
            continue
        seen.add(identity)
        files.append(
            _OwnedFileMetadata(
                path=path,
                size=int(metadata.st_size),
                device=identity[0],
                inode=identity[1],
            )
        )
    return tuple(files), stable


def _verify_owned_file_snapshot(
    files: tuple[_OwnedFileMetadata, ...],
    *,
    root: Path,
    tolerate_disappearance: bool,
) -> bool:
    """Verify identities after callers finish deriving existing-byte credit.

    Size growth in an actively written file does not invalidate the identity;
    the next boundary can refit the observed high-water mark upward.
    """
    if not files:
        return True
    root = root.resolve(strict=True)
    root_device = int(root.stat().st_dev)
    stable = True
    for item in files:
        try:
            resolved_parent = item.path.parent.resolve(strict=True)
            resolved_parent.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "Owned file path escapes its project root through an ancestor: "
                f"{item.path}"
            ) from exc
        except FileNotFoundError:
            if not tolerate_disappearance:
                raise
            stable = False
            continue
        try:
            metadata = item.path.lstat()
        except FileNotFoundError:
            if not tolerate_disappearance:
                raise
            stable = False
            continue
        if int(metadata.st_dev) != root_device:
            raise ValueError(
                "Owned file path moved to a different filesystem than its project root: "
                f"{item.path}"
            )
        if (
            not stat_module.S_ISREG(metadata.st_mode)
            or int(metadata.st_ino) != item.inode
        ):
            if not tolerate_disappearance:
                raise RuntimeError(
                    f"Owned file identity changed during storage estimation: {item.path}"
                )
            stable = False
    return stable


def _owned_file_bytes(
    paths: list[Path],
    *,
    root: Path,
    tolerate_disappearance: bool = False,
) -> int:
    """Return allocated payload bytes once per inode."""
    files, stable = _owned_file_snapshot(
        paths,
        root=root,
        tolerate_disappearance=tolerate_disappearance,
    )
    total = sum(item.size for item in files)
    stable = _verify_owned_file_snapshot(
        files,
        root=root,
        tolerate_disappearance=tolerate_disappearance,
    ) and stable
    if tolerate_disappearance and not stable:
        return 0
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


def storage_project_steps(
    setup_dir: Path,
    project_dir: Path,
) -> list[tuple[Path, datetime, datetime]]:
    """Return authoritative materialized or virtual storage-planning steps.

    A pristine or safely repairable preparation tree uses the immutable virtual
    windows from project configuration. Runtime evidence or completed
    preparation authority makes missing/invalid materialized steps fail closed.
    """
    leaf_preparation = setup_dir / ".openamundsen-da/manifests/leaf_preparation.json"

    def has_preparation_authority() -> bool:
        if not leaf_preparation.is_file():
            return False
        try:
            return (
                json.loads(leaf_preparation.read_text(encoding="utf-8")).get("status")
                == "success"
            )
        except (OSError, json.JSONDecodeError):
            return True

    steps_root = project_dir / "steps"
    if not steps_root.exists():
        run_manifest = setup_dir / "run_manifest.json"
        if run_manifest.exists():
            try:
                manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
                status = str(manifest["status"]).strip().lower()
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Cannot plan missing project steps because the leaf run manifest "
                    f"is invalid: {run_manifest}"
                ) from exc
            if status not in {"running", "failed", "paused_low_disk", "success"}:
                raise ValueError(
                    "Cannot plan missing project steps because the leaf run manifest "
                    f"has unsupported status {status!r}: {run_manifest}"
                )
            raise FileNotFoundError(
                "Prepared steps are missing after the leaf project started "
                f"(status={status!r}): {project_dir}"
            )
        if has_preparation_authority():
            raise FileNotFoundError(
                "Prepared steps are missing despite authoritative leaf preparation: "
                f"{project_dir}"
            )
        return [
            (project_dir / "steps" / plan.name, plan.start, plan.end)
            for plan in plan_project_steps(setup_dir, project_dir)
        ]
    if not steps_root.is_dir():
        raise FileNotFoundError(f"Steps path is not a directory: {steps_root}")

    runtime_patterns = (
        "step_*/assim/prior_forcing_manifest.json",
        "step_*/assim/rejuvenate_manifest.json",
        "step_*/ensembles/*/*/results/member_run.json",
    )
    has_runtime_evidence = any(
        next(steps_root.glob(pattern), None) is not None
        for pattern in runtime_patterns
    )
    preparation_is_authoritative = has_preparation_authority()
    try:
        materialized = list_steps_sorted(project_dir)
    except (FileNotFoundError, ValueError, YAMLError):
        if has_runtime_evidence or preparation_is_authoritative:
            raise
        materialized = []
    if not materialized:
        if has_runtime_evidence or preparation_is_authoritative:
            raise FileNotFoundError(
                f"Prepared steps directory is empty or invalid: {steps_root}"
            )
        return [
            (project_dir / "steps" / plan.name, plan.start, plan.end)
            for plan in plan_project_steps(setup_dir, project_dir)
        ]
    try:
        virtual = [
            (project_dir / "steps" / plan.name, plan.start, plan.end)
            for plan in plan_project_steps(setup_dir, project_dir)
        ]
    except (ValueError, FileNotFoundError):
        virtual = []
    if not virtual:
        windows: list[tuple[Path, datetime, datetime]] = []
        for step in materialized:
            step_cfg = read_step_config(step) or {}
            try:
                windows.append(
                    (
                        step,
                        datetime.fromisoformat(str(step_cfg["start_date"])),
                        datetime.fromisoformat(str(step_cfg["end_date"])),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Cannot estimate storage for invalid step window in {step}") from exc
        return windows
    expected_by_name = {path.name: (path, start, end) for path, start, end in virtual}
    unexpected = [step.name for step in materialized if step.name not in expected_by_name]
    if unexpected:
        raise ValueError(
            "Prepared steps differ from the immutable virtual plan: "
            + ", ".join(unexpected)
        )
    for step in materialized:
        try:
            step_cfg = read_step_config(step) or {}
        except (FileNotFoundError, YAMLError):
            if has_runtime_evidence or preparation_is_authoritative:
                raise
            return virtual
        try:
            start = datetime.fromisoformat(str(step_cfg["start_date"]))
            end = datetime.fromisoformat(str(step_cfg["end_date"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot estimate storage for invalid step window in {step}") from exc
        _expected_path, expected_start, expected_end = expected_by_name[step.name]
        if start != expected_start or end != expected_end:
            raise ValueError(f"Prepared step differs from virtual window: {step}")
    return virtual


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


def _forcing_plot_storage_bound(
    *,
    setup_dir: Path,
    project_dir: Path,
    steps: list[tuple[Path, datetime, datetime]],
    member_count: int,
    tolerate_disappearance: bool = False,
) -> int:
    """Reserve derived per-step forcing PNGs until leaf finalization."""
    station_count = _station_count(setup_dir)
    if station_count == 0:
        return 0
    plot_days = sum(
        _window_sample_count(start, end, "1D")
        for _step, start, end in steps
    )
    expected = int(
        station_count
        * member_count
        * plot_days
        * FORCING_PLOT_BYTES_PER_STATION_MEMBER_DAY
        * OBSERVED_REFIT_MARGIN
    )
    existing = _owned_file_bytes(
        list(project_dir.glob("steps/step_*/plots/forcing/*.png")),
        root=project_dir,
        tolerate_disappearance=tolerate_disappearance,
    )
    return max(0, expected - existing)


def _retained_diagnostics_storage_bound(
    *,
    project_dir: Path,
    member_count: int,
) -> int:
    """Reserve retained logs, diagnostics, metadata and rendered leaf output."""
    calibrated = int(
        (EUREGIO_ES30_RETAINED_DIAGNOSTICS_BYTES / EUREGIO_AUDIT_LEAF_COUNT)
        * (member_count / EUREGIO_AUDIT_MEMBER_COUNT)
        * OBSERVED_REFIT_MARGIN
    )
    paths: set[Path] = set()
    for pattern in (
        "steps/step_*/assim/**/*",
        "steps/step_*/ensembles/*/*/results/member_run.json",
        "steps/step_*/ensembles/*/*/meteo/stations.csv",
        "**/*.log",
    ):
        paths.update(
            path.resolve()
            for path in project_dir.glob(pattern)
            if path.is_file() and not path.is_symlink()
        )
    paths.update(
        path.resolve()
        for path in project_dir.glob("steps/step_*/plots/**/*")
        if path.is_file()
        and not path.is_symlink()
        and "forcing" not in path.relative_to(project_dir).parts
    )
    for directory in ("benchmark", "maps", "misc", "plots", "reports"):
        root = project_dir / "results" / directory
        paths.update(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    paths.update(
        path.resolve()
        for path in (project_dir / "results").glob("*")
        if path.is_file() and not path.is_symlink()
    )
    existing = _owned_file_bytes(list(paths), root=project_dir)
    expected = max(calibrated, int(existing * OBSERVED_REFIT_MARGIN))
    return max(0, expected - existing)


def _point_storage_bound(
    *,
    setup_dir: Path,
    setup_cfg: dict,
    output_data: dict,
    steps: list[tuple[Path, datetime, datetime]],
    model_timestep: object,
    member_count: int,
    project_dir: Path | None = None,
    tolerate_disappearance: bool = False,
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
    observed_rate = (
        _observed_point_bytes_per_value(
            project_dir,
            tolerate_disappearance=tolerate_disappearance,
        )
        if project_dir is not None
        else 0.0
    )
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


def _observed_point_bytes_per_value(
    project_dir: Path | None,
    *,
    tolerate_disappearance: bool = False,
) -> float:
    """Calibrate the point bound upward from already produced CSVs."""
    if project_dir is None:
        return 0.0
    project_dir = Path(project_dir).resolve()
    paths = list(
        project_dir.glob("steps/step_*/ensembles/*/*/results/point_*.csv")
    )
    files, _stable_snapshot = _owned_file_snapshot(
        paths,
        root=project_dir,
        tolerate_disappearance=tolerate_disappearance,
    )
    measured = 0.0
    for item in files:
        try:
            with item.path.open(
                "r",
                encoding="utf-8-sig",
                errors="strict",
                newline="",
            ) as stream:
                metadata = os.fstat(stream.fileno())
                if int(metadata.st_dev) != item.device:
                    raise ValueError(
                        "Owned file path moved to a different filesystem than its "
                        f"project root: {item.path}"
                    )
                if (
                    not stat_module.S_ISREG(metadata.st_mode)
                    or int(metadata.st_ino) != item.inode
                ):
                    if not tolerate_disappearance:
                        raise RuntimeError(
                            "Owned file identity changed during storage estimation: "
                            f"{item.path}"
                        )
                    continue
                size = int(metadata.st_size)
                reader = csv.reader(stream)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                rows = sum(1 for row in reader if row)
        except FileNotFoundError:
            if not tolerate_disappearance:
                raise
            continue
        values = rows * max(1, len(header) - 1)
        if values:
            measured = max(measured, size / values)
    _verify_owned_file_snapshot(
        files,
        root=project_dir,
        tolerate_disappearance=tolerate_disappearance,
    )
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
    tolerate_disappearance: bool = False,
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
    existing_paths = list(
        project_dir.glob("steps/step_*/ensembles/*/*/results/output_grids*.nc")
    )
    existing_files, stable_snapshot = _owned_file_snapshot(
        existing_paths,
        root=project_dir,
        tolerate_disappearance=tolerate_disappearance,
    )
    measured_rate = 0.0
    for item in existing_files:
        try:
            step_name = item.path.relative_to(project_dir / "steps").parts[0]
        except (ValueError, IndexError):
            continue
        samples = sample_counts.get(step_name, 0)
        if samples > 0:
            measured_rate = max(
                measured_rate,
                item.size / (grid_cell_count * samples),
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
    stable_snapshot = _verify_owned_file_snapshot(
        existing_files,
        root=project_dir,
        tolerate_disappearance=tolerate_disappearance,
    ) and stable_snapshot
    existing = (
        0
        if tolerate_disappearance and not stable_snapshot
        else sum(item.size for item in existing_files)
    )
    return max(0, expected - existing), existing


def _restart_storage_bound(
    *,
    project_dir: Path,
    grid_cell_count: int,
    member_count: int,
    step_count: int,
    compact: bool,
    state_pattern: str,
    overwrite: bool = False,
) -> tuple[int, int]:
    state_paths = list(
        project_dir.glob(
            f"steps/step_*/ensembles/*/*/results/{state_pattern}"
        )
    )
    newest_step: str | None = None
    if compact and state_paths:
        # Only the newest checkpoint generation is durable input to the next
        # transition. Older generations are eligible for concurrent rolling
        # cleanup and must therefore never reduce the projected-growth reserve.
        step_root = project_dir / "steps"
        state_paths_by_step: dict[str, list[Path]] = {}
        for path in state_paths:
            try:
                step_name = path.relative_to(step_root).parts[0]
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Restart state path is outside the project steps: {path}"
                ) from exc
            state_paths_by_step.setdefault(step_name, []).append(path)
        if len(state_paths_by_step) == 1:
            newest_step = next(iter(state_paths_by_step))
        else:
            step_rank = {
                step.name: index
                for index, step in enumerate(list_steps_sorted(project_dir))
            }
            unknown_steps = sorted(set(state_paths_by_step) - set(step_rank))
            if unknown_steps:
                raise ValueError(
                    "Restart state paths belong to unknown project steps: "
                    + ", ".join(unknown_steps)
                )
            newest_step = max(state_paths_by_step, key=step_rank.__getitem__)

    state_files, stable_snapshot = _owned_file_snapshot(
        state_paths,
        root=project_dir,
        tolerate_disappearance=compact,
    )
    existing_files = state_files
    if compact and newest_step is not None:
        existing_files = tuple(
            item
            for item in state_files
            if item.path.relative_to(step_root).parts[0] == newest_step
        )
    existing = sum(item.size for item in existing_files)
    # Every observed generation remains useful as an upward size high-water,
    # even though only the newest generation receives existing-byte credit.
    largest_checkpoint = max((item.size for item in state_files), default=0)
    measured_per_checkpoint = 0.0
    if largest_checkpoint:
        measured_per_checkpoint = (
            largest_checkpoint
            * member_count
            * OBSERVED_REFIT_MARGIN
        )
    checkpoint_bound = max(
        grid_cell_count * member_count * STATE_BYTES_PER_CELL_MEMBER,
        int(measured_per_checkpoint),
    )
    stable_snapshot = _verify_owned_file_snapshot(
        state_files,
        root=project_dir,
        tolerate_disappearance=compact,
    ) and stable_snapshot
    if compact and not stable_snapshot:
        # A final cleanup can remove the newest generation while another leaf
        # recomputes the coordinated reserve. Do not credit any bytes from that
        # changing generation; the fixed/observed checkpoint bound still
        # reserves a complete replacement conservatively.
        existing = 0
    if compact:
        baseline_expected = checkpoint_bound if step_count else 0
        transition_expected = checkpoint_bound if step_count > 1 else 0
    else:
        baseline_expected = checkpoint_bound * step_count
        transition_expected = 0
    if overwrite:
        # Each accepted checkpoint remains in place until its validated
        # same-directory replacement is promoted. Reserve the entire new
        # generation rather than subtracting the old bytes already in use.
        return baseline_expected, transition_expected
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
    source_catalog: SourceCatalog | None = None,
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
    steps = storage_project_steps(setup_dir, project_dir)
    model_timestep = setup_cfg.get("timestep") or "1h"
    output_data = _merged_output_data(setup_cfg, project_cfg)
    da_cfg = project_cfg.get("data_assimilation") or {}
    output_cfg = da_cfg.get("output") or {}
    default_retention = (
        "full"
        if str(project_cfg.get("run_mode", "")).lower() == "subdomain"
        else "compact"
    )
    retention = str(output_cfg.get("retention", default_retention)).strip().lower()
    if retention not in {"compact", "full"}:
        raise ValueError(f"Invalid output retention in {project_yaml}: {retention!r}")
    compact = retention == "compact"

    forcing_expected = sum(
        estimate_step_forcing_bytes(
            setup_dir / "meteo",
            start=start,
            end=end,
            ensemble_size=ensemble_size,
            source_catalog=source_catalog,
        )
        for _step, start, end in steps
    )
    forcing_existing = _owned_file_bytes(
        [
            path
            for path in project_dir.glob("steps/step_*/ensembles/*/*/meteo/*.csv")
            if path.name != "stations.csv"
        ],
        root=project_dir,
        tolerate_disappearance=compact,
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
        tolerate_disappearance=compact,
    )

    point_expected = _point_storage_bound(
        setup_dir=setup_dir,
        setup_cfg=setup_cfg,
        output_data=output_data,
        steps=steps,
        model_timestep=model_timestep,
        member_count=member_count,
        project_dir=project_dir,
        tolerate_disappearance=compact,
    )
    point_existing = _owned_file_bytes(
        [
            path
            for path in project_dir.glob("steps/step_*/ensembles/*/*/results/point_*.csv")
        ],
        root=project_dir,
        tolerate_disappearance=compact,
    )
    point_additional = max(0, point_expected - point_existing)

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
        overwrite=overwrite,
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
    derived_forcing_plots = _forcing_plot_storage_bound(
        setup_dir=setup_dir,
        project_dir=project_dir,
        steps=steps,
        member_count=member_count,
        tolerate_disappearance=compact,
    )
    retained_diagnostics = _retained_diagnostics_storage_bound(
        project_dir=project_dir,
        member_count=member_count,
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
        derived_forcing_plot_bytes=derived_forcing_plots,
        retained_diagnostics_bytes=retained_diagnostics,
    )


def estimate_project_storage_reserve(
    *,
    setup_dir: str | Path,
    project_dir: str | Path,
    overwrite: bool = False,
    grid_cell_count: int | None = None,
    source_catalog: SourceCatalog | None = None,
) -> int:
    """Estimate all additional bytes needed to finish one project safely."""
    return estimate_project_storage_components(
        setup_dir=setup_dir,
        project_dir=project_dir,
        overwrite=overwrite,
        grid_cell_count=grid_cell_count,
        source_catalog=source_catalog,
    ).total_bytes


def estimate_coordinated_storage_reserve(
    projects: tuple[StorageReservationProject, ...],
    *,
    outer_workers: int,
    parent_finalization_reserve_bytes: int,
    overwrite: bool = False,
    source_catalog: SourceCatalog | None = None,
    progress: Callable[[int, int, Path, str], None] | None = None,
) -> tuple[int, dict[str, ProjectStorageEstimate]]:
    """Reserve one admitted leaf cohort plus unfinished parent finalization.

    Successful compact leaves are cleaned before the next cohort is admitted.
    Their retained bytes are already represented in filesystem usage. Only the
    rolling second checkpoint within the active cohort is concurrency-bound.
    """
    estimates: dict[str, ProjectStorageEstimate] = {}
    total_projects = len(projects)
    for project_index, project in enumerate(projects, start=1):
        if progress is not None:
            progress(
                project_index,
                total_projects,
                project.project_dir,
                "start",
            )
        if project.run_manifest is not None and project.run_manifest.is_file():
            try:
                data = json.loads(project.run_manifest.read_text(encoding="utf-8"))
                completed_during_reservation = (
                    project.completion_not_before_ns <= 0
                    or project.run_manifest.stat().st_mtime_ns
                    >= project.completion_not_before_ns
                )
                finalization_path = project.setup_dir / "leaf_finalization_manifest.json"
                try:
                    finalization = json.loads(
                        finalization_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError, TypeError):
                    finalization = {}
                if (
                    str(data.get("status", "")).lower() == "success"
                    and isinstance(data.get("scientific_identity"), str)
                    and finalization.get("status") == "success"
                    and finalization.get("scientific_identity")
                    == data.get("scientific_identity")
                    and (not overwrite or completed_during_reservation)
                ):
                    if progress is not None:
                        progress(
                            project_index,
                            total_projects,
                            project.project_dir,
                            "complete",
                        )
                    continue
            except (OSError, ValueError, TypeError):
                pass
        estimate = estimate_project_storage_components(
            setup_dir=project.setup_dir,
            project_dir=project.project_dir,
            overwrite=overwrite,
            grid_cell_count=project.grid_cell_count,
            source_catalog=source_catalog,
        )
        estimates[str(project.project_dir)] = estimate
        if progress is not None:
            progress(
                project_index,
                total_projects,
                project.project_dir,
                "complete",
            )
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
        + max(0, int(parent_finalization_reserve_bytes))
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
    try:
        start = datetime.fromisoformat(str(project_cfg["start_date"]))
        end = datetime.fromisoformat(str(project_cfg["end_date"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Cannot budget parent compact merge because the top-level project "
            "start_date/end_date are invalid"
        ) from exc
    if end < start:
        raise ValueError("Cannot budget parent compact merge for an invalid project window")
    model_timestep = setup_cfg.get("timestep") or "1h"
    total_samples = _configured_compact_grid_samples(
        setup_output_data=output_data,
        setup_cfg=setup_cfg,
        project_cfg=project_cfg,
        start=start,
        end=end,
        model_timestep=model_timestep,
    )
    if total_samples == 0:
        raise ValueError(
            "Cannot budget parent compact merge because no DA grid metrics are configured"
        )
    return int(
        grid_cell_count * total_samples * GRID_BYTES_PER_CELL_SAMPLE * OBSERVED_REFIT_MARGIN
        + FILE_OVERHEAD_BYTES
    )


def estimate_parent_render_bytes(
    *,
    project_dir: str | Path,
    grid_cell_count: int,
    overwrite: bool = False,
) -> int:
    """Reserve parent maps, plots and reports before the render stage.

    The first-run bound treats each configured event as one full-grid RGBA-like
    render plus a same-sized temporary. Existing render products can only refit
    the bound upward; accepted files already consume filesystem ``used`` bytes
    and therefore are not counted twice when not overwriting.
    """
    project_dir = Path(project_dir).resolve()
    if grid_cell_count < 1:
        raise ValueError("grid_cell_count must be positive")
    project_cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    events = list(
        ((project_cfg.get("data_assimilation") or {}).get("assimilation_events"))
        or []
    )
    planned = max(
        PARENT_RENDER_MIN_BYTES,
        int(
            max(1, len(events))
            * grid_cell_count
            * PARENT_RENDER_BYTES_PER_CELL_EVENT
            * OBSERVED_REFIT_MARGIN
        ),
    )
    existing_files = [
        path
        for directory in ("maps", "plots", "reports")
        for path in (project_dir / "results" / directory).rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    existing = _owned_file_bytes(existing_files, root=project_dir)
    refitted = max(planned, int(existing * OBSERVED_REFIT_MARGIN))
    return refitted if overwrite else max(0, refitted - existing)


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
    "EUREGIO_ES30_RETAINED_DIAGNOSTICS_BYTES",
    "OPERATIONAL_RESERVE_FRACTION",
    "SOFT_USED_FRACTION",
    "ProjectStorageEstimate",
    "StorageReservationProject",
    "check_step_admission",
    "estimate_compact_timeseries_bytes",
    "estimate_coordinated_storage_reserve",
    "estimate_parent_compact_merge_bytes",
    "estimate_parent_render_bytes",
    "estimate_project_storage_components",
    "estimate_project_storage_reserve",
    "estimate_step_forcing_bytes",
    "storage_project_steps",
]
