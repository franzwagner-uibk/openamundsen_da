"""Strict setup/project configuration boundary for public workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openamundsen_da.exceptions import ProjectValidationError
from openamundsen_da.io.paths import find_project_yaml, find_setup_yaml
from openamundsen_da.util.station_da import is_station_variable
from openamundsen_da.util.yaml_utils import read_yaml_mapping

_PROJECT_KEYS = {"data_assimilation", "end_date", "obs", "run_mode", "start_date"}
_OBS_KEYS = {"snowcover", "stations", "wetsnow"}
_OBS_PRODUCT_KEYS = {"classes", "dir", "format", "product_tag", "summary_csv", "wet_snow_line_diagnostics_csv"}
_OBS_STATION_KEYS = {"dir"}
_DA_KEYS = {
    "assimilation_events",
    "benchmark",
    "h_of_x",
    "landcover_mask",
    "likelihood",
    "output",
    "prior_forcing",
    "rejuvenation",
    "resampling",
    "restart",
    "station",
    "subdomain_event_filter",
    "uncertainty",
    "wet_snow",
    "wet_snow_line",
}
_EVENT_KEYS = {"date", "product", "variable"}
_MODEL_GRID_FORMATS = {"geotiff", "netcdf"}


@dataclass(frozen=True)
class ProjectConfiguration:
    """Validated immutable boundary around one project and its owning setup."""

    setup_dir: Path
    project_dir: Path
    setup_yaml: Path
    project_yaml: Path
    setup: dict[str, Any]
    project: dict[str, Any]
    model_grid_format: str


def _unknown_keys(mapping: dict[str, Any], allowed: set[str], *, path: str, errors: list[str]) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        errors.append(f"Unknown configuration key(s) at {path}: {', '.join(unknown)}")


def _mapping(raw: object, *, path: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        errors.append(f"Expected mapping at {path}")
        return {}
    return raw


def _required(mapping: dict[str, Any], key: str, *, path: str, errors: list[str]) -> object | None:
    raw = mapping.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        errors.append(f"Missing required configuration key: {path}.{key}")
        return None
    return raw


def _contained_path(setup_dir: Path, raw: object, *, path: str, errors: list[str]) -> Path | None:
    if raw is None or not str(raw).strip():
        errors.append(f"Missing required configuration key: {path}")
        return None
    candidate = Path(str(raw))
    if candidate.is_absolute():
        errors.append(f"{path} must be setup-relative, got absolute path {candidate}")
        return None
    resolved = (setup_dir / candidate).resolve()
    try:
        resolved.relative_to(setup_dir)
    except ValueError:
        errors.append(f"{path} escapes the setup directory: {candidate}")
        return None
    return resolved


def _validate_observation_product(
    *,
    name: str,
    raw: object,
    setup_dir: Path,
    errors: list[str],
) -> None:
    path = f"project.obs.{name}"
    section = _mapping(raw, path=path, errors=errors)
    _unknown_keys(section, _OBS_PRODUCT_KEYS, path=path, errors=errors)
    _contained_path(setup_dir, _required(section, "dir", path=path, errors=errors), path=f"{path}.dir", errors=errors)
    _contained_path(
        setup_dir,
        _required(section, "summary_csv", path=path, errors=errors),
        path=f"{path}.summary_csv",
        errors=errors,
    )
    _required(section, "product_tag", path=path, errors=errors)
    fmt = _required(section, "format", path=path, errors=errors)
    if fmt is not None and str(fmt).strip().lower() not in _MODEL_GRID_FORMATS:
        errors.append(f"{path}.format must be one of: geotiff, netcdf")
    classes = _mapping(section.get("classes"), path=f"{path}.classes", errors=errors)
    required_classes = ("valid", "cloud", "water", "nodata") if name == "snowcover" else ("valid", "wet", "exclude")
    for key in required_classes:
        _required(classes, key, path=f"{path}.classes", errors=errors)


def _validate_events(project: dict[str, Any], *, errors: list[str]) -> set[str]:
    da = _mapping(project.get("data_assimilation"), path="project.data_assimilation", errors=errors)
    _unknown_keys(da, _DA_KEYS, path="project.data_assimilation", errors=errors)
    raw_events = da.get("assimilation_events")
    if not isinstance(raw_events, list) or not raw_events:
        errors.append("project.data_assimilation.assimilation_events must be a non-empty list")
        return set()

    variables: set[str] = set()
    seen_dates: set[str] = set()
    for index, raw in enumerate(raw_events):
        path = f"project.data_assimilation.assimilation_events[{index}]"
        event = _mapping(raw, path=path, errors=errors)
        _unknown_keys(event, _EVENT_KEYS, path=path, errors=errors)
        date = _required(event, "date", path=path, errors=errors)
        variable_raw = _required(event, "variable", path=path, errors=errors)
        if date is not None:
            date_text = str(date)
            if date_text in seen_dates:
                errors.append(f"Duplicate assimilation event date: {date_text}")
            seen_dates.add(date_text)
        if variable_raw is None:
            continue
        variable = str(variable_raw).strip().lower()
        if variable == "wet_snow_fraction":
            errors.append(f"{path}.variable uses removed alias 'wet_snow_fraction'; use 'wet_snow'")
            variable = "wet_snow"
        variables.add(variable)
        if not is_station_variable(variable):
            _required(event, "product", path=path, errors=errors)
    return variables


def _validate_uncertainty(project: dict[str, Any], variables: set[str], setup_dir: Path, errors: list[str]) -> None:
    da = project.get("data_assimilation") if isinstance(project.get("data_assimilation"), dict) else {}
    uncertainty = _mapping(da.get("uncertainty"), path="project.data_assimilation.uncertainty", errors=errors)
    required_products = set()
    if "scf" in variables:
        required_products.add("scf")
    if variables & {"wet_snow", "wet_snow_line"}:
        required_products.add("wet_snow")
    for product in sorted(required_products):
        path = f"project.data_assimilation.uncertainty.{product}"
        section = _mapping(uncertainty.get(product), path=path, errors=errors)
        enabled = _required(section, "enabled", path=path, errors=errors)
        if enabled is True:
            _contained_path(
                setup_dir,
                _required(section, "input_dir", path=path, errors=errors),
                path=f"{path}.input_dir",
                errors=errors,
            )
            _mapping(section.get("ingest"), path=f"{path}.ingest", errors=errors)
            _mapping(section.get("assimilation"), path=f"{path}.assimilation", errors=errors)


def load_project_configuration(project_dir: str | Path) -> ProjectConfiguration:
    """Load and aggregate-validate one canonical project before any writes."""
    requested = Path(project_dir).expanduser()
    if not requested.is_dir():
        raise ProjectValidationError([f"Project directory not found: {requested}"])
    resolved_project = requested.resolve()
    if resolved_project.parent.name != "projects":
        raise ProjectValidationError([f"Project directory must be directly under <setup>/projects: {resolved_project}"])
    setup_dir = resolved_project.parent.parent.resolve()

    errors: list[str] = []
    try:
        setup_yaml = find_setup_yaml(setup_dir)
    except FileNotFoundError as exc:
        errors.append(str(exc))
        setup_yaml = setup_dir / f"{setup_dir.name}.yml"
    try:
        project_yaml = find_project_yaml(resolved_project)
    except FileNotFoundError as exc:
        errors.append(str(exc))
        project_yaml = resolved_project / f"{resolved_project.name}.yml"
    if errors:
        raise ProjectValidationError(errors)

    try:
        setup = read_yaml_mapping(setup_yaml, error_cls=ProjectValidationError, context="Setup YAML root")
    except ProjectValidationError:
        raise
    try:
        project = read_yaml_mapping(project_yaml, error_cls=ProjectValidationError, context="Project YAML root")
    except ProjectValidationError:
        raise

    _unknown_keys(project, _PROJECT_KEYS, path="project", errors=errors)
    for forbidden in ("data_assimilation", "obs", "run_mode"):
        if forbidden in setup:
            errors.append(f"Setup YAML must not contain project-owned key: {forbidden}")
    for key in ("start_date", "end_date", "run_mode"):
        _required(project, key, path="project", errors=errors)

    output_data = _mapping(setup.get("output_data"), path="setup.output_data", errors=errors)
    grids = _mapping(output_data.get("grids"), path="setup.output_data.grids", errors=errors)
    fmt_raw = _required(grids, "format", path="setup.output_data.grids", errors=errors)
    model_grid_format = str(fmt_raw).strip().lower() if fmt_raw is not None else ""
    if model_grid_format not in _MODEL_GRID_FORMATS:
        errors.append("setup.output_data.grids.format must be one of: geotiff, netcdf")

    input_data = _mapping(setup.get("input_data"), path="setup.input_data", errors=errors)
    input_grids = _mapping(input_data.get("grids"), path="setup.input_data.grids", errors=errors)
    _contained_path(
        setup_dir,
        _required(input_grids, "dir", path="setup.input_data.grids", errors=errors),
        path="setup.input_data.grids.dir",
        errors=errors,
    )

    variables = _validate_events(project, errors=errors)
    obs = _mapping(project.get("obs"), path="project.obs", errors=errors)
    _unknown_keys(obs, _OBS_KEYS, path="project.obs", errors=errors)
    stations = _mapping(obs.get("stations"), path="project.obs.stations", errors=errors)
    _unknown_keys(stations, _OBS_STATION_KEYS, path="project.obs.stations", errors=errors)
    _contained_path(
        setup_dir,
        _required(stations, "dir", path="project.obs.stations", errors=errors),
        path="project.obs.stations.dir",
        errors=errors,
    )
    if "scf" in variables:
        _validate_observation_product(name="snowcover", raw=obs.get("snowcover"), setup_dir=setup_dir, errors=errors)
    if variables & {"wet_snow", "wet_snow_line"}:
        _validate_observation_product(name="wetsnow", raw=obs.get("wetsnow"), setup_dir=setup_dir, errors=errors)
    _validate_uncertainty(project, variables, setup_dir, errors)

    da = project.get("data_assimilation") if isinstance(project.get("data_assimilation"), dict) else {}
    da_output = _mapping(da.get("output"), path="project.data_assimilation.output", errors=errors)
    compact_grids = _mapping(da_output.get("grids"), path="project.data_assimilation.output.grids", errors=errors)
    compact_format = _required(compact_grids, "format", path="project.data_assimilation.output.grids", errors=errors)
    if compact_format is not None and str(compact_format).strip().lower() != "netcdf":
        errors.append("project.data_assimilation.output.grids.format must be netcdf")

    if errors:
        raise ProjectValidationError(errors)
    return ProjectConfiguration(
        setup_dir=setup_dir,
        project_dir=resolved_project,
        setup_yaml=setup_yaml.resolve(),
        project_yaml=project_yaml.resolve(),
        setup=setup,
        project=project,
        model_grid_format=model_grid_format,
    )


__all__ = ["ProjectConfiguration", "load_project_configuration"]
