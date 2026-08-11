"""Strict setup/project configuration boundary for public workflows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from openamundsen_da.exceptions import ProjectValidationError
from openamundsen_da.io.paths import find_project_yaml, find_setup_yaml
from openamundsen_da.util.station_da import is_station_variable
from openamundsen_da.util.yaml_utils import read_yaml_mapping

_PROJECT_KEYS = {"data_assimilation", "end_date", "obs", "run_mode", "start_date"}
_OBS_KEYS = {"snowcover", "stations", "wetsnow"}
_OBS_PRODUCT_KEYS = {
    "acquisition_manifest",
    "classes",
    "dir",
    "filename_time_parser",
    "format",
    "product_tag",
    "summary_csv",
    "wet_snow_line_diagnostics_csv",
}
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
_EVENT_KEYS = {"date", "observation_time", "product", "variable"}
_RESTART_KEYS = {"dump_state", "state_pattern"}
_MODEL_GRID_FORMATS = {"geotiff", "netcdf"}
_PRIOR_FORCING_KEYS = {
    "ensemble_size",
    "mu_p",
    "random_seed",
    "sigma_p",
    "sigma_rh",
    "sigma_sw",
    "sigma_t",
}
_REJUVENATION_KEYS = {"mu_p", "seed", "sigma_p", "sigma_rh", "sigma_sw", "sigma_t"}
_RESAMPLING_KEYS = {"algorithm", "ess_threshold", "ess_threshold_ratio", "seed"}
_LIKELIHOOD_COMMON_KEYS = {
    "min_sigma",
    "min_support_coverage_ratio",
    "obs_sigma",
    "sigma_floor",
    "use_binomial",
}
_LIKELIHOOD_FRACTION_KEYS = _LIKELIHOOD_COMMON_KEYS | {"sigma_cloud_scale"}
_LIKELIHOOD_WSL_KEYS = _LIKELIHOOD_COMMON_KEYS | {
    "min_model_finite_fraction",
    "min_wet_bands",
    "min_wet_pixels_total",
}


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


def _finite_number(
    mapping: dict[str, Any],
    key: str,
    *,
    path: str,
    errors: list[str],
    minimum: float | None = None,
) -> float | None:
    raw = _required(mapping, key, path=path, errors=errors)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        errors.append(f"{path}.{key} must be numeric")
        return None
    if not math.isfinite(value):
        errors.append(f"{path}.{key} must be finite")
    elif minimum is not None and value < minimum:
        errors.append(f"{path}.{key} must be >= {minimum}")
    return value


def _nonnegative_seed(mapping: dict[str, Any], key: str, *, path: str, errors: list[str]) -> None:
    raw = _required(mapping, key, path=path, errors=errors)
    if raw is None:
        return
    try:
        value = int(raw)
    except (TypeError, ValueError):
        errors.append(f"{path}.{key} must be an integer")
        return
    if value < 0:
        errors.append(f"{path}.{key} must be non-negative")


def _validate_pf_stages(da: dict[str, Any], *, errors: list[str]) -> None:
    prior_path = "project.data_assimilation.prior_forcing"
    prior = _mapping(da.get("prior_forcing"), path=prior_path, errors=errors)
    _unknown_keys(prior, _PRIOR_FORCING_KEYS, path=prior_path, errors=errors)
    ensemble_size = _finite_number(prior, "ensemble_size", path=prior_path, errors=errors, minimum=1.0)
    if ensemble_size is not None and not float(ensemble_size).is_integer():
        errors.append(f"{prior_path}.ensemble_size must be an integer")
    _nonnegative_seed(prior, "random_seed", path=prior_path, errors=errors)
    _finite_number(prior, "mu_p", path=prior_path, errors=errors)
    for key in ("sigma_t", "sigma_p", "sigma_rh", "sigma_sw"):
        _finite_number(prior, key, path=prior_path, errors=errors, minimum=0.0)

    rejuvenation_path = "project.data_assimilation.rejuvenation"
    rejuvenation = _mapping(da.get("rejuvenation"), path=rejuvenation_path, errors=errors)
    if "rebase_open_loop" in rejuvenation:
        errors.append(
            f"{rejuvenation_path}.rebase_open_loop is unsupported; rejuvenation always rebases from setup forcing"
        )
    _unknown_keys(rejuvenation, _REJUVENATION_KEYS, path=rejuvenation_path, errors=errors)
    _nonnegative_seed(rejuvenation, "seed", path=rejuvenation_path, errors=errors)
    for key in ("sigma_t", "sigma_p", "sigma_rh", "sigma_sw"):
        if key in rejuvenation:
            _finite_number(rejuvenation, key, path=rejuvenation_path, errors=errors, minimum=0.0)
    if "mu_p" in rejuvenation:
        _finite_number(rejuvenation, "mu_p", path=rejuvenation_path, errors=errors)

    resampling_path = "project.data_assimilation.resampling"
    resampling = _mapping(da.get("resampling"), path=resampling_path, errors=errors)
    _unknown_keys(resampling, _RESAMPLING_KEYS, path=resampling_path, errors=errors)
    algorithm = _required(resampling, "algorithm", path=resampling_path, errors=errors)
    if algorithm is not None and str(algorithm).strip().lower() != "systematic":
        errors.append(f"{resampling_path}.algorithm must be systematic")
    _nonnegative_seed(resampling, "seed", path=resampling_path, errors=errors)
    has_absolute = resampling.get("ess_threshold") is not None
    has_ratio = resampling.get("ess_threshold_ratio") is not None
    if has_absolute == has_ratio:
        errors.append(
            f"{resampling_path} must define exactly one of ess_threshold or ess_threshold_ratio"
        )
    elif has_ratio:
        ratio = _finite_number(
            resampling,
            "ess_threshold_ratio",
            path=resampling_path,
            errors=errors,
        )
        if ratio is not None and not 0.0 < ratio <= 1.0:
            errors.append(f"{resampling_path}.ess_threshold_ratio must lie in (0, 1]")
    else:
        threshold = _finite_number(resampling, "ess_threshold", path=resampling_path, errors=errors)
        if threshold is not None and threshold <= 0.0:
            errors.append(f"{resampling_path}.ess_threshold must be positive")


def _validate_likelihood(da: dict[str, Any], variables: set[str], *, errors: list[str]) -> None:
    observables = variables & {"scf", "wet_snow", "wet_snow_line"}
    raw_root = da.get("likelihood")
    if raw_root is None and not observables:
        return
    root_path = "project.data_assimilation.likelihood"
    root = _mapping(raw_root, path=root_path, errors=errors)
    _unknown_keys(root, {"scf", "wet_snow", "wet_snow_line"}, path=root_path, errors=errors)

    for observable in sorted(set(root) | observables):
        if observable not in {"scf", "wet_snow", "wet_snow_line"}:
            continue
        path = f"{root_path}.{observable}"
        section = _mapping(root.get(observable), path=path, errors=errors)
        allowed = _LIKELIHOOD_WSL_KEYS if observable == "wet_snow_line" else _LIKELIHOOD_FRACTION_KEYS
        _unknown_keys(section, allowed, path=path, errors=errors)
        for key in sorted(allowed):
            _required(section, key, path=path, errors=errors)

        obs_sigma = _finite_number(section, "obs_sigma", path=path, errors=errors)
        if obs_sigma is not None and obs_sigma <= 0.0:
            errors.append(f"{path}.obs_sigma must be positive")
        min_sigma = _finite_number(section, "min_sigma", path=path, errors=errors)
        if min_sigma is not None and min_sigma <= 0.0:
            errors.append(f"{path}.min_sigma must be positive")
        _finite_number(section, "sigma_floor", path=path, errors=errors, minimum=0.0)
        coverage = _finite_number(
            section,
            "min_support_coverage_ratio",
            path=path,
            errors=errors,
        )
        if coverage is not None and not 0.0 <= coverage <= 1.0:
            errors.append(f"{path}.min_support_coverage_ratio must lie in [0, 1]")
        use_binomial = section.get("use_binomial")
        if not isinstance(use_binomial, bool):
            errors.append(f"{path}.use_binomial must be a boolean")
        elif observable == "wet_snow_line" and use_binomial:
            errors.append(f"{path}.use_binomial must be false")

        if observable == "wet_snow_line":
            finite_fraction = _finite_number(
                section,
                "min_model_finite_fraction",
                path=path,
                errors=errors,
            )
            if finite_fraction is not None and not 0.0 <= finite_fraction <= 1.0:
                errors.append(f"{path}.min_model_finite_fraction must lie in [0, 1]")
            for key in ("min_wet_pixels_total", "min_wet_bands"):
                value = _finite_number(section, key, path=path, errors=errors, minimum=0.0)
                if value is not None and not value.is_integer():
                    errors.append(f"{path}.{key} must be an integer")
        else:
            _finite_number(section, "sigma_cloud_scale", path=path, errors=errors, minimum=0.0)


def _validate_observation_product(
    *,
    name: str,
    raw: object,
    setup_dir: Path,
    require_summary: bool,
    errors: list[str],
) -> None:
    path = f"project.obs.{name}"
    section = _mapping(raw, path=path, errors=errors)
    _unknown_keys(section, _OBS_PRODUCT_KEYS, path=path, errors=errors)
    _contained_path(setup_dir, _required(section, "dir", path=path, errors=errors), path=f"{path}.dir", errors=errors)
    if require_summary:
        _contained_path(
            setup_dir,
            _required(section, "summary_csv", path=path, errors=errors),
            path=f"{path}.summary_csv",
            errors=errors,
        )
    elif section.get("summary_csv") is not None:
        _contained_path(setup_dir, section["summary_csv"], path=f"{path}.summary_csv", errors=errors)
    if section.get("acquisition_manifest") is not None:
        _contained_path(
            setup_dir,
            section["acquisition_manifest"],
            path=f"{path}.acquisition_manifest",
            errors=errors,
        )
    parser = section.get("filename_time_parser")
    if parser is not None and str(parser).strip().lower() not in {"sentinel_1", "sentinel-1", "s1"}:
        errors.append(f"{path}.filename_time_parser must be one of: sentinel_1")
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
    if "subdomain_event_filter" in da:
        errors.append(
            "project.data_assimilation.subdomain_event_filter is no longer supported. "
            "Finalize every project or leaf schedule before execution and store the complete "
            "selection in data_assimilation.assimilation_events."
        )
    if "restart" in da:
        restart = _mapping(
            da.get("restart"),
            path="project.data_assimilation.restart",
            errors=errors,
        )
        _unknown_keys(
            restart,
            _RESTART_KEYS,
            path="project.data_assimilation.restart",
            errors=errors,
        )
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
        observation_time = event.get("observation_time")
        if observation_time is not None:
            from openamundsen_da.util.observation_time import parse_utc_timestamp

            try:
                stamp = parse_utc_timestamp(observation_time, field=f"{path}.observation_time")
                if date is not None and stamp.date().isoformat() != str(date):
                    errors.append(
                        f"{path}.observation_time has UTC date {stamp.date()}, expected {date}"
                    )
            except ValueError as exc:
                errors.append(str(exc))
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
    run_mode = str(project.get("run_mode", "")).strip().lower()
    if run_mode not in {"single", "subdomain"}:
        errors.append("project.run_mode must be one of: single, subdomain")

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
    input_meteo = _mapping(input_data.get("meteo"), path="setup.input_data.meteo", errors=errors)
    if input_meteo.get("dir") is not None:
        _contained_path(
            setup_dir,
            input_meteo["dir"],
            path="setup.input_data.meteo.dir",
            errors=errors,
        )

    variables = _validate_events(project, errors=errors)
    da = project.get("data_assimilation") if isinstance(project.get("data_assimilation"), dict) else {}
    _validate_pf_stages(da, errors=errors)
    _validate_likelihood(da, variables, errors=errors)
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
        _validate_observation_product(
            name="snowcover",
            raw=obs.get("snowcover"),
            setup_dir=setup_dir,
            require_summary=run_mode == "single",
            errors=errors,
        )
    if variables & {"wet_snow", "wet_snow_line"}:
        _validate_observation_product(
            name="wetsnow",
            raw=obs.get("wetsnow"),
            setup_dir=setup_dir,
            require_summary=run_mode == "single",
            errors=errors,
        )
    _validate_uncertainty(project, variables, setup_dir, errors)

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
