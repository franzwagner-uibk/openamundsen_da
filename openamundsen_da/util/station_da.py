"""Shared configuration and path helpers for ROI-based station assimilation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import abspath_relative_to, find_project_yaml
from openamundsen_da.util.config_validators import require_mapping


STATION_DA_METADATA_FILENAME = "stations_da_metadata.csv"


@dataclass(frozen=True)
class StationVariableSpec:
    """Observation/model column mapping for one station-assimilated variable."""

    event_variable: str
    obs_column: str
    model_column: str
    sigma_floor_key: str
    label: str


@dataclass(frozen=True)
class StationAssimilationConfig:
    """Resolved station DA configuration for one project."""

    obs_dir: Path
    metadata_path: Path
    default_station_uncertainty_pct: float
    min_station_uncertainty_pct: float
    hs_sigma_abs_min: float
    swe_sigma_abs_min: float
    single_station_factor: float

    def sigma_floor_for(self, variable: str) -> float:
        spec = station_variable_spec(variable)
        return float(getattr(self, spec.sigma_floor_key))


_SPECS = {
    "station_hs": StationVariableSpec(
        event_variable="station_hs",
        obs_column="snow_depth",
        model_column="snow_depth",
        sigma_floor_key="hs_sigma_abs_min",
        label="Station HS",
    ),
    "station_swe": StationVariableSpec(
        event_variable="station_swe",
        obs_column="swe",
        model_column="swe",
        sigma_floor_key="swe_sigma_abs_min",
        label="Station SWE",
    ),
}


def station_variable_spec(variable: str) -> StationVariableSpec:
    """Return the station variable spec for one supported event variable."""
    key = str(variable).strip().lower()
    if key not in _SPECS:
        raise ValueError(f"Unsupported station assimilation variable: {variable!r}")
    return _SPECS[key]


def is_station_variable(variable: str | None) -> bool:
    """Return True when the event variable is a supported station DA variable."""
    if variable is None:
        return False
    return str(variable).strip().lower() in _SPECS


def is_station_metadata_file(path: str | Path) -> bool:
    """Return True when a path points to the station DA metadata CSV."""
    return Path(path).name == STATION_DA_METADATA_FILENAME


def station_observation_csvs(obs_dir: Path) -> list[Path]:
    """Return sorted station observation CSVs, excluding station metadata."""
    return [
        csv_path
        for csv_path in sorted(Path(obs_dir).glob("*.csv"))
        if csv_path.is_file() and not is_station_metadata_file(csv_path)
    ]


def load_station_assimilation_config(setup_dir: Path, project_dir: Path) -> StationAssimilationConfig:
    """Load station DA config and resolve the setup-level observation directory."""
    cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")

    obs_cfg = require_mapping(cfg.get("obs"), path="project.obs")
    stations_obs_cfg = require_mapping(obs_cfg.get("stations"), path="project.obs.stations")
    obs_dir_raw = stations_obs_cfg.get("dir")
    if obs_dir_raw is None or str(obs_dir_raw).strip() == "":
        raise ValueError("Missing required configuration key: project.obs.stations.dir")
    obs_dir = Path(abspath_relative_to(setup_dir, str(obs_dir_raw)))

    da_cfg = require_mapping(cfg.get("data_assimilation"), path="project.data_assimilation")
    station_cfg = require_mapping(da_cfg.get("station"), path="project.data_assimilation.station")

    def _read_required_float(key: str) -> float:
        if key not in station_cfg:
            raise ValueError(f"Missing required configuration key: project.data_assimilation.station.{key}")
        raw = station_cfg.get(key)
        try:
            value = float(raw)
        except Exception as exc:
            raise ValueError(
                f"Invalid project.data_assimilation.station.{key}: {raw!r}"
            ) from exc
        if value <= 0.0:
            raise ValueError(f"project.data_assimilation.station.{key} must be > 0")
        return value

    default_pct = _read_required_float("default_station_uncertainty_pct")
    min_pct = _read_required_float("min_station_uncertainty_pct")
    hs_sigma_abs_min = _read_required_float("hs_sigma_abs_min")
    swe_sigma_abs_min = _read_required_float("swe_sigma_abs_min")
    single_station_factor = _read_required_float("single_station_factor")

    if default_pct < min_pct:
        raise ValueError(
            "project.data_assimilation.station.default_station_uncertainty_pct "
            "must be >= min_station_uncertainty_pct"
        )

    return StationAssimilationConfig(
        obs_dir=obs_dir,
        metadata_path=obs_dir / STATION_DA_METADATA_FILENAME,
        default_station_uncertainty_pct=default_pct,
        min_station_uncertainty_pct=min_pct,
        hs_sigma_abs_min=hs_sigma_abs_min,
        swe_sigma_abs_min=swe_sigma_abs_min,
        single_station_factor=single_station_factor,
    )
