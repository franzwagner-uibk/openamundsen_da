"""Shared configuration and path helpers for ROI-based station assimilation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import abspath_relative_to, find_project_yaml
from openamundsen_da.util.config_validators import require_mapping


STATION_DA_METADATA_FILENAME = "stations_da_metadata.csv"
STATION_SNOW_DEPTH_METADATA_FILENAME = "stations_snow_depth.csv"
STATION_METADATA_FILENAMES = {
    STATION_DA_METADATA_FILENAME,
    STATION_SNOW_DEPTH_METADATA_FILENAME,
}


@dataclass(frozen=True)
class StationVariableSpec:
    """Observation/model column mapping for one station-assimilated variable."""

    event_variable: str
    obs_column: str
    model_column: str
    metadata_sigma_column: str
    label: str


@dataclass(frozen=True)
class StationAssimilationConfig:
    """Resolved station DA configuration for one project."""

    obs_dir: Path
    metadata_path: Path
    default_station_uncertainty_pct: float
    min_station_uncertainty_pct: float
    single_station_factor: float


@dataclass(frozen=True)
class StationSigmaBaseResolution:
    """Base station uncertainty terms for one observation value."""

    station_uncertainty_pct: float
    uncertainty_source: str
    sigma_abs_floor: float
    sigma_base: float


_SPECS = {
    "station_hs": StationVariableSpec(
        event_variable="station_hs",
        obs_column="snow_depth",
        model_column="snow_depth",
        metadata_sigma_column="hs_sigma_abs_min",
        label="Station HS",
    ),
    "station_swe": StationVariableSpec(
        event_variable="station_swe",
        obs_column="swe",
        model_column="swe",
        metadata_sigma_column="swe_sigma_abs_min",
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
    """Return True when a path points to known station metadata CSVs."""
    return Path(path).name in STATION_METADATA_FILENAMES


def station_observation_csvs(obs_dir: Path) -> list[Path]:
    """Return sorted station observation CSVs, excluding station metadata."""
    return [
        csv_path
        for csv_path in sorted(Path(obs_dir).glob("*.csv"))
        if csv_path.is_file() and not is_station_metadata_file(csv_path)
    ]


def normalize_station_id_series(series: pd.Series) -> pd.Series:
    """Return station IDs normalized for matching without losing leading zeros."""
    return series.astype("string").fillna("").str.strip().str.lower()


def _parse_role_flag(raw: object, *, column: str, station_id: str, metadata_path: Path) -> bool:
    if pd.isna(raw) or str(raw).strip() == "":
        return True
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid {column} for station {station_id!r} in {metadata_path}: {raw!r}")


def read_station_metadata(metadata_path: Path) -> pd.DataFrame:
    """Read station DA metadata keyed by station_id, validating optional sigma columns."""
    base_columns = [
        "station_uncertainty_pct",
        "hs_sigma_abs_min",
        "swe_sigma_abs_min",
        "use_for_da",
        "use_for_benchmark",
    ]
    if not metadata_path.is_file():
        logger.warning(
            "Station DA metadata file not found: {}. Active station DA will fail until station-wise absolute sigma metadata are provided.",
            metadata_path,
        )
        return pd.DataFrame(columns=base_columns)

    df = pd.read_csv(metadata_path, dtype={"station_id": "string"})
    if df.empty:
        logger.warning(
            "Station DA metadata file is empty: {}. Active station DA will fail until station-wise absolute sigma metadata are provided.",
            metadata_path,
        )
        return pd.DataFrame(columns=base_columns)
    if "station_id" not in df.columns:
        raise ValueError(f"Station DA metadata file missing required column 'station_id': {metadata_path}")
    if "station_uncertainty_pct" not in df.columns:
        raise ValueError(f"Station DA metadata file missing required column 'station_uncertainty_pct': {metadata_path}")

    out = df.copy()
    out["station_id"] = normalize_station_id_series(out["station_id"])
    out = out[out["station_id"] != ""].copy()
    if out.empty:
        logger.warning(
            "Station DA metadata file has no usable station_id rows: {}. Active station DA will fail until station-wise absolute sigma metadata are provided.",
            metadata_path,
        )
        return pd.DataFrame(columns=base_columns)
    for col in ("hs_sigma_abs_min", "swe_sigma_abs_min"):
        if col not in out.columns:
            out[col] = np.nan
            continue
        normalized: list[float] = []
        for station_id, raw in zip(out["station_id"], out[col], strict=False):
            if pd.isna(raw) or str(raw).strip() == "":
                normalized.append(np.nan)
                continue
            try:
                value = float(raw)
            except Exception as exc:
                raise ValueError(
                    f"Invalid {col} for station {station_id!r} in {metadata_path}: {raw!r}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(f"{col} for station {station_id!r} in {metadata_path} is not finite")
            if value <= 0.0:
                raise ValueError(f"{col} for station {station_id!r} in {metadata_path} must be > 0")
            normalized.append(float(value))
        out[col] = normalized

    for col in ("use_for_da", "use_for_benchmark"):
        if col not in out.columns:
            out[col] = True
            continue
        out[col] = [
            _parse_role_flag(raw, column=col, station_id=station_id, metadata_path=metadata_path)
            for station_id, raw in zip(out["station_id"], out[col], strict=False)
        ]
    return out.drop_duplicates(subset=["station_id"], keep="last").set_index("station_id")


def station_ids_disabled_for_role(metadata_df: pd.DataFrame, role: str) -> set[str]:
    """Return station IDs explicitly disabled for a metadata role.

    Missing metadata or missing role columns keep the historical default: a
    station is usable unless the project metadata explicitly marks it false.
    """
    role_key = str(role).strip().lower()
    if role_key not in {"da", "benchmark"}:
        raise ValueError(f"Unsupported station role: {role!r}")
    column = "use_for_da" if role_key == "da" else "use_for_benchmark"
    if metadata_df.empty or column not in metadata_df.columns:
        return set()
    disabled: set[str] = set()
    for station_id, raw in metadata_df[column].items():
        if isinstance(raw, (bool, np.bool_)):
            enabled = bool(raw)
        else:
            enabled = _parse_role_flag(
                raw,
                column=column,
                station_id=str(station_id),
                metadata_path=Path("<metadata>"),
            )
        if not enabled:
            disabled.add(str(station_id).strip().lower())
    return disabled


def resolve_station_uncertainty_pct(
    station_id: str,
    metadata_df: pd.DataFrame,
    config: StationAssimilationConfig,
) -> tuple[float, str]:
    """Return effective station uncertainty percentage and its source."""
    source = "metadata"
    value = None
    station_key = str(station_id).strip().lower()
    if station_key in metadata_df.index:
        raw = metadata_df.loc[station_key, "station_uncertainty_pct"]
        if isinstance(raw, pd.Series):
            raw = raw.iloc[-1]
        if pd.isna(raw) or str(raw).strip() == "":
            source = "default"
            value = float(config.default_station_uncertainty_pct)
            logger.warning(
                "Station {} has empty station_uncertainty_pct in {}; using project default {:.3f}%",
                station_key,
                config.metadata_path,
                value,
            )
        else:
            try:
                value = float(raw)
            except Exception as exc:
                raise ValueError(
                    f"Invalid station_uncertainty_pct for station {station_key!r} in {config.metadata_path}: {raw!r}"
                ) from exc
    else:
        source = "default"
        value = float(config.default_station_uncertainty_pct)
        logger.warning(
            "Station {} missing in {}; using project default {:.3f}%",
            station_key,
            config.metadata_path,
            value,
        )

    if not np.isfinite(value):
        raise ValueError(f"Station uncertainty for {station_key!r} is not finite")
    if value <= 0.0:
        raise ValueError(f"Station uncertainty for {station_key!r} must be > 0")
    if value < float(config.min_station_uncertainty_pct):
        logger.warning(
            "Station {} uncertainty {:.3f}% below configured minimum {:.3f}%; clamping to minimum.",
            station_key,
            value,
            config.min_station_uncertainty_pct,
        )
        value = float(config.min_station_uncertainty_pct)
    return float(value), source


def resolve_station_sigma_abs_floor(
    station_id: str,
    metadata_df: pd.DataFrame,
    metadata_path: Path,
    variable: str,
) -> float:
    """Return the metadata-only absolute sigma floor for one active station."""
    spec = station_variable_spec(variable)
    column = spec.metadata_sigma_column
    station_key = str(station_id).strip().lower()
    if station_key not in metadata_df.index:
        raise ValueError(
            f"Station {station_key!r} missing in {metadata_path}. "
            f"Active {variable} assimilation requires metadata column {column!r} for every station."
        )

    raw = metadata_df.loc[station_key, column]
    if isinstance(raw, pd.Series):
        raw = raw.iloc[-1]
    if pd.isna(raw) or str(raw).strip() == "":
        raise ValueError(
            f"Station {station_key!r} missing required metadata value {column!r} in {metadata_path} "
            f"for active {variable} assimilation."
        )

    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"{column} for station {station_key!r} in {metadata_path} is not finite")
    if value <= 0.0:
        raise ValueError(f"{column} for station {station_key!r} in {metadata_path} must be > 0")
    return float(value)


def resolve_station_sigma_base(
    *,
    station_id: str,
    obs_value: float,
    variable: str,
    config: StationAssimilationConfig,
    metadata_df: pd.DataFrame,
) -> StationSigmaBaseResolution:
    """Return the base effective sigma used for station DA and sigma-aware benchmarking."""
    sigma_abs_floor = resolve_station_sigma_abs_floor(
        station_id=station_id,
        metadata_df=metadata_df,
        metadata_path=config.metadata_path,
        variable=variable,
    )
    pct, source = resolve_station_uncertainty_pct(station_id, metadata_df, config)
    sigma_rel = (pct / 100.0) * float(obs_value)
    sigma_base = float(np.hypot(sigma_rel, sigma_abs_floor))
    return StationSigmaBaseResolution(
        station_uncertainty_pct=float(pct),
        uncertainty_source=source,
        sigma_abs_floor=float(sigma_abs_floor),
        sigma_base=float(sigma_base),
    )


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
        single_station_factor=single_station_factor,
    )
