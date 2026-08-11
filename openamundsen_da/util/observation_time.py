"""Strict satellite acquisition-time and model-time matching helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import pandas as pd

from openamundsen_da.io.paths import find_setup_yaml
from openamundsen_da.util.yaml_utils import read_yaml_mapping


@dataclass(frozen=True)
class ObservationTimeMatch:
    """One UTC observation time matched to one model-clock timestamp."""

    observation_time: datetime
    model_time: datetime
    offset_seconds: float


@dataclass(frozen=True)
class ModelClockConfig:
    """Model-clock fields required for bounded observation matching."""

    timestep: pd.Timedelta
    timezone: object


@dataclass(frozen=True)
class SeriesTimeMatch:
    """One value matched to a model-clock timestamp within its timestep."""

    matched_time: pd.Timestamp
    value: float
    offset_seconds: float


class SeriesTimeMatchError(ValueError):
    """Base error for station/model series time matching."""


class SeriesTimeUnavailableError(SeriesTimeMatchError):
    """Raised when no value lies inside the permitted model-time window."""


class SeriesTimeAmbiguityError(SeriesTimeMatchError):
    """Raised when multiple values are equally near the model time."""


@dataclass(frozen=True)
class AcquisitionTime:
    """Resolved source acquisition time and its persisted provenance."""

    value: datetime
    source: str
    quality: str


def parse_utc_timestamp(raw: object, *, field: str) -> datetime:
    """Parse a full timezone-aware ISO-8601 timestamp and return UTC."""
    text = str(raw).strip() if raw is not None else ""
    if not text:
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        stamp = pd.Timestamp(text)
    except Exception as exc:
        raise ValueError(f"{field} must be a valid ISO-8601 timestamp: {raw!r}") from exc
    if stamp.tzinfo is None:
        raise ValueError(f"{field} must include a timezone offset or Z: {raw!r}")
    return stamp.tz_convert("UTC").to_pydatetime()


def midnight_fallback(day: datetime | pd.Timestamp | object) -> datetime:
    """Return UTC midnight for a date-only observation fallback."""
    stamp = pd.Timestamp(day)
    return datetime.combine(stamp.date(), time.min, tzinfo=timezone.utc)


def model_timezone(raw: object) -> timezone | ZoneInfo:
    """Resolve the openAMUNDSEN timezone configuration."""
    if isinstance(raw, bool):
        raise ValueError("setup.timezone must be a UTC offset in hours or an IANA timezone name")
    if isinstance(raw, (int, float)):
        return timezone(pd.Timedelta(hours=float(raw)).to_pytimedelta())
    text = str(raw).strip()
    if not text:
        raise ValueError("setup.timezone must be configured")
    try:
        return ZoneInfo(text)
    except Exception as exc:
        try:
            return timezone(pd.Timedelta(hours=float(text)).to_pytimedelta())
        except Exception:
            raise ValueError(
                "setup.timezone must be a UTC offset in hours or an IANA timezone name"
            ) from exc


def parse_model_timestep(raw: object) -> pd.Timedelta:
    """Parse one positive, fixed model timestep."""
    text = str(raw).strip() if raw is not None else ""
    if not text:
        raise ValueError("setup.timestep must be configured")
    try:
        offset = pd.tseries.frequencies.to_offset(text)
        value = pd.Timedelta(offset.nanos, unit="ns")
    except Exception as exc:
        raise ValueError(
            f"setup.timestep must be a positive fixed pandas-compatible frequency: {raw!r}"
        ) from exc
    if value <= pd.Timedelta(0):
        raise ValueError("setup.timestep must be positive")
    return value


def load_model_clock_config(setup_dir: Path) -> ModelClockConfig:
    """Load the authoritative timestep and timezone from a setup YAML."""
    setup_yaml = find_setup_yaml(setup_dir)
    config = read_yaml_mapping(
        setup_yaml,
        error_cls=ValueError,
        context="Setup YAML root",
    )
    if "timezone" not in config:
        raise ValueError(f"Setup configuration must define timezone: {setup_yaml}")
    timezone_config = config["timezone"]
    model_timezone(timezone_config)
    return ModelClockConfig(
        timestep=parse_model_timestep(config.get("timestep")),
        timezone=timezone_config,
    )


def _as_utc_index(index: pd.DatetimeIndex, *, timezone_config: object) -> pd.DatetimeIndex:
    tz = model_timezone(timezone_config)
    if index.tz is None:
        localized = index.tz_localize(tz, ambiguous="raise", nonexistent="raise")
    else:
        localized = index.tz_convert(tz)
    return localized.tz_convert("UTC")


def _as_utc_timestamp(value: datetime | pd.Timestamp, *, timezone_config: object) -> pd.Timestamp:
    tz = model_timezone(timezone_config)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(tz, ambiguous="raise", nonexistent="raise")
    else:
        timestamp = timestamp.tz_convert(tz)
    return timestamp.tz_convert("UTC")


def match_series_value_to_model_time(
    series: pd.Series,
    *,
    model_time: datetime | pd.Timestamp,
    timestep: pd.Timedelta | object,
    timezone_config: object,
    require_exact: bool = False,
) -> SeriesTimeMatch:
    """Return the unique nearest value within half a model timestep.

    Naive timestamps are interpreted in the configured model timezone. Ties,
    duplicate nearest timestamps and values outside the allowed window are
    rejected. Model-output callers can set ``require_exact`` to enforce the
    exact model-clock timestamp.
    """
    if series is None or series.empty:
        raise SeriesTimeUnavailableError("Cannot match an empty time series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Time series must use a DatetimeIndex")

    step = timestep if isinstance(timestep, pd.Timedelta) else parse_model_timestep(timestep)
    if step <= pd.Timedelta(0):
        raise ValueError("Model timestep must be positive")
    target_utc = _as_utc_timestamp(model_time, timezone_config=timezone_config)
    candidate_utc = _as_utc_index(series.index, timezone_config=timezone_config)
    valid_positions = [index for index, stamp in enumerate(candidate_utc) if not pd.isna(stamp)]
    if not valid_positions:
        raise SeriesTimeUnavailableError("Time series contains no valid timestamps")

    offsets = [abs(candidate_utc[index] - target_utc) for index in valid_positions]
    minimum = min(offsets)
    nearest_positions = [
        position
        for position, offset in zip(valid_positions, offsets, strict=True)
        if offset == minimum
    ]
    if len(nearest_positions) != 1:
        matched = [str(series.index[position]) for position in nearest_positions]
        raise SeriesTimeAmbiguityError(
            f"Ambiguous nearest timestamp for model time {pd.Timestamp(model_time)}: {matched}"
        )

    allowed = pd.Timedelta(0) if require_exact else step / 2
    if minimum > allowed:
        qualifier = "exact model timestamp" if require_exact else f"half the {step} model timestep"
        raise SeriesTimeUnavailableError(
            f"Nearest timestamp is {minimum.total_seconds():.0f} s from model time "
            f"{pd.Timestamp(model_time)}, exceeding {qualifier}"
        )

    position = nearest_positions[0]
    raw_value = series.iloc[position]
    try:
        value = float(raw_value)
    except Exception as exc:
        raise ValueError(
            f"Matched value at {series.index[position]} is not numeric: {raw_value!r}"
        ) from exc
    return SeriesTimeMatch(
        matched_time=pd.Timestamp(series.index[position]),
        value=value,
        offset_seconds=float(minimum.total_seconds()),
    )


def match_observation_to_model_time(
    *,
    observation_time: datetime,
    model_times: list[datetime] | pd.DatetimeIndex,
    timezone_config: object,
) -> ObservationTimeMatch:
    """Match to the unique nearest model time within half a model timestep."""
    if observation_time.tzinfo is None:
        raise ValueError("observation_time must be timezone-aware")
    if len(model_times) == 0:
        raise ValueError("Cannot match observation time against an empty model timeline")

    tz = model_timezone(timezone_config)
    obs_utc = observation_time.astimezone(timezone.utc)
    candidates = pd.DatetimeIndex(model_times)
    if candidates.tz is None:
        candidates = candidates.tz_localize(tz)
    else:
        candidates = candidates.tz_convert(tz)
    candidate_utc = candidates.tz_convert("UTC")
    obs_stamp = pd.Timestamp(obs_utc)
    offsets = abs(candidate_utc - obs_stamp)
    minimum = offsets.min()
    nearest = [idx for idx, value in enumerate(offsets) if value == minimum]
    if len(nearest) != 1:
        raise ValueError(
            f"Observation time {obs_utc.isoformat()} is tied between multiple model timesteps"
        )

    if len(candidate_utc) == 1:
        if minimum != pd.Timedelta(0):
            raise ValueError("A single model timestep can only be matched exactly")
    else:
        ordered = candidate_utc.sort_values().unique()
        diffs = pd.Series(ordered[1:] - ordered[:-1])
        if (diffs <= pd.Timedelta(0)).any():
            raise ValueError("Model timeline must contain unique increasing timestamps")
        timestep = diffs.min()
        if not (diffs == timestep).all():
            raise ValueError("Model timeline must have a regular timestep for observation matching")
        if minimum > timestep / 2:
            raise ValueError(
                f"Nearest model timestep is {minimum.total_seconds():.0f} s from observation, "
                f"exceeding half the {timestep.total_seconds():.0f} s model timestep"
            )

    index = nearest[0]
    model_stamp = candidates[index]
    return ObservationTimeMatch(
        observation_time=obs_utc,
        model_time=model_stamp.tz_localize(None).to_pydatetime(),
        offset_seconds=float(minimum.total_seconds()),
    )


def read_acquisition_manifest(path: Path) -> pd.DataFrame:
    """Read and strictly validate a tracked offline acquisition-time manifest."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Acquisition-time manifest not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "product",
        "source",
        "product_identity",
        "acquisition_time",
        "time_source",
        "time_quality",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Acquisition-time manifest {path} is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"Acquisition-time manifest is empty: {path}")
    duplicate_key = frame["product"].str.upper() + "|" + frame["source"].map(lambda value: Path(value).name)
    if duplicate_key.duplicated().any():
        duplicates = sorted(duplicate_key.loc[duplicate_key.duplicated(keep=False)].unique())
        raise ValueError(f"Acquisition-time manifest contains duplicate source rows: {duplicates}")
    for index, row in frame.iterrows():
        parse_utc_timestamp(row["acquisition_time"], field=f"{path}:row {index + 2}.acquisition_time")
        if (
            not row["product"].strip()
            or not row["product_identity"].strip()
            or not row["time_source"].strip()
            or not row["time_quality"].strip()
        ):
            raise ValueError(f"Acquisition-time manifest contains empty required values at row {index + 2}")
    return frame


def acquisition_from_manifest(path: Path, *, product: str, source: str) -> dict[str, str]:
    """Resolve exactly one acquisition record by product and source identity."""
    frame = read_acquisition_manifest(path)
    source_name = Path(str(source)).name
    candidates = frame[
        frame["product"].str.upper().eq(str(product).strip().upper())
        & frame["source"].map(lambda item: Path(item).name).eq(source_name)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one acquisition-time manifest row for product={product!r}, "
            f"source={source_name!r}; found {len(candidates)}"
        )
    return {key: str(value) for key, value in candidates.iloc[0].items()}


def _timestamp_from_json(value: object) -> object | None:
    if isinstance(value, dict):
        preferred = ("acquisition_time", "datetime", "start_datetime", "sensing_time")
        lowered = {str(key).lower(): item for key, item in value.items()}
        for key in preferred:
            if key in lowered:
                return lowered[key]
        for item in value.values():
            found = _timestamp_from_json(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _timestamp_from_json(item)
            if found is not None:
                return found
    return None


def _generic_sidecar_time(value: object) -> object | None:
    if not isinstance(value, dict):
        return None
    for key, item in value.items():
        if str(key).lower() == "time":
            if isinstance(item, list) and len(item) == 1:
                return item[0]
            return item
    return None


def _parse_sidecar_timestamp(raw: object, *, source: Path) -> datetime | None:
    text = str(raw).strip() if raw is not None else ""
    if not text:
        return None
    try:
        stamp = pd.Timestamp(text)
    except Exception as exc:
        raise ValueError(f"Invalid timestamp in sidecar metadata for {source}: {raw!r}") from exc
    # A date-only midnight value is not an acquisition timestamp. It can only
    # support the documented midnight fallback after stronger sources fail.
    if stamp.hour == stamp.minute == stamp.second == stamp.microsecond == 0:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").to_pydatetime()


def _sidecar_acquisition_time(source_path: Path) -> datetime | None:
    candidates = (
        source_path.with_suffix(source_path.suffix + ".json"),
        source_path.with_suffix(source_path.suffix + ".aux.json"),
        source_path.with_suffix(".json"),
    )
    for sidecar in candidates:
        if not sidecar.is_file():
            continue
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid JSON sidecar metadata: {sidecar}") from exc
        raw = _timestamp_from_json(payload)
        if raw is None:
            raw = _generic_sidecar_time(payload)
        stamp = _parse_sidecar_timestamp(raw, source=sidecar)
        if stamp is not None:
            return stamp
    return None


def _raster_metadata_acquisition_time(source_path: Path) -> datetime | None:
    if source_path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    if not source_path.is_file():
        return None
    try:
        import rasterio

        with rasterio.open(source_path) as dataset:
            tags = {str(key).lower(): value for key, value in dataset.tags().items()}
    except Exception as exc:
        raise ValueError(f"Could not read raster metadata from {source_path}") from exc
    for key in ("acquisition_time", "datetime", "start_datetime", "sensing_time"):
        if key not in tags:
            continue
        return parse_utc_timestamp(tags[key], field=f"{source_path}:{key}")
    return None


def _filename_acquisition_time(source_path: Path, parser: str | None) -> datetime | None:
    if parser is None:
        return None
    parser_name = str(parser).strip().lower()
    if parser_name not in {"sentinel_1", "sentinel-1", "s1"}:
        raise ValueError(f"Unsupported filename_time_parser: {parser!r}")
    match = re.search(
        r"(?P<year>20\d{2})[_-]?(?P<month>\d{2})[_-]?(?P<day>\d{2})"
        r"(?:T|[_-])(?P<hour>\d{2})[_-]?(?P<minute>\d{2})[_-]?(?P<second>\d{2})",
        source_path.name,
    )
    if match is None:
        return None
    values = {key: int(value) for key, value in match.groupdict().items()}
    return datetime(**values, tzinfo=timezone.utc)


def resolve_acquisition_time(
    *,
    source_path: Path,
    product: str,
    observation_date: object,
    cf_time: object | None = None,
    filename_parser: str | None = None,
    manifest_path: Path | None = None,
) -> AcquisitionTime:
    """Resolve acquisition time using the documented strict precedence."""
    source_path = Path(source_path)
    if cf_time is not None:
        stamp = pd.Timestamp(cf_time)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return AcquisitionTime(stamp.tz_convert("UTC").to_pydatetime(), "cf_time_coordinate", "verified")
    raster_time = _raster_metadata_acquisition_time(source_path)
    if raster_time is not None:
        return AcquisitionTime(raster_time, "raster_metadata", "verified")
    sidecar_time = _sidecar_acquisition_time(source_path)
    if sidecar_time is not None:
        return AcquisitionTime(sidecar_time, "sidecar_metadata", "derived")
    filename_time = _filename_acquisition_time(source_path, filename_parser)
    if filename_time is not None:
        return AcquisitionTime(filename_time, "filename_parser", "derived")
    if manifest_path is not None:
        row = acquisition_from_manifest(manifest_path, product=product, source=source_path.name)
        return AcquisitionTime(
            parse_utc_timestamp(row["acquisition_time"], field=f"{manifest_path}:acquisition_time"),
            row["time_source"],
            row["time_quality"],
        )
    return AcquisitionTime(midnight_fallback(observation_date), "midnight_fallback", "fallback_midnight")


__all__ = [
    "ObservationTimeMatch",
    "AcquisitionTime",
    "acquisition_from_manifest",
    "match_observation_to_model_time",
    "midnight_fallback",
    "model_timezone",
    "parse_utc_timestamp",
    "read_acquisition_manifest",
    "resolve_acquisition_time",
]
