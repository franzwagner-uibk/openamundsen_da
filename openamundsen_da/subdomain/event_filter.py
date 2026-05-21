"""Per-sub-domain assimilation-event availability filtering."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.util.station_da import (
    STATION_DA_METADATA_FILENAME,
    read_station_metadata,
    station_ids_disabled_for_role,
    station_observation_csvs,
    station_variable_spec,
)
from openamundsen_da.util.ts import parse_datetime_opt, read_timeseries_csv


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str = ""
    metric: str = ""
    value: float | None = None
    threshold: float | None = None
    active_station_ids: tuple[str, ...] = ()


def _yaml() -> Any:
    import ruamel.yaml as _yaml_mod

    y = _yaml_mod.YAML()
    y.default_flow_style = False
    return y


def _read_project_yaml(path: Path) -> dict:
    y = _yaml()
    with Path(path).open("r", encoding="utf-8-sig") as f:
        data = y.load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Project YAML root must be a mapping: {path}")
    return data


def _write_project_yaml(path: Path, data: dict) -> None:
    y = _yaml()
    buf = io.StringIO()
    y.dump(data, buf)
    Path(path).write_text(buf.getvalue(), encoding="utf-8")


def _as_bool(raw: object, *, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _event_date(raw: object, *, index: int, project_yaml: Path) -> datetime:
    dt = parse_datetime_opt(str(raw))
    if dt is None:
        raise ValueError(
            f"Invalid or missing date at data_assimilation.assimilation_events[{index}].date in {project_yaml}"
        )
    return dt


def _event_variable(raw: object) -> str:
    return str(raw or "").strip().lower()


def _availability_summary_path(setup_dir: Path, project_name: str, variable: str) -> Path | None:
    if variable == "scf":
        return setup_dir / "obs" / project_name / "scf_summary.csv"
    if variable in {"wet_snow", "wet_snow_fraction", "wet_snow_line"}:
        return setup_dir / "obs" / project_name / "wet_snow_summary.csv"
    return None


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read observation summary {path}: {exc}") from exc
    if "date" not in df.columns:
        raise ValueError(f"Observation summary missing required column 'date': {path}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df.dropna(subset=["date"])


def _float_cfg(cfg: dict, key: str, default: float | None = None) -> float | None:
    if key not in cfg:
        return default
    raw = cfg.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"Invalid subdomain_event_filter.{key}: {raw!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"subdomain_event_filter.{key} must be finite")
    return value


def _summary_availability(
    *,
    event_date: datetime,
    variable: str,
    cfg: dict,
    summaries: dict[Path, pd.DataFrame],
    summary_path: Path,
) -> Availability:
    if summary_path not in summaries:
        summaries[summary_path] = _load_summary(summary_path)
    df = summaries[summary_path]
    if df.empty:
        return Availability(False, reason=f"missing_summary:{summary_path.name}")
    rows = df.loc[df["date"] == event_date.date()]
    if rows.empty:
        return Availability(False, reason="missing_observation_date")

    row = rows.iloc[-1]
    max_invalid = _float_cfg(cfg, "max_invalid_fraction")
    min_valid = _float_cfg(cfg, "min_valid_fraction")
    max_cloud = _float_cfg(cfg, "max_cloud_fraction")
    if max_invalid is not None:
        if "invalid_fraction" not in row.index:
            return Availability(False, reason="missing_invalid_fraction", threshold=max_invalid)
        value = float(row["invalid_fraction"])
        if not np.isfinite(value) or value > max_invalid:
            return Availability(
                False,
                reason="invalid_fraction_above_threshold",
                metric="invalid_fraction",
                value=value,
                threshold=max_invalid,
            )
    if max_cloud is not None:
        if "cloud_fraction" not in row.index:
            return Availability(False, reason="missing_cloud_fraction", threshold=max_cloud)
        value = float(row["cloud_fraction"])
        if not np.isfinite(value) or value > max_cloud:
            return Availability(
                False,
                reason="cloud_fraction_above_threshold",
                metric="cloud_fraction",
                value=value,
                threshold=max_cloud,
            )
    if min_valid is not None:
        value = _valid_fraction(row)
        if value is None:
            return Availability(False, reason="missing_valid_fraction", threshold=min_valid)
        if value < min_valid:
            return Availability(
                False,
                reason="valid_fraction_below_threshold",
                metric="valid_fraction",
                value=value,
                threshold=min_valid,
            )
    return Availability(True)


def _valid_fraction(row: pd.Series) -> float | None:
    if "invalid_fraction" in row.index:
        value = float(row["invalid_fraction"])
        if np.isfinite(value):
            return float(1.0 - value)
    if {"n_valid", "n_invalid"}.issubset(set(row.index)):
        valid = float(row["n_valid"])
        invalid = float(row["n_invalid"])
        total = valid + invalid
        if np.isfinite(total) and total > 0:
            return float(valid / total)
    return None


def _read_station_series(csv_path: Path, value_col: str) -> pd.Series:
    last_error: Exception | None = None
    for time_col in ("time", "date"):
        try:
            df = read_timeseries_csv(csv_path, time_col, [value_col])
            series = pd.to_numeric(df[value_col], errors="coerce").dropna()
            return series[np.isfinite(series) & (series >= 0.0)]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"Could not read station series {csv_path}: {last_error}")


def _nearest_station_value(series: pd.Series, event_dt: datetime, max_delta: timedelta | None) -> bool:
    if series.empty:
        return False
    target = pd.Timestamp(event_dt)
    deltas = (series.index - target).to_series(index=series.index).abs()
    if deltas.empty:
        return False
    min_delta = deltas.min()
    if max_delta is not None and min_delta > max_delta:
        return False
    return len(deltas[deltas == min_delta]) == 1


def _station_availability(
    *,
    event_date: datetime,
    variable: str,
    cfg: dict,
    obs_dir: Path,
) -> Availability:
    try:
        spec = station_variable_spec(variable)
    except ValueError:
        return Availability(True)
    metadata_df = read_station_metadata(obs_dir / STATION_DA_METADATA_FILENAME)
    disabled_ids = station_ids_disabled_for_role(metadata_df, "da")
    min_active = int(_float_cfg(cfg, "min_active_stations", 1.0) or 1)
    max_delta_hours = _float_cfg(cfg, "max_time_delta_hours", 36.0)
    max_delta = timedelta(hours=float(max_delta_hours)) if max_delta_hours is not None else None

    active: list[str] = []
    for csv_path in station_observation_csvs(obs_dir):
        station_id = csv_path.stem.strip().lower()
        if station_id in disabled_ids:
            continue
        try:
            series = _read_station_series(csv_path, spec.obs_column)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping station {} while checking event availability: {}", station_id, exc)
            continue
        if _nearest_station_value(series, event_date, max_delta):
            active.append(station_id)

    if len(active) < min_active:
        return Availability(
            False,
            reason="active_station_count_below_minimum",
            metric="active_station_count",
            value=float(len(active)),
            threshold=float(min_active),
            active_station_ids=tuple(active),
        )
    return Availability(True, active_station_ids=tuple(active))


def _station_benchmark_supported(obs_dir: Path) -> bool:
    """Return whether local station observations contain at least one benchmark-enabled station."""
    metadata_df = read_station_metadata(obs_dir / STATION_DA_METADATA_FILENAME)
    if metadata_df.empty or "station_id" not in metadata_df.columns:
        return False
    if "use_for_benchmark" in metadata_df.columns:
        enabled = ~metadata_df["use_for_benchmark"].astype(str).str.strip().str.lower().isin(
            {"false", "0", "no", "n", "off"}
        )
        metadata_df = metadata_df.loc[enabled].copy()
    available_ids = {path.stem.strip().lower() for path in station_observation_csvs(obs_dir)}
    benchmark_ids = {str(sid).strip().lower() for sid in metadata_df["station_id"].dropna()}
    return bool(available_ids & benchmark_ids)


def _drop_station_benchmark_variables_if_unsupported(da_cfg: dict, obs_dir: Path) -> bool:
    """Prune station benchmark variables for stationless sub-domains."""
    benchmark_cfg = da_cfg.get("benchmark")
    if not isinstance(benchmark_cfg, dict):
        return False
    if _station_benchmark_supported(obs_dir):
        return False

    station_vars = {"station_hs", "station_swe"}
    changed = False
    for key in ("variables", "independent_variables", "performance_scores_exclude_variables"):
        raw = benchmark_cfg.get(key)
        if not isinstance(raw, list):
            continue
        filtered = [item for item in raw if str(item).strip().lower() not in station_vars]
        if filtered != raw:
            benchmark_cfg[key] = filtered
            changed = True
    if changed:
        logger.info("Pruned station benchmark variables for stationless sub-domain")
    return changed


def _var_cfg(filter_cfg: dict, variable: str, subdomain_id: str) -> dict:
    variables_cfg = filter_cfg.get("variables") if isinstance(filter_cfg, dict) else {}
    cfg = variables_cfg.get(variable) if isinstance(variables_cfg, dict) else None
    if cfg is None and variable == "wet_snow_line":
        cfg = variables_cfg.get("wet_snow") if isinstance(variables_cfg, dict) else None
    out = dict(cfg or {}) if isinstance(cfg, dict) else {}

    subdomains_cfg = filter_cfg.get("subdomains") if isinstance(filter_cfg, dict) else {}
    subdomain_cfg = subdomains_cfg.get(subdomain_id) if isinstance(subdomains_cfg, dict) else None
    if isinstance(subdomain_cfg, dict):
        sub_vars = subdomain_cfg.get("variables")
        if isinstance(sub_vars, dict):
            override = sub_vars.get(variable)
            if override is None and variable == "wet_snow_line":
                override = sub_vars.get("wet_snow")
            if isinstance(override, dict):
                out.update(override)
    return out


def _is_supported_variable(variable: str) -> bool:
    return variable in {"scf", "wet_snow", "wet_snow_fraction", "wet_snow_line", "station_hs", "station_swe"}


def filter_project_events_for_subdomain(
    *,
    project_yaml: Path,
    setup_dir: Path,
    project_name: str,
    subdomain_id: str,
    dropped_events_csv: Path,
) -> list[dict[str, object]]:
    """Apply optional sub-domain event filtering to one copied project YAML.

    The filter is configured under
    ``data_assimilation.subdomain_event_filter``. Supported event families are
    checked for local availability in every sub-domain; unavailable events are
    dropped only when the filter is enabled.
    """
    project_yaml = Path(project_yaml)
    cfg = _read_project_yaml(project_yaml)
    da_cfg = cfg.get("data_assimilation") or {}
    if not isinstance(da_cfg, dict):
        raise ValueError(f"Project data_assimilation section must be a mapping: {project_yaml}")
    events = da_cfg.get("assimilation_events") or []
    if not isinstance(events, list):
        raise ValueError(f"Project assimilation_events must be a list: {project_yaml}")

    filter_cfg = da_cfg.get("subdomain_event_filter") or {}
    if filter_cfg and not isinstance(filter_cfg, dict):
        raise ValueError("data_assimilation.subdomain_event_filter must be a mapping")
    enabled = _as_bool(filter_cfg.get("enabled") if isinstance(filter_cfg, dict) else None, default=False)
    drop_unavailable = _as_bool(
        filter_cfg.get("drop_unavailable") if isinstance(filter_cfg, dict) else None,
        default=enabled,
    )

    summaries: dict[Path, pd.DataFrame] = {}
    kept: list[object] = []
    dropped: list[dict[str, object]] = []
    unavailable: list[str] = []
    station_obs_dir = setup_dir / "obs" / "stations"

    for idx, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise ValueError(
                f"Expected mapping at data_assimilation.assimilation_events[{idx}], got {type(event).__name__}"
            )
        variable = _event_variable(event.get("variable"))
        if not _is_supported_variable(variable):
            kept.append(event)
            continue
        event_dt = _event_date(event.get("date"), index=idx, project_yaml=project_yaml)
        variable_cfg = _var_cfg(filter_cfg, variable, subdomain_id)
        summary_path = _availability_summary_path(setup_dir, project_name, variable)
        if summary_path is not None:
            availability = _summary_availability(
                event_date=event_dt,
                variable=variable,
                cfg=variable_cfg,
                summaries=summaries,
                summary_path=summary_path,
            )
        else:
            availability = _station_availability(
                event_date=event_dt,
                variable=variable,
                cfg=variable_cfg,
                obs_dir=station_obs_dir,
            )

        if availability.available:
            kept.append(event)
            continue

        row = {
            "subdomain_id": subdomain_id,
            "date": event_dt.date().isoformat(),
            "variable": variable,
            "product": str(event.get("product") or ""),
            "reason": availability.reason,
            "metric": availability.metric,
            "value": availability.value,
            "threshold": availability.threshold,
            "active_station_ids": ";".join(availability.active_station_ids),
            "project_yaml": str(project_yaml),
        }
        if enabled and drop_unavailable:
            dropped.append(row)
            logger.info(
                "Dropped unavailable event for sub-domain {}: {} {} ({})",
                subdomain_id,
                row["date"],
                variable,
                availability.reason,
            )
            continue
        unavailable.append(f"{row['date']} {variable}: {availability.reason}")
        kept.append(event)

    _write_dropped_events(dropped_events_csv, dropped)
    if unavailable:
        raise ValueError(
            f"Configured assimilation events are unavailable in sub-domain {subdomain_id}: "
            + "; ".join(unavailable)
        )
    benchmark_changed = _drop_station_benchmark_variables_if_unsupported(da_cfg, station_obs_dir)
    if enabled and drop_unavailable:
        da_cfg["assimilation_events"] = kept
        if not kept:
            raise ValueError(f"All assimilation events were dropped for sub-domain {subdomain_id}")
    if (enabled and drop_unavailable) or benchmark_changed:
        _write_project_yaml(project_yaml, cfg)
    return dropped


def _write_dropped_events(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "subdomain_id",
        "date",
        "variable",
        "product",
        "reason",
        "metric",
        "value",
        "threshold",
        "active_station_ids",
        "project_yaml",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
