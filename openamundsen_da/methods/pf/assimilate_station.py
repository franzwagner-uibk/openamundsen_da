"""ROI-based station snow assimilation for point HS and SWE observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.io.paths import default_results_dir, infer_project_dir, list_member_dirs
from openamundsen_da.util.observation_time import (
    ModelClockConfig,
    SeriesTimeMatch,
    SeriesTimeUnavailableError,
    load_model_clock_config,
    match_series_value_to_model_time,
)
from openamundsen_da.util.station_da import (
    StationAssimilationConfig,
    load_station_assimilation_config,
    read_station_metadata,
    resolve_station_sigma_base,
    station_ids_disabled_for_role,
    station_observation_csvs,
    station_variable_spec,
)
from openamundsen_da.util.stats import effective_sample_size, gaussian_logpdf, normalize_log_weights
from openamundsen_da.util.ts import read_timeseries_csv


@dataclass(frozen=True)
class ActiveStation:
    """One active station used in the likelihood for a given DA date."""

    station_id: str
    obs_csv: Path
    obs_time: pd.Timestamp
    obs_time_offset_seconds: float
    obs_value: float
    station_uncertainty_pct: float
    uncertainty_source: str
    sigma_abs_floor: float
    sigma_base: float
    sigma: float
    single_station_inflated: bool


@dataclass(frozen=True)
class StationAssimilationResult:
    """Weights and detailed station diagnostics for one DA date."""

    weights: pd.DataFrame
    diagnostics: pd.DataFrame
    support_audit: pd.DataFrame


def _read_station_metadata(metadata_path: Path) -> pd.DataFrame:
    return read_station_metadata(metadata_path)


def _read_timeseries_with_fallback(csv_path: Path, value_col: str) -> pd.DataFrame:
    """Read one station or model time series, accepting either 'time' or 'date'."""
    last_error: Exception | None = None
    for time_col in ("time", "date"):
        try:
            return read_timeseries_csv(csv_path, time_col, [value_col])
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"Could not read '{value_col}' with time/date column from {csv_path}: {last_error}")


def _matched_value(
    csv_path: Path,
    value_col: str,
    target_dt: datetime,
    *,
    model_clock: ModelClockConfig,
    require_exact: bool,
    require_nonnegative: bool,
) -> SeriesTimeMatch:
    """Return one time-bounded timestamp/value pair for one CSV."""
    df = _read_timeseries_with_fallback(csv_path, value_col)
    series = pd.to_numeric(df[value_col], errors="coerce")
    series = series[np.isfinite(series)]
    if require_nonnegative:
        series = series[series >= 0.0]
    if series.empty:
        raise SeriesTimeUnavailableError(f"No non-empty '{value_col}' data found in {csv_path}")
    try:
        matched = match_series_value_to_model_time(
            series,
            model_time=target_dt,
            timestep=model_clock.timestep,
            timezone_config=model_clock.timezone,
            require_exact=require_exact,
        )
    except SeriesTimeUnavailableError as exc:
        raise SeriesTimeUnavailableError(f"{csv_path}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{csv_path}: {exc}") from exc
    return matched


def _candidate_station_ids(obs_dir: Path, members: list[Path]) -> list[str]:
    """Return station IDs present both in obs/stations and member point outputs."""
    obs_ids = {
        csv_path.stem.strip().lower()
        for csv_path in station_observation_csvs(obs_dir)
    }
    if not obs_ids:
        raise FileNotFoundError(f"No station observation CSVs found in {obs_dir}")

    point_ids: set[str] = set()
    for member_dir in members:
        results_dir = default_results_dir(member_dir)
        if not results_dir.is_dir():
            continue
        point_ids = {
            p.stem[len("point_") :].strip().lower()
            for p in sorted(results_dir.glob("point_*.csv"))
            if p.is_file()
        }
        if point_ids:
            break
    if not point_ids:
        raise FileNotFoundError(
            f"No point_*.csv model outputs found under member results in {members[0].parent if members else obs_dir}"
        )

    station_ids = sorted(obs_ids & point_ids)
    if not station_ids:
        raise FileNotFoundError(
            f"No overlapping station IDs between observation CSVs in {obs_dir} and model point outputs"
        )
    return station_ids


def _build_active_stations(
    *,
    obs_dir: Path,
    members: list[Path],
    date: datetime,
    config: StationAssimilationConfig,
    model_clock: ModelClockConfig,
    variable: str,
) -> tuple[list[ActiveStation], pd.DataFrame]:
    """Resolve active stations with observation values and effective sigmas."""
    spec = station_variable_spec(variable)
    metadata_df = _read_station_metadata(config.metadata_path)
    disabled_station_ids = station_ids_disabled_for_role(metadata_df, "da")

    active: list[ActiveStation] = []
    audit_rows: list[dict[str, object]] = []
    for station_id in _candidate_station_ids(obs_dir, members):
        if station_id in disabled_station_ids:
            logger.debug("Skipping station {} for {}: use_for_da=false", station_id, variable)
            audit_rows.append(
                {
                    "station_id": station_id,
                    "variable": variable,
                    "target_time": pd.Timestamp(date),
                    "status": "disabled",
                    "reason": "use_for_da=false",
                }
            )
            continue
        obs_csv = obs_dir / f"{station_id}.csv"
        try:
            obs_match = _matched_value(
                obs_csv,
                spec.obs_column,
                date,
                model_clock=model_clock,
                require_exact=False,
                require_nonnegative=True,
            )
        except SeriesTimeUnavailableError as exc:
            audit_rows.append(
                {
                    "station_id": station_id,
                    "variable": variable,
                    "target_time": pd.Timestamp(date),
                    "status": "unavailable",
                    "reason": str(exc),
                }
            )
            continue
        if obs_match.interpolated:
            logger.info(
                "Interpolated station {} {} at model time {} from {} ({:.6f}) and {} ({:.6f}); mean={:.6f}",
                station_id,
                variable,
                pd.Timestamp(date),
                obs_match.source_times[0],
                obs_match.source_values[0],
                obs_match.source_times[1],
                obs_match.source_values[1],
                obs_match.value,
            )
        obs_time = obs_match.matched_time
        obs_value = obs_match.value
        obs_time_offset_seconds = obs_match.offset_seconds
        if not np.isfinite(obs_value):
            audit_rows.append(
                {
                    "station_id": station_id,
                    "variable": variable,
                    "target_time": pd.Timestamp(date),
                    "status": "unavailable",
                    "reason": "observation is not finite",
                }
            )
            continue
        if obs_value < 0.0:
            audit_rows.append(
                {
                    "station_id": station_id,
                    "variable": variable,
                    "target_time": pd.Timestamp(date),
                    "status": "unavailable",
                    "reason": f"observation is negative ({obs_value:.6f})",
                }
            )
            continue
        sigma_terms = resolve_station_sigma_base(
            station_id=station_id,
            obs_value=float(obs_value),
            variable=variable,
            config=config,
            metadata_df=metadata_df,
        )
        active.append(
            ActiveStation(
                station_id=station_id,
                obs_csv=obs_csv,
                obs_time=obs_time,
                obs_time_offset_seconds=obs_time_offset_seconds,
                obs_value=float(obs_value),
                station_uncertainty_pct=float(sigma_terms.station_uncertainty_pct),
                uncertainty_source=sigma_terms.uncertainty_source,
                sigma_abs_floor=float(sigma_terms.sigma_abs_floor),
                sigma_base=float(sigma_terms.sigma_base),
                sigma=float(sigma_terms.sigma_base),
                single_station_inflated=False,
            )
        )
        audit_rows.append(
            {
                "station_id": station_id,
                "variable": variable,
                "target_time": pd.Timestamp(date),
                "status": "active",
                "reason": "interpolated" if obs_match.interpolated else "matched",
                "matched_obs_time": obs_time,
                "obs_time_offset_seconds": obs_time_offset_seconds,
                "obs_value": float(obs_value),
            }
        )

    if not active:
        raise ValueError(f"No active station observations found for {variable} on {date.date()} in {obs_dir}")

    if len(active) == 1:
        st = active[0]
        inflated_sigma = float(st.sigma_base) * float(config.single_station_factor)
        logger.info(
            "Only one active station ({}) found for {} on {}. Inflating sigma by factor {:.3f} -> {:.6f}.",
            st.station_id,
            variable,
            date.date(),
            config.single_station_factor,
            inflated_sigma,
        )
        active[0] = ActiveStation(
            station_id=st.station_id,
            obs_csv=st.obs_csv,
            obs_time=st.obs_time,
            obs_time_offset_seconds=st.obs_time_offset_seconds,
            obs_value=st.obs_value,
            station_uncertainty_pct=st.station_uncertainty_pct,
            uncertainty_source=st.uncertainty_source,
            sigma_abs_floor=st.sigma_abs_floor,
            sigma_base=st.sigma_base,
            sigma=inflated_sigma,
            single_station_inflated=True,
        )

    audit = pd.DataFrame(audit_rows).sort_values("station_id").reset_index(drop=True)
    unavailable = int((audit["status"] == "unavailable").sum())
    disabled = int((audit["status"] == "disabled").sum())
    logger.info(
        "{} station support | date={} active={} unavailable={} disabled={}",
        variable,
        pd.Timestamp(date).strftime("%Y-%m-%d"),
        len(active),
        unavailable,
        disabled,
    )
    return active, audit


def assimilate_station_for_date(
    *,
    setup_dir: Path,
    step_dir: Path,
    ensemble: str,
    date: datetime,
    variable: str,
) -> StationAssimilationResult:
    """Assimilate one station variable for one ROI/date using point outputs."""
    project_dir = infer_project_dir(step_dir)
    config = load_station_assimilation_config(setup_dir, project_dir)
    model_clock = load_model_clock_config(setup_dir)
    spec = station_variable_spec(variable)

    members = list_member_dirs(step_dir / "ensembles", ensemble)
    if not members:
        raise RuntimeError(f"No members found under {step_dir}/ensembles/{ensemble}")

    active_stations, support_audit = _build_active_stations(
        obs_dir=config.obs_dir,
        members=members,
        date=date,
        config=config,
        model_clock=model_clock,
        variable=variable,
    )

    member_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    for member_dir in members:
        results_dir = default_results_dir(member_dir)
        member_residuals: list[float] = []
        member_sigmas: list[float] = []
        normalized_sq_terms: list[float] = []
        total_log_likelihood = 0.0

        for station in active_stations:
            model_csv = results_dir / f"point_{station.station_id}.csv"
            if not model_csv.is_file():
                raise FileNotFoundError(
                    f"Missing model point output for station {station.station_id} in {results_dir}"
                )
            model_match = _matched_value(
                model_csv,
                spec.model_column,
                date,
                model_clock=model_clock,
                require_exact=True,
                require_nonnegative=False,
            )
            model_time = model_match.matched_time
            model_value = model_match.value
            model_time_offset_seconds = model_match.offset_seconds
            if not np.isfinite(model_value):
                raise ValueError(
                    f"Model value for station {station.station_id} is not finite in {model_csv}"
                )
            if model_value < 0.0:
                raise ValueError(
                    f"Model value for station {station.station_id} is negative in {model_csv}: "
                    f"{model_value:.6f}"
                )

            residual = float(station.obs_value) - float(model_value)
            sigma = float(station.sigma)
            log_likelihood = float(gaussian_logpdf(np.asarray([residual]), np.asarray([sigma]))[0])
            total_log_likelihood += log_likelihood
            member_residuals.append(residual)
            member_sigmas.append(sigma)
            normalized_sq_terms.append((residual / sigma) ** 2)
            diagnostics_rows.append(
                {
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d %H:%M:%S"),
                    "variable": variable,
                    "station_id": station.station_id,
                    "member_id": member_dir.name,
                    "obs_value": station.obs_value,
                    "model_value": float(model_value),
                    "residual": residual,
                    "sigma_abs_floor": station.sigma_abs_floor,
                    "sigma_base": station.sigma_base,
                    "sigma": sigma,
                    "station_uncertainty_pct": station.station_uncertainty_pct,
                    "uncertainty_source": station.uncertainty_source,
                    "single_station_inflated": bool(station.single_station_inflated),
                    "matched_obs_time": station.obs_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "obs_time_offset_seconds": station.obs_time_offset_seconds,
                    "matched_model_time": pd.Timestamp(model_time).strftime("%Y-%m-%d %H:%M:%S"),
                    "model_time_offset_seconds": model_time_offset_seconds,
                    "obs_csv": str(station.obs_csv),
                    "model_csv": str(model_csv),
                    "log_likelihood": log_likelihood,
                }
            )

        member_rows.append(
            {
                "member_id": member_dir.name,
                "value_obs": float(np.mean([s.obs_value for s in active_stations])),
                "value_model": float(np.mean([s.obs_value - r for s, r in zip(active_stations, member_residuals)])),
                "residual": float(np.mean(member_residuals)),
                "residual_abs_mean": float(np.mean(np.abs(member_residuals))),
                "residual_rms": float(np.sqrt(np.mean(np.square(member_residuals)))),
                "normalized_residual_rms": float(np.sqrt(np.mean(normalized_sq_terms))),
                "sigma": float(np.mean(member_sigmas)),
                "n_stations": len(active_stations),
                "log_weight": total_log_likelihood,
            }
        )

    weights_df = pd.DataFrame(member_rows).sort_values("member_id").reset_index(drop=True)
    weights_df["weight"] = normalize_log_weights(weights_df["log_weight"].to_numpy(dtype=float))
    ess = effective_sample_size(weights_df["weight"].to_numpy(dtype=float))

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df = diagnostics_df.merge(
        weights_df[["member_id", "weight", "log_weight"]],
        on="member_id",
        how="left",
        suffixes=("", "_member"),
    )
    diagnostics_df = diagnostics_df.rename(columns={"weight": "final_weight", "log_weight": "final_log_weight"})
    diagnostics_df = diagnostics_df.sort_values(["station_id", "member_id"]).reset_index(drop=True)

    logger.info(
        "{} assimilation | date={} members={} active_stations={} ESS={:.1f}",
        spec.label,
        pd.Timestamp(date).strftime("%Y-%m-%d"),
        len(weights_df),
        len(active_stations),
        ess,
    )
    return StationAssimilationResult(
        weights=weights_df,
        diagnostics=diagnostics_df,
        support_audit=support_audit,
    )


def assimilate_station_hs_for_date(
    *,
    setup_dir: Path,
    step_dir: Path,
    ensemble: str,
    date: datetime,
) -> StationAssimilationResult:
    """Assimilate station snow depth (HS) observations for one DA date."""
    return assimilate_station_for_date(
        setup_dir=setup_dir,
        step_dir=step_dir,
        ensemble=ensemble,
        date=date,
        variable="station_hs",
    )


def assimilate_station_swe_for_date(
    *,
    setup_dir: Path,
    step_dir: Path,
    ensemble: str,
    date: datetime,
) -> StationAssimilationResult:
    """Assimilate station SWE observations for one DA date."""
    return assimilate_station_for_date(
        setup_dir=setup_dir,
        step_dir=step_dir,
        ensemble=ensemble,
        date=date,
        variable="station_swe",
    )
