"""ROI-based station snow assimilation for point HS and SWE observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.io.paths import default_results_dir, infer_project_dir, list_member_dirs
from openamundsen_da.util.station_da import (
    StationAssimilationConfig,
    load_station_assimilation_config,
    read_station_metadata,
    resolve_station_sigma_base,
    resolve_station_sigma_abs_floor,
    resolve_station_uncertainty_pct,
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


def _nearest_value(csv_path: Path, value_col: str, target_dt: datetime) -> tuple[pd.Timestamp, float]:
    """Return the unambiguous nearest timestamp/value pair for one CSV."""
    df = _read_timeseries_with_fallback(csv_path, value_col)
    series = df[value_col].dropna()
    if series.empty:
        raise ValueError(f"No non-empty '{value_col}' data found in {csv_path}")

    target = pd.Timestamp(target_dt)
    deltas = (series.index - target).to_series(index=series.index).abs()
    min_delta = deltas.min()
    nearest = deltas[deltas == min_delta]
    if nearest.empty:
        raise ValueError(f"Could not determine nearest timestamp in {csv_path}")
    if len(nearest) > 1:
        raise ValueError(
            f"Ambiguous nearest timestamp in {csv_path} for target {target}: {list(nearest.index.astype(str))}"
        )
    matched_time = pd.Timestamp(nearest.index[0])
    return matched_time, float(series.loc[matched_time])


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


def _resolve_station_uncertainty_pct(
    station_id: str,
    metadata_df: pd.DataFrame,
    config: StationAssimilationConfig,
) -> tuple[float, str]:
    return resolve_station_uncertainty_pct(station_id, metadata_df, config)


def _resolve_station_sigma_abs_floor(
    station_id: str,
    metadata_df: pd.DataFrame,
    metadata_path: Path,
    variable: str,
) -> float:
    return resolve_station_sigma_abs_floor(station_id, metadata_df, metadata_path, variable)


def _build_active_stations(
    *,
    obs_dir: Path,
    members: list[Path],
    date: datetime,
    config: StationAssimilationConfig,
    variable: str,
) -> list[ActiveStation]:
    """Resolve active stations with observation values and effective sigmas."""
    spec = station_variable_spec(variable)
    metadata_df = _read_station_metadata(config.metadata_path)

    active: list[ActiveStation] = []
    for station_id in _candidate_station_ids(obs_dir, members):
        obs_csv = obs_dir / f"{station_id}.csv"
        try:
            obs_time, obs_value = _nearest_value(obs_csv, spec.obs_column, date)
        except ValueError as exc:
            logger.warning("Skipping station {} for {} on {}: {}", station_id, variable, date.date(), exc)
            continue
        if not np.isfinite(obs_value):
            logger.warning("Skipping station {} for {} on {}: observation is not finite", station_id, variable, date.date())
            continue
        if obs_value < 0.0:
            logger.warning(
                "Skipping station {} for {} on {}: observation is negative ({:.6f})",
                station_id,
                variable,
                date.date(),
                obs_value,
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
                obs_value=float(obs_value),
                station_uncertainty_pct=float(sigma_terms.station_uncertainty_pct),
                uncertainty_source=sigma_terms.uncertainty_source,
                sigma_abs_floor=float(sigma_terms.sigma_abs_floor),
                sigma_base=float(sigma_terms.sigma_base),
                sigma=float(sigma_terms.sigma_base),
                single_station_inflated=False,
            )
        )

    if not active:
        raise ValueError(f"No active station observations found for {variable} on {date.date()} in {obs_dir}")

    if len(active) == 1:
        st = active[0]
        inflated_sigma = float(st.sigma_base) * float(config.single_station_factor)
        logger.warning(
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
            obs_value=st.obs_value,
            station_uncertainty_pct=st.station_uncertainty_pct,
            uncertainty_source=st.uncertainty_source,
            sigma_abs_floor=st.sigma_abs_floor,
            sigma_base=st.sigma_base,
            sigma=inflated_sigma,
            single_station_inflated=True,
        )

    return active


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
    spec = station_variable_spec(variable)

    members = list_member_dirs(step_dir / "ensembles", ensemble)
    if not members:
        raise RuntimeError(f"No members found under {step_dir}/ensembles/{ensemble}")

    active_stations = _build_active_stations(
        obs_dir=config.obs_dir,
        members=members,
        date=date,
        config=config,
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
            model_time, model_value = _nearest_value(model_csv, spec.model_column, date)
            if not np.isfinite(model_value):
                raise ValueError(
                    f"Model value for station {station.station_id} is not finite in {model_csv}"
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
                    "matched_model_time": pd.Timestamp(model_time).strftime("%Y-%m-%d %H:%M:%S"),
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
    return StationAssimilationResult(weights=weights_df, diagnostics=diagnostics_df)


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
