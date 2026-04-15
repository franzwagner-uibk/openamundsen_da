"""Benchmark case extraction for supported observation families."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from loguru import logger

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml, list_steps_sorted, read_step_config
from openamundsen_da.methods.viz.fraction_series import (
    default_fraction_obs_path,
    load_named_member_series,
    load_open_loop_fraction_series,
)
from openamundsen_da.observer.fraction_obs import resolve_obs_product_tag
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.da_observables import weights_csv_name
from openamundsen_da.util.station_da import (
    is_station_variable,
    load_station_assimilation_config,
    read_station_metadata,
    resolve_station_sigma_base,
    station_observation_csvs,
    station_variable_spec,
)
from openamundsen_da.util.ts import parse_datetime_opt, read_timeseries_csv


SUPPORTED_BENCHMARK_VARIABLES = ("scf", "wet_snow", "station_hs", "station_swe")


@dataclass(frozen=True)
class BenchmarkVariableSpec:
    variable: str
    kind: str
    obs_value_col: str
    model_value_col: str
    summary_filename: str | None = None
    member_filename: str | None = None


@dataclass(frozen=True)
class RawBenchmarkCase:
    score_set: str
    variable: str
    stream: str
    timestamp: pd.Timestamp
    obs_id: str
    step_name: str | None
    obs_value: float
    open_loop_value: float
    da_informed_values: tuple[float, ...] | None
    prior_values: tuple[float, ...] | None
    posterior_values: tuple[float, ...] | None
    posterior_weights: tuple[float, ...] | None
    sigma_base: float | None = None


@dataclass(frozen=True)
class AnalysisEventContext:
    variable: str
    event_date: date
    step_dir: Path
    step_name: str
    assimilation_dt: datetime
    product: str


@dataclass(frozen=True)
class StepWindow:
    step_name: str
    start: pd.Timestamp
    end: pd.Timestamp


_SPECS = {
    "scf": BenchmarkVariableSpec(
        variable="scf",
        kind="fraction",
        obs_value_col="scf",
        model_value_col="scf",
        summary_filename="scf_summary.csv",
        member_filename="point_scf_roi.csv",
    ),
    "wet_snow": BenchmarkVariableSpec(
        variable="wet_snow",
        kind="fraction",
        obs_value_col="wet_snow_fraction",
        model_value_col="wet_snow_fraction",
        summary_filename="wet_snow_summary.csv",
        member_filename="point_wet_snow_roi.csv",
    ),
    "station_hs": BenchmarkVariableSpec(
        variable="station_hs",
        kind="station",
        obs_value_col="snow_depth",
        model_value_col="snow_depth",
    ),
    "station_swe": BenchmarkVariableSpec(
        variable="station_swe",
        kind="station",
        obs_value_col="swe",
        model_value_col="swe",
    ),
}


def benchmark_variable_spec(variable: str) -> BenchmarkVariableSpec:
    key = str(variable).strip().lower()
    if key == "wet_snow_fraction":
        key = "wet_snow"
    if key not in _SPECS:
        raise ValueError(f"Unsupported benchmark variable: {variable!r}")
    return _SPECS[key]


def benchmark_supported_variables() -> tuple[str, ...]:
    return SUPPORTED_BENCHMARK_VARIABLES


def project_window(project_dir: Path) -> tuple[date, date]:
    cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    start_dt = parse_datetime_opt(str(cfg.get("start_date")))
    end_dt = parse_datetime_opt(str(cfg.get("end_date")))
    if start_dt is None or end_dt is None:
        raise ValueError(f"Project YAML missing valid start_date/end_date under {project_dir}")
    return start_dt.date(), end_dt.date()


def step_windows(project_dir: Path) -> list[StepWindow]:
    windows: list[StepWindow] = []
    for step_dir in list_steps_sorted(project_dir):
        cfg = read_step_config(step_dir) or {}
        start_dt = parse_datetime_opt(str(cfg.get("start_date")))
        end_dt = parse_datetime_opt(str(cfg.get("end_date")))
        if start_dt is None or end_dt is None:
            raise ValueError(f"Step {step_dir.name} is missing valid start_date/end_date")
        windows.append(
            StepWindow(
                step_name=step_dir.name,
                start=pd.Timestamp(start_dt),
                end=pd.Timestamp(end_dt),
            )
        )
    return windows


def event_dates_by_variable(project_dir: Path) -> dict[str, set[date]]:
    out: dict[str, set[date]] = {}
    for ev in load_assimilation_events(project_dir):
        out.setdefault(ev.variable, set()).add(ev.date)
    return out


def _first_event_date_by_variable(project_dir: Path) -> dict[str, date]:
    first_dates: dict[str, date] = {}
    for variable, dates in event_dates_by_variable(project_dir).items():
        if dates:
            first_dates[variable] = min(dates)
    return first_dates


def analysis_event_contexts(project_dir: Path, *, variables: Iterable[str] | None = None) -> list[AnalysisEventContext]:
    selected = None
    if variables is not None:
        selected = {benchmark_variable_spec(v).variable for v in variables}

    steps = list_steps_sorted(project_dir)
    events = load_assimilation_events(project_dir)
    contexts: list[AnalysisEventContext] = []
    for idx, ev in enumerate(events[: min(len(events), len(steps))]):
        if selected is not None and ev.variable not in selected:
            continue
        step_dir = steps[idx]
        cfg = read_step_config(step_dir) or {}
        start_dt = parse_datetime_opt(str(cfg.get("start_date")))
        if start_dt is None:
            raise ValueError(f"Step {step_dir.name} is missing a valid start_date")
        contexts.append(
            AnalysisEventContext(
                variable=ev.variable,
                event_date=ev.date,
                step_dir=step_dir,
                step_name=step_dir.name,
                assimilation_dt=datetime.combine(ev.date, pd.Timestamp(start_dt).time()),
                product=ev.product,
            )
        )
    return contexts


def _has_active_station_link(variable: str, *, obs_date: date, first_event_dates: dict[str, date]) -> bool:
    if not is_station_variable(variable):
        return False
    return any(
        is_station_variable(other_variable) and other_variable != variable and first_date <= obs_date
        for other_variable, first_date in first_event_dates.items()
    )


def _obs_stream(
    variable: str,
    timestamp: pd.Timestamp,
    event_dates: dict[str, set[date]],
    *,
    first_event_dates: dict[str, date],
) -> str:
    obs_date = timestamp.date()
    variable_dates = event_dates.get(variable, set())
    if obs_date in variable_dates:
        return "assimilation_fit"
    first_variable_date = first_event_dates.get(variable)
    if first_variable_date is not None and first_variable_date < obs_date:
        return "semi_independent"
    if _has_active_station_link(variable, obs_date=obs_date, first_event_dates=first_event_dates):
        return "semi_independent"
    return "independent"


def _match_step_name(timestamp: pd.Timestamp, windows: list[StepWindow]) -> str | None:
    for window in windows:
        if window.start <= timestamp <= window.end:
            return window.step_name
        if window.start.date() <= timestamp.date() <= window.end.date():
            return window.step_name
    return None


def _series_from_open_loop_df(df: pd.DataFrame | None, value_col: str) -> pd.Series | None:
    if df is None or df.empty or value_col not in df.columns:
        return None
    series = df.set_index("date")[value_col].dropna().sort_index()
    return None if series.empty else series


def _series_exact_value(series: pd.Series | None, timestamp: pd.Timestamp) -> float | None:
    if series is None:
        return None
    try:
        value = series.loc[timestamp]
    except KeyError:
        return None
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    try:
        f = float(value)
    except Exception:
        return None
    if not pd.notna(f):
        return None
    return f


def _nearest_series_value(series: pd.Series | None, timestamp: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    if series is None or series.empty:
        return None
    deltas = (series.index - timestamp).to_series(index=series.index).abs()
    if deltas.empty:
        return None
    min_delta = deltas.min()
    nearest = deltas[deltas == min_delta]
    if nearest.empty or len(nearest) > 1:
        return None
    matched_time = pd.Timestamp(nearest.index[0])
    value = float(series.loc[matched_time])
    return matched_time, value


def _filter_series_to_window(series: pd.Series, *, start_date: date, end_date: date) -> pd.Series:
    mask = (series.index.date >= start_date) & (series.index.date <= end_date)
    return series.loc[mask]


def _fraction_summary_rows(
    *,
    setup_dir: Path,
    project_dir: Path,
    variable: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    spec = benchmark_variable_spec(variable)
    assert spec.summary_filename is not None
    if variable == "scf":
        resolve_obs_product_tag("scf", setup_dir=setup_dir, project_dir=project_dir)
    elif variable == "wet_snow":
        resolve_obs_product_tag("wet_snow", setup_dir=setup_dir, project_dir=project_dir)
    summary_path = default_fraction_obs_path(setup_dir, project_dir.name, spec.summary_filename)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Benchmark summary CSV not found for {variable}: {summary_path}")
    df = pd.read_csv(summary_path, parse_dates=["date"])
    if spec.obs_value_col not in df.columns:
        raise ValueError(f"{summary_path} missing required column {spec.obs_value_col!r}")
    df = df.dropna(subset=[spec.obs_value_col]).copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    return df.loc[mask].sort_values("date").reset_index(drop=True)


def _station_observation_series(
    *,
    setup_dir: Path,
    project_dir: Path,
    variable: str,
    start_date: date,
    end_date: date,
) -> dict[str, pd.Series]:
    spec = station_variable_spec(variable)
    station_cfg = load_station_assimilation_config(setup_dir, project_dir)
    out: dict[str, pd.Series] = {}
    for csv_path in station_observation_csvs(station_cfg.obs_dir):
        station_id = csv_path.stem.strip().lower()
        try:
            df = read_timeseries_csv(csv_path, "time", [spec.obs_column])
        except Exception:
            df = read_timeseries_csv(csv_path, "date", [spec.obs_column])
        series = df[spec.obs_column].dropna()
        if series.empty:
            continue
        series = _filter_series_to_window(series, start_date=start_date, end_date=end_date)
        if not series.empty:
            out[station_id] = series
    return out


def _open_loop_series_for_variable(project_dir: Path, variable: str, *, station_id: str | None = None) -> pd.Series | None:
    spec = benchmark_variable_spec(variable)
    if spec.kind == "fraction":
        assert spec.member_filename is not None
        df = load_open_loop_fraction_series(project_dir, spec.member_filename, spec.model_value_col)
        return _series_from_open_loop_df(df, spec.model_value_col)
    if station_id is None:
        raise ValueError(f"station_id is required for station benchmark variable {variable!r}")
    df = load_open_loop_fraction_series(project_dir, f"point_{station_id}.csv", spec.model_value_col)
    return _series_from_open_loop_df(df, spec.model_value_col)


def _named_member_series_for_variable(
    project_dir: Path,
    variable: str,
    *,
    station_id: str | None = None,
) -> dict[str, pd.Series]:
    spec = benchmark_variable_spec(variable)
    if spec.kind == "fraction":
        assert spec.member_filename is not None
        return load_named_member_series(project_dir, spec.member_filename, spec.model_value_col)
    if station_id is None:
        raise ValueError(f"station_id is required for station benchmark variable {variable!r}")
    return load_named_member_series(project_dir, f"point_{station_id}.csv", spec.model_value_col)


def _member_values_exact(named_series: dict[str, pd.Series], timestamp: pd.Timestamp) -> dict[str, float]:
    values: dict[str, float] = {}
    for member_id, series in named_series.items():
        value = _series_exact_value(series, timestamp)
        if value is not None:
            values[member_id] = value
    return values


def _member_values_nearest(named_series: dict[str, pd.Series], timestamp: pd.Timestamp) -> tuple[pd.Timestamp, dict[str, float]] | None:
    matched_time: pd.Timestamp | None = None
    values: dict[str, float] = {}
    for member_id, series in named_series.items():
        matched = _nearest_series_value(series, timestamp)
        if matched is None:
            return None
        current_time, value = matched
        if matched_time is None:
            matched_time = current_time
        elif current_time != matched_time:
            return None
        values[member_id] = value
    if matched_time is None:
        return None
    return matched_time, values


def _station_sigma_context(
    *,
    setup_dir: Path,
    project_dir: Path,
) -> tuple:
    config = load_station_assimilation_config(setup_dir, project_dir)
    metadata_df = read_station_metadata(config.metadata_path)
    return config, metadata_df


def _station_case_sigma_base(
    *,
    setup_dir: Path,
    project_dir: Path,
    variable: str,
    station_id: str,
    obs_value: float,
    sigma_context: tuple | None,
) -> float:
    config, metadata_df = sigma_context or _station_sigma_context(
        setup_dir=setup_dir,
        project_dir=project_dir,
    )
    sigma_terms = resolve_station_sigma_base(
        station_id=station_id,
        obs_value=float(obs_value),
        variable=variable,
        config=config,
        metadata_df=metadata_df,
    )
    return float(sigma_terms.sigma_base)


def _weights_for_event(step_dir: Path, variable: str, assimilation_dt: datetime) -> pd.DataFrame:
    weights_path = step_dir / "assim" / weights_csv_name(variable, assimilation_dt)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Benchmark weights CSV not found: {weights_path}")
    df = pd.read_csv(weights_path)
    if "member_id" not in df.columns or "weight" not in df.columns:
        raise ValueError(f"{weights_path} must contain member_id and weight columns")
    out = df[["member_id", "weight"]].copy()
    out["member_id"] = out["member_id"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["member_id", "weight"]).sort_values("member_id").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{weights_path} contains no usable member weights")
    return out


def _aligned_posterior(
    member_values: dict[str, float],
    weights_df: pd.DataFrame,
    *,
    variable: str,
    timestamp: pd.Timestamp,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values: list[float] = []
    weights: list[float] = []
    missing: list[str] = []
    for row in weights_df.itertuples(index=False):
        member_id = str(row.member_id)
        if member_id not in member_values:
            missing.append(member_id)
            continue
        values.append(float(member_values[member_id]))
        weights.append(float(row.weight))
    if missing:
        raise ValueError(
            f"Missing member values for weighted posterior {variable} at {timestamp.date()}: {', '.join(sorted(missing))}"
        )
    if not values:
        raise ValueError(f"No aligned member values found for weighted posterior {variable} at {timestamp.date()}")
    return tuple(values), tuple(weights)


def extract_continuous_cases(
    *,
    project_dir: Path,
    setup_dir: Path,
    variables: Iterable[str],
) -> list[RawBenchmarkCase]:
    start_date, end_date = project_window(project_dir)
    windows = step_windows(project_dir)
    events_by_var = event_dates_by_variable(project_dir)
    first_event_dates = _first_event_date_by_variable(project_dir)
    out: list[RawBenchmarkCase] = []
    station_sigma_context = None

    for variable in sorted({benchmark_variable_spec(v).variable for v in variables}):
        spec = benchmark_variable_spec(variable)
        if spec.kind == "fraction":
            obs_df = _fraction_summary_rows(
                setup_dir=setup_dir,
                project_dir=project_dir,
                variable=variable,
                start_date=start_date,
                end_date=end_date,
            )
            open_loop_series = _open_loop_series_for_variable(project_dir, variable)
            named_series = _named_member_series_for_variable(project_dir, variable)
            if open_loop_series is None or not named_series:
                logger.warning("Skipping continuous benchmark for {}: missing model-side series", variable)
                continue
            for row in obs_df.itertuples(index=False):
                timestamp = pd.Timestamp(row.date)
                obs_value = float(getattr(row, spec.obs_value_col))
                open_loop_value = _series_exact_value(open_loop_series, timestamp)
                member_values = _member_values_exact(named_series, timestamp)
                if open_loop_value is None or not member_values:
                    logger.warning("Skipping {} benchmark case at {}: missing model values", variable, timestamp.date())
                    continue
                out.append(
                    RawBenchmarkCase(
                        score_set="continuous",
                        variable=variable,
                        stream=_obs_stream(
                            variable,
                            timestamp,
                            events_by_var,
                            first_event_dates=first_event_dates,
                        ),
                        timestamp=timestamp,
                        obs_id="roi",
                        step_name=_match_step_name(timestamp, windows),
                        obs_value=obs_value,
                        open_loop_value=open_loop_value,
                        da_informed_values=tuple(member_values[mid] for mid in sorted(member_values)),
                        prior_values=None,
                        posterior_values=None,
                        posterior_weights=None,
                    )
                )
            continue

        if station_sigma_context is None:
            station_sigma_context = _station_sigma_context(setup_dir=setup_dir, project_dir=project_dir)
        obs_series_by_station = _station_observation_series(
            setup_dir=setup_dir,
            project_dir=project_dir,
            variable=variable,
            start_date=start_date,
            end_date=end_date,
        )
        if not obs_series_by_station:
            logger.warning("Skipping continuous benchmark for {}: no station observations found", variable)
            continue
        for station_id, obs_series in sorted(obs_series_by_station.items()):
            open_loop_series = _open_loop_series_for_variable(project_dir, variable, station_id=station_id)
            named_series = _named_member_series_for_variable(project_dir, variable, station_id=station_id)
            if open_loop_series is None or not named_series:
                logger.warning("Skipping {} benchmark for station {}: missing model-side series", variable, station_id)
                continue
            for timestamp, obs_value in obs_series.items():
                timestamp = pd.Timestamp(timestamp)
                open_loop_value = _series_exact_value(open_loop_series, timestamp)
                member_values = _member_values_exact(named_series, timestamp)
                if open_loop_value is None or not member_values:
                    continue
                out.append(
                    RawBenchmarkCase(
                        score_set="continuous",
                        variable=variable,
                        stream=_obs_stream(
                            variable,
                            timestamp,
                            events_by_var,
                            first_event_dates=first_event_dates,
                        ),
                        timestamp=timestamp,
                        obs_id=station_id,
                        step_name=_match_step_name(timestamp, windows),
                        obs_value=float(obs_value),
                        open_loop_value=open_loop_value,
                        da_informed_values=tuple(member_values[mid] for mid in sorted(member_values)),
                        prior_values=None,
                        posterior_values=None,
                        posterior_weights=None,
                        sigma_base=_station_case_sigma_base(
                            setup_dir=setup_dir,
                            project_dir=project_dir,
                            variable=variable,
                            station_id=station_id,
                            obs_value=float(obs_value),
                            sigma_context=station_sigma_context,
                        ),
                    )
                )
    return out


def extract_analysis_cases(
    *,
    project_dir: Path,
    setup_dir: Path,
    variables: Iterable[str],
) -> list[RawBenchmarkCase]:
    selected = {benchmark_variable_spec(v).variable for v in variables}
    start_date, end_date = project_window(project_dir)
    windows = step_windows(project_dir)
    out: list[RawBenchmarkCase] = []

    fraction_obs_cache: dict[str, pd.DataFrame] = {}
    station_obs_cache: dict[str, dict[str, pd.Series]] = {}
    open_loop_cache: dict[tuple[str, str | None], pd.Series | None] = {}
    members_cache: dict[tuple[str, str | None], dict[str, pd.Series]] = {}
    station_sigma_context = None

    events_by_var = event_dates_by_variable(project_dir)
    first_event_dates = _first_event_date_by_variable(project_dir)

    for ctx in analysis_event_contexts(project_dir):
        if not (start_date <= ctx.event_date <= end_date):
            continue
        weights_df = _weights_for_event(ctx.step_dir, ctx.variable, ctx.assimilation_dt)

        for benchmark_variable in sorted(selected):
            spec = benchmark_variable_spec(benchmark_variable)
            stream: str | None = None

            if spec.kind == "fraction":
                if benchmark_variable not in fraction_obs_cache:
                    fraction_obs_cache[benchmark_variable] = _fraction_summary_rows(
                        setup_dir=setup_dir,
                        project_dir=project_dir,
                        variable=benchmark_variable,
                        start_date=start_date,
                        end_date=end_date,
                    )
                obs_df = fraction_obs_cache[benchmark_variable]
                match = obs_df.loc[obs_df["date"].dt.date == ctx.event_date]
                if match.empty:
                    logger.warning(
                        "Skipping analysis benchmark for {} on {}: missing observation row",
                        benchmark_variable,
                        ctx.event_date,
                    )
                    continue
                obs_row = match.iloc[-1]
                obs_value = float(obs_row[spec.obs_value_col])
                timestamp = pd.Timestamp(obs_row["date"])
                key = (benchmark_variable, None)
                if key not in open_loop_cache:
                    open_loop_cache[key] = _open_loop_series_for_variable(project_dir, benchmark_variable)
                if key not in members_cache:
                    members_cache[key] = _named_member_series_for_variable(project_dir, benchmark_variable)
                open_loop_value = _series_exact_value(open_loop_cache[key], timestamp)
                member_values = _member_values_exact(members_cache[key], timestamp)
                if open_loop_value is None or not member_values:
                    raise ValueError(f"Missing model values for analysis benchmark {benchmark_variable} on {ctx.event_date}")
                prior_values = tuple(member_values[mid] for mid in sorted(member_values))
                posterior_values, posterior_weights = _aligned_posterior(
                    member_values,
                    weights_df,
                    variable=benchmark_variable,
                    timestamp=timestamp,
                )
                stream = _obs_stream(
                    benchmark_variable,
                    timestamp,
                    events_by_var,
                    first_event_dates=first_event_dates,
                )
                out.append(
                    RawBenchmarkCase(
                        score_set="analysis",
                        variable=benchmark_variable,
                        stream=stream,
                        timestamp=timestamp,
                        obs_id="roi",
                        step_name=_match_step_name(timestamp, windows) or ctx.step_name,
                        obs_value=obs_value,
                        open_loop_value=open_loop_value,
                        da_informed_values=None,
                        prior_values=prior_values,
                        posterior_values=posterior_values,
                        posterior_weights=posterior_weights,
                    )
                )
                continue

            if benchmark_variable not in station_obs_cache:
                station_obs_cache[benchmark_variable] = _station_observation_series(
                    setup_dir=setup_dir,
                    project_dir=project_dir,
                    variable=benchmark_variable,
                    start_date=start_date,
                    end_date=end_date,
                )
            for station_id, obs_series in sorted(station_obs_cache[benchmark_variable].items()):
                key = (benchmark_variable, station_id)
                if key not in open_loop_cache:
                    open_loop_cache[key] = _open_loop_series_for_variable(
                        project_dir,
                        benchmark_variable,
                        station_id=station_id,
                    )
                if key not in members_cache:
                    members_cache[key] = _named_member_series_for_variable(
                        project_dir,
                        benchmark_variable,
                        station_id=station_id,
                    )
                open_loop_series = open_loop_cache[key]
                named_series = members_cache[key]
                if open_loop_series is None or not named_series:
                    continue
                obs_match = _nearest_series_value(obs_series, pd.Timestamp(ctx.assimilation_dt))
                open_loop_match = _nearest_series_value(open_loop_series, pd.Timestamp(ctx.assimilation_dt))
                members_match = _member_values_nearest(named_series, pd.Timestamp(ctx.assimilation_dt))
                if obs_match is None or open_loop_match is None or members_match is None:
                    continue
                obs_time, obs_value = obs_match
                _, open_loop_value = open_loop_match
                member_time, member_values = members_match
                if member_time != obs_time:
                    continue
                if station_sigma_context is None:
                    station_sigma_context = _station_sigma_context(setup_dir=setup_dir, project_dir=project_dir)
                prior_values = tuple(member_values[mid] for mid in sorted(member_values))
                posterior_values, posterior_weights = _aligned_posterior(
                    member_values,
                    weights_df,
                    variable=benchmark_variable,
                    timestamp=obs_time,
                )
                stream = _obs_stream(
                    benchmark_variable,
                    obs_time,
                    events_by_var,
                    first_event_dates=first_event_dates,
                )
                out.append(
                    RawBenchmarkCase(
                        score_set="analysis",
                        variable=benchmark_variable,
                        stream=stream,
                        timestamp=obs_time,
                        obs_id=station_id,
                        step_name=ctx.step_name,
                        obs_value=float(obs_value),
                        open_loop_value=float(open_loop_value),
                        da_informed_values=None,
                        prior_values=prior_values,
                        posterior_values=posterior_values,
                        posterior_weights=posterior_weights,
                        sigma_base=_station_case_sigma_base(
                            setup_dir=setup_dir,
                            project_dir=project_dir,
                            variable=benchmark_variable,
                            station_id=station_id,
                            obs_value=float(obs_value),
                            sigma_context=station_sigma_context,
                        ),
                    )
                )
    return out


__all__ = [
    "AnalysisEventContext",
    "BenchmarkVariableSpec",
    "RawBenchmarkCase",
    "StepWindow",
    "analysis_event_contexts",
    "benchmark_supported_variables",
    "benchmark_variable_spec",
    "event_dates_by_variable",
    "extract_analysis_cases",
    "extract_continuous_cases",
    "project_window",
    "step_windows",
]
