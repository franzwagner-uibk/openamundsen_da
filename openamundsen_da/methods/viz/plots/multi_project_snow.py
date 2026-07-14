"""Multi-project snow time-series plots for completed DA projects."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from loguru import logger

from openamundsen_da.io.paths import (
    find_project_yaml,
    list_member_dirs,
    list_steps_sorted,
    read_step_config,
)
from openamundsen_da.methods.viz.plots.common import (
    apply_fraction_grid,
    apply_month_interval_axis_labels,
    force_figure_text_black,
    format_station_label,
    save_figure_png,
    set_matplotlib_text_black,
)
from openamundsen_da.methods.viz.plots.ensemble_meta import load_stations_table_from_steps
from openamundsen_da.methods.viz.plots.theme import (
    BAND_ALPHA,
    COLOR_DA_OBS,
    COLOR_OPEN_LOOP,
    GRID_ALPHA,
    GRID_LS,
    GRID_LW,
    LS_STATION_OBS,
    LW_DA_OBS,
    LW_MEAN,
    LW_OPEN,
    da_variable_fill_color,
    da_variable_line_color,
)
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.station_da import station_observation_csvs
from openamundsen_da.util.stats import envelope
from openamundsen_da.util.ts import apply_window, concat_series, read_timeseries_csv, resample_and_smooth


DEFAULT_STATIONS = ("latschbloder", "proviantdepot")
DEFAULT_VARIABLES = ("snow_depth", "swe")
DEFAULT_OUTPUT_SUBDIR = Path("results") / "plots" / "multi_year_snow"
CONTEXT_MAP_NAME = "context_map.png"


@dataclass(frozen=True)
class SnowPlotSeries:
    """Data needed for one multi-project snow plot."""

    open_loop: pd.Series
    members: list[pd.Series]
    obs: pd.Series | None
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class GeneratedSnowPlots:
    """Paths written by the multi-project snow plotting command."""

    plot_paths: tuple[Path, ...]
    context_map: Path | None


def _parse_datetime(text: object) -> pd.Timestamp | None:
    if text is None:
        return None
    value = str(text).strip().replace("_", "-")
    if not value:
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).tz_localize(None) if getattr(parsed, "tzinfo", None) is not None else pd.Timestamp(parsed)


def _resolve_project_dirs(
    *,
    setup: Path | None,
    projects: Sequence[str] | None,
    project_dirs: Sequence[Path] | None,
) -> list[Path]:
    names = list(projects or [])
    direct_dirs = [Path(path).resolve() for path in (project_dirs or [])]
    if setup is not None or names:
        if setup is None or not names:
            raise ValueError("Use --setup together with at least one --project")
        if direct_dirs:
            raise ValueError("Use either --setup/--project or --project-dir, not both")
        root = Path(setup).resolve()
        resolved = [(root / "projects" / name).resolve() for name in names]
    else:
        if len(direct_dirs) < 1:
            raise ValueError("Provide --setup/--project or at least one --project-dir")
        resolved = direct_dirs

    for project_dir in resolved:
        if not project_dir.is_dir():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")
        find_project_yaml(project_dir)
        if not (project_dir / "steps").is_dir():
            raise FileNotFoundError(f"Completed project is missing steps/: {project_dir}")
    return resolved


def _infer_setup_root(project_dirs: Sequence[Path]) -> Path | None:
    parents = {Path(project_dir).resolve().parent.parent for project_dir in project_dirs if project_dir.parent.name == "projects"}
    if len(parents) == 1:
        return next(iter(parents))
    return None


def _default_output_dir(setup: Path | None, project_dirs: Sequence[Path]) -> Path:
    if setup is not None:
        return Path(setup).resolve() / DEFAULT_OUTPUT_SUBDIR
    inferred = _infer_setup_root(project_dirs)
    if inferred is not None:
        return inferred / DEFAULT_OUTPUT_SUBDIR
    return Path(project_dirs[0]).resolve() / DEFAULT_OUTPUT_SUBDIR


def _project_bounds(project_dir: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    import ruamel.yaml as _yaml

    yaml = _yaml.YAML(typ="safe")
    with find_project_yaml(project_dir).open("r", encoding="utf-8") as f:
        cfg = yaml.load(f) or {}
    start = _parse_datetime(cfg.get("start_date"))
    end = _parse_datetime(cfg.get("end_date"))
    if start is not None and end is not None:
        return start, end

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for step_dir in list_steps_sorted(project_dir):
        step_cfg = read_step_config(step_dir)
        step_start = _parse_datetime(step_cfg.get("start_date"))
        step_end = _parse_datetime(step_cfg.get("end_date"))
        if step_start is not None:
            starts.append(step_start)
        if step_end is not None:
            ends.append(step_end)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _time_bounds(project_dirs: Sequence[Path]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for project_dir in project_dirs:
        start, end = _project_bounds(project_dir)
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    if not starts or not ends:
        raise ValueError("Could not determine multi-project time bounds from project/step YAML files")
    return min(starts), max(ends)


def _year_span_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.year}_{end.year}"


def _variable_meta(variable: str) -> tuple[str, str, str, str]:
    key = str(variable).strip().lower()
    if key == "snow_depth":
        return "snow_depth", "point_snow_depth_roi.csv", "Snow depth", "Snow depth [m]"
    if key == "swe":
        return "swe", "point_swe_roi.csv", "SWE", "SWE [mm]"
    raise ValueError(f"Unsupported snow variable: {variable!r}")


def _station_point_filename(station_id: str) -> str:
    return f"point_{station_id.strip().lower()}.csv"


def _read_result_series(csv_path: Path, variable: str, *, daily: bool, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    df = read_timeseries_csv(csv_path, "time", [variable])
    if daily:
        df = resample_and_smooth(df, "D", {variable: "mean"}, None)
    df = apply_window(df, start.to_pydatetime(), end.to_pydatetime())
    series = pd.to_numeric(df[variable], errors="coerce").dropna()
    series.name = variable
    return series


def _load_result_collection(
    project_dirs: Sequence[Path],
    *,
    filename: str,
    variable: str,
    daily: bool,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Series, list[pd.Series]]:
    open_loop_segments: list[pd.Series] = []
    member_segments: dict[str, list[pd.Series]] = {}

    for project_dir in project_dirs:
        for step_dir in list_steps_sorted(project_dir):
            open_loop_path = step_dir / "ensembles" / "prior" / "open_loop" / "results" / filename
            if open_loop_path.is_file():
                try:
                    series = _read_result_series(open_loop_path, variable, daily=daily, start=start, end=end)
                    if not series.empty:
                        open_loop_segments.append(series)
                except ValueError as exc:
                    if f"Missing column '{variable}'" not in str(exc):
                        raise

            for member_dir in list_member_dirs(step_dir / "ensembles", "prior"):
                csv_path = member_dir / "results" / filename
                if not csv_path.is_file():
                    continue
                try:
                    series = _read_result_series(csv_path, variable, daily=daily, start=start, end=end)
                except ValueError as exc:
                    if f"Missing column '{variable}'" in str(exc):
                        continue
                    raise
                if not series.empty:
                    member_segments.setdefault(member_dir.name, []).append(series)

    members = [concat_series(segments).sort_index() for _, segments in sorted(member_segments.items()) if segments]
    members = [series for series in members if not series.empty]
    return concat_series(open_loop_segments).sort_index(), members


def _load_station_observation(
    setup_root: Path | None,
    station_id: str,
    variable: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    if setup_root is None:
        return None
    obs_dir = Path(setup_root) / "obs" / "stations"
    if not obs_dir.is_dir():
        return None
    station_files = {path.stem.lower(): path for path in station_observation_csvs(obs_dir)}
    csv_path = station_files.get(station_id.lower())
    if csv_path is None:
        return None
    try:
        df = read_timeseries_csv(csv_path, "time", [variable])
    except ValueError as exc:
        if f"Missing column '{variable}'" in str(exc):
            return None
        raise
    df[variable] = pd.to_numeric(df[variable], errors="coerce").mask(lambda values: values < 0.0)
    df = resample_and_smooth(df, "D", {variable: "mean"}, None)
    df = apply_window(df, start.to_pydatetime(), end.to_pydatetime())
    series = df[variable].dropna()
    if series.empty:
        return None
    series.name = variable
    return series


def _load_snow_plot_series(
    project_dirs: Sequence[Path],
    *,
    setup_root: Path | None,
    variable: str,
    station_id: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> SnowPlotSeries:
    var_col, roi_filename, _, _ = _variable_meta(variable)
    if station_id is None:
        filename = roi_filename
        daily = False
        obs = None
    else:
        filename = _station_point_filename(station_id)
        daily = True
        obs = _load_station_observation(setup_root, station_id, var_col, start=start, end=end)
    open_loop, members = _load_result_collection(
        project_dirs,
        filename=filename,
        variable=var_col,
        daily=daily,
        start=start,
        end=end,
    )
    if open_loop.empty and not members and obs is None:
        label = f"{station_id} {var_col}" if station_id is not None else f"ROI {var_col}"
        raise FileNotFoundError(f"No data available for {label}")
    return SnowPlotSeries(open_loop=open_loop, members=members, obs=obs, start=start, end=end)


def _stations_table(project_dirs: Sequence[Path]) -> pd.DataFrame | None:
    step_dirs: list[Path] = []
    for project_dir in project_dirs:
        try:
            step_dirs.extend(list_steps_sorted(project_dir))
        except Exception:
            continue
    if not step_dirs:
        return None
    try:
        return load_stations_table_from_steps(step_dirs, "prior")
    except Exception:
        return None


def _station_label(station_id: str, stations_df: pd.DataFrame | None) -> str:
    if stations_df is None:
        return station_id
    _base, _alt, label = format_station_label(station_id, stations_df, fallback=station_id)
    return label


def _apply_time_axis(ax, start: pd.Timestamp, end: pd.Timestamp) -> None:
    ax.set_xlim(start.to_pydatetime(), end.to_pydatetime())
    apply_month_interval_axis_labels(ax, (start, end), interval=3, labelsize=8.0)


def _apply_snow_y_axis(ax, series: Sequence[pd.Series]) -> None:
    values: list[float] = []
    for item in series:
        if item is not None and not item.empty:
            values.extend(pd.to_numeric(item, errors="coerce").dropna().tolist())
    finite = pd.Series(values, dtype=float).dropna()
    upper = float(finite.max()) if not finite.empty else 1.0
    upper = max(0.1, upper * 1.08)
    ax.set_ylim(0.0, upper)


def _plot_snow_series(
    *,
    series: SnowPlotSeries,
    variable: str,
    title: str,
    ylabel: str,
    output: Path,
    backend: str,
) -> Path:
    import matplotlib

    matplotlib.use(backend or "Agg")
    set_matplotlib_text_black(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    var_key, _, _, _ = _variable_meta(variable)
    color_key = "station_hs" if var_key == "snow_depth" else "station_swe"
    line_color = da_variable_line_color(color_key)
    fill_color = da_variable_fill_color(color_key)

    fig, ax = plt.subplots(figsize=(9.6, 3.05))
    mean, lo, hi = envelope(series.members, q_low=0.0, q_high=1.0)
    if not mean.empty:
        ax.fill_between(mean.index, lo.values, hi.values, color=fill_color, alpha=BAND_ALPHA, edgecolor="none", zorder=2)
        ax.plot(mean.index, mean.values, color=line_color, lw=LW_MEAN, zorder=4)
    if not series.open_loop.empty:
        ax.plot(series.open_loop.index, series.open_loop.values, color=COLOR_OPEN_LOOP, lw=LW_OPEN, zorder=5)
    if series.obs is not None and not series.obs.empty:
        ax.plot(
            series.obs.index,
            series.obs.values,
            color=COLOR_DA_OBS,
            lw=LW_DA_OBS,
            ls=LS_STATION_OBS,
            zorder=6,
        )

    ax.set_title(title, loc="left", fontsize=10.2, pad=7.0)
    ax.set_ylabel(ylabel, fontsize=9.0)
    apply_fraction_grid(ax, y_step=None)
    ax.grid(True, ls=GRID_LS, lw=GRID_LW, alpha=GRID_ALPHA)
    _apply_time_axis(ax, series.start, series.end)
    _apply_snow_y_axis(ax, [series.open_loop, mean, lo, hi, series.obs if series.obs is not None else pd.Series(dtype=float)])

    handles = []
    if not series.open_loop.empty:
        handles.append(Line2D([0], [0], color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open loop"))
    if not mean.empty:
        handles.extend(
            [
                Patch(facecolor=fill_color, edgecolor=fill_color, alpha=BAND_ALPHA, label="ensemble (with mean)"),
            ]
        )
    if series.obs is not None and not series.obs.empty:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLOR_DA_OBS,
                lw=LW_DA_OBS,
                ls=LS_STATION_OBS,
                label="station observation",
            )
        )
    if handles:
        ax.legend(handles=handles, loc="upper left", ncol=min(4, len(handles)), frameon=True, fontsize=8.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    force_figure_text_black(fig, [ax])
    save_figure_png(fig, output, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info("Wrote {}", output)
    return output


def _expected_outputs(
    output_dir: Path,
    *,
    stations: Sequence[str],
    variables: Sequence[str],
    year_label: str,
    context_source: Path | None,
) -> list[Path]:
    paths: list[Path] = []
    for station_id in stations:
        station_token = station_id.strip().lower()
        for variable in variables:
            paths.append(output_dir / f"station_{station_token}_{variable}_{year_label}.png")
    for variable in variables:
        paths.append(output_dir / f"roi_mean_{variable}_{year_label}.png")
    if context_source is not None:
        paths.append(output_dir / CONTEXT_MAP_NAME)
    return paths


def _context_map_source(project_dirs: Sequence[Path]) -> Path | None:
    for project_dir in project_dirs:
        candidate = project_dir / "results" / "maps" / "setup_overview.png"
        if candidate.is_file():
            return candidate
    return None


def _copy_context_map(source: Path | None, output_dir: Path, *, overwrite: bool) -> Path | None:
    if source is None:
        logger.warning("No setup_overview.png found in project results/maps directories; skipping context map")
        return None
    output = output_dir / CONTEXT_MAP_NAME
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output} (use --overwrite)")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    logger.info("Wrote {}", output)
    return output


def plot_multi_project_snow(
    project_dirs: Sequence[Path],
    *,
    output_dir: Path,
    setup_root: Path | None = None,
    stations: Sequence[str] = DEFAULT_STATIONS,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    overwrite: bool = False,
    backend: str = "Agg",
) -> GeneratedSnowPlots:
    project_dirs = [Path(path).resolve() for path in project_dirs]
    if not project_dirs:
        raise ValueError("At least one project directory is required")
    output_dir = Path(output_dir).resolve()
    setup_root = Path(setup_root).resolve() if setup_root is not None else _infer_setup_root(project_dirs)
    start, end = _time_bounds(project_dirs)
    year_label = _year_span_label(start, end)
    context_source = _context_map_source(project_dirs)

    expected = _expected_outputs(output_dir, stations=stations, variables=variables, year_label=year_label, context_source=context_source)
    existing = [path for path in expected if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing[:10])
        raise FileExistsError(f"Output file(s) already exist; use --overwrite:\n{formatted}")

    stations_df = _stations_table(project_dirs)
    written: list[Path] = []
    for station_id in stations:
        station_token = station_id.strip().lower()
        station_label = _station_label(station_token, stations_df)
        for variable in variables:
            var_col, _, title_var, ylabel = _variable_meta(variable)
            plot_series = _load_snow_plot_series(
                project_dirs,
                setup_root=setup_root,
                variable=var_col,
                station_id=station_token,
                start=start,
                end=end,
            )
            output = output_dir / f"station_{station_token}_{var_col}_{year_label}.png"
            written.append(
                _plot_snow_series(
                    series=plot_series,
                    variable=var_col,
                    title=f"{title_var} at {station_label}, {start.year}-{end.year}",
                    ylabel=ylabel,
                    output=output,
                    backend=backend,
                )
            )

    for variable in variables:
        var_col, _, title_var, ylabel = _variable_meta(variable)
        plot_series = _load_snow_plot_series(
            project_dirs,
            setup_root=setup_root,
            variable=var_col,
            station_id=None,
            start=start,
            end=end,
        )
        output = output_dir / f"roi_mean_{var_col}_{year_label}.png"
        written.append(
            _plot_snow_series(
                series=plot_series,
                variable=var_col,
                title=f"Mean {title_var} in ROI, {start.year}-{end.year}",
                ylabel=ylabel,
                output=output,
                backend=backend,
            )
        )

    context_map = _copy_context_map(context_source, output_dir, overwrite=overwrite)
    return GeneratedSnowPlots(plot_paths=tuple(written), context_map=context_map)


def cli_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oa-da-plot-multi-project-snow",
        description="Create multi-project snow station and ROI plots from completed openAMUNDSEN-DA projects.",
    )
    parser.add_argument("--setup", type=Path, help="Setup root containing projects/")
    parser.add_argument("--project", action="append", default=[], help="Project name under --setup/projects (repeatable)")
    parser.add_argument("--project-dir", action="append", type=Path, default=[], help="Completed project directory (repeatable)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: <setup>/results/plots/multi_year_snow)")
    parser.add_argument("--station", action="append", dest="stations", help="Station id to plot (repeatable)")
    parser.add_argument("--variable", action="append", dest="variables", choices=DEFAULT_VARIABLES, help="Snow variable to plot")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output PNGs")
    parser.add_argument("--backend", default="Agg", help="Matplotlib backend (default: Agg)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_cli_logger(args.log_level or "INFO", enqueue=False)
    try:
        project_dirs = _resolve_project_dirs(setup=args.setup, projects=args.project, project_dirs=args.project_dir)
        output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(args.setup, project_dirs)
        result = plot_multi_project_snow(
            project_dirs,
            output_dir=output_dir,
            setup_root=args.setup,
            stations=tuple(args.stations or DEFAULT_STATIONS),
            variables=tuple(args.variables or DEFAULT_VARIABLES),
            overwrite=bool(args.overwrite),
            backend=str(args.backend or "Agg"),
        )
    except Exception as exc:
        logger.error("Multi-project snow plotting failed: {}", exc)
        return 1

    logger.info("Finished multi-project snow plots: {} plot(s), context map: {}", len(result.plot_paths), result.context_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
