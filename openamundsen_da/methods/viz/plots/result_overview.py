"""Plot the setup-level result overview for fraction, ROI, and station series."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from string import ascii_lowercase

import pandas as pd
from loguru import logger
import numpy as np

from openamundsen_da.benchmark.pipeline.core import load_benchmark_config
from openamundsen_da.methods.viz.plots.benchmark.core import (
    build_event_skill_plot_data,
    compute_event_skill_plot_positions,
    draw_score_metric_panel,
    score_legend_handles,
    score_legend_handler_map,
    score_metric_ylim,
    score_variable_sort_key,
)
from openamundsen_da.io.paths import (
    abspath_relative_to,
    find_project_yaml,
    list_member_dirs,
    list_steps_sorted,
    project_fraction_envelope_path,
    project_result_overview_custom_output_path,
)
from openamundsen_da.methods.viz.station_meta import load_ensemble_station_table_from_steps
from openamundsen_da.methods.viz.plots.theme import (
    BAND_ALPHA,
    COLOR_DA_OBS,
    FIGHEIGHT_OVERVIEW_ROW,
    FIGWIDTH_OVERVIEW_PAPER,
    OVERVIEW_SCORE_PANEL_HEIGHT_FACTOR,
    LW_MEMBER,
    LW_MEAN,
    LW_OPEN,
    SIZE_DA_OBS,
    LW_DA_OBS,
    da_variable_style,
)
from openamundsen_da.methods.viz.plots.common import (
    apply_fraction_grid,
    draw_assim_labels,
    draw_assimilation_markers,
    draw_assimilation_vlines,
    force_figure_text_black,
    format_station_label,
    result_title_pad,
    result_axis_scale,
    save_figure_png,
    set_matplotlib_text_black,
)
from openamundsen_da.methods.viz.fraction_series import (
    default_result_overview_output,
    load_fraction_series,
    load_member_series,
    load_open_loop_fraction_series,
)
from openamundsen_da.methods.viz.maps.panel_renderers import (
    _wsl_prior_summary_from_weights_df,
)
from openamundsen_da.methods.viz.plots.assimilation.ess_timeline import (
    ess_axis_ticks,
    ess_title,
    load_setup_ess_series,
    load_setup_ess_threshold,
)
from openamundsen_da.observer.summary_io import load_scf_summary as _load_scf_obs
from openamundsen_da.observer.summary_paths import resolve_fraction_summary_path
from openamundsen_da.util.da_events import AssimilationEvent, load_assimilation_events
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.station_da import station_observation_csvs
from openamundsen_da.util.stats import envelope
from openamundsen_da.util.ts import concat_series, read_timeseries_csv
from openamundsen_da.util.yaml_utils import read_yaml_mapping


@dataclass(frozen=True)
class PanelSpec:
    panel: str
    title: str | None = None
    show_obs: bool = True
    station_id: str | None = None


@dataclass
class StationPanelData:
    station_id: str
    display_name: str
    altitude_m: float | None
    open_loop: pd.Series | None
    members: list[pd.Series]
    obs: pd.Series | None

    @property
    def has_data(self) -> bool:
        return bool(self.members) or self.open_loop is not None or self.obs is not None


@dataclass
class EssPanelData:
    series: pd.DataFrame | None
    ensemble_size: int | None
    threshold: float | None = None

    @property
    def has_data(self) -> bool:
        return self.series is not None and not self.series.empty


_PANEL_ALIASES = {
    "fsc": "fSC",
    "wsf": "WSF",
    "wsla": "WSLA",
    "roi-swe": "roi-swe",
    "roi-sd": "roi-sd",
    "station-sd": "station-sd",
    "station-swe": "station-swe",
    "ess": "ess",
    "scores-crpss": "scores-crpss",
    "scores-ner": "scores-ner",
    "scores-zskill": "scores-zskill",
}

_DEFAULT_PANELS = [
    PanelSpec(panel="fSC"),
    PanelSpec(panel="WSF"),
    PanelSpec(panel="WSLA"),
    PanelSpec(panel="roi-swe"),
    PanelSpec(panel="roi-sd"),
]

_PANEL_YLABELS = {
    "fSC": "snow cover fraction",
    "WSF": "wet snow fraction (WSF)",
    "WSLA": "wet snow line altitude (WSLA) [m a.s.l.]",
    "roi-swe": "swe [mm]",
    "roi-sd": "snow depth [m]",
    "station-sd": "snow depth [m]",
    "station-swe": "swe [mm]",
    "ess": "ESS",
    "scores-crpss": "CRPSS",
    "scores-ner": "NER",
    "scores-zskill": "zSkill",
}

_DEFAULT_TITLES = {
    "fSC": "snow cover fraction (roi) - openAMUNDSEN ensemble and satellite observations",
    "WSF": "wet snow fraction (roi) - openAMUNDSEN ensemble and satellite observations",
    "WSLA": "wet snow line altitude (roi) - openAMUNDSEN ensemble and satellite observations",
    "roi-swe": "mean swe (roi) - openAMUNDSEN ensemble and open loop",
    "roi-sd": "mean snow depth (roi) - openAMUNDSEN ensemble and open loop",
    "ess": "effective sample size",
    "scores-crpss": "CRPSS",
    "scores-ner": "NER",
    "scores-zskill": "zSkill",
}

_STATION_PANEL_META = {
    "station-sd": {
        "value_col": "snow_depth",
        "variable_key": "SD",
    },
    "station-swe": {
        "value_col": "swe",
        "variable_key": "SWE",
    },
}

_PANEL_VARIABLE_KEYS = {
    "fSC": "scf",
    "WSF": "wet_snow",
    "WSLA": "wet_snow_line",
    "roi-swe": "station_swe",
    "roi-sd": "station_hs",
    "station-swe": "station_swe",
    "station-sd": "station_hs",
    "ess": "scf",
}

_ASSIM_STYLES = {
    "scf": {"ls": "--"},
    "wet_snow": {"ls": "--"},
    "wet_snow_line": {"ls": "--"},
    "station_hs": {"ls": "--"},
    "station_swe": {"ls": "--"},
}

_STATION_PANEL_EVENT_VARIABLE = {
    "station-sd": "station_hs",
    "station-swe": "station_swe",
}

_ASSIM_LABEL_ROW_OFFSETS_PTS = [2.0, 8.0]
_ASSIM_LABEL_MIN_SPACING_DAYS = 18.0


class _LabeledLegendTuple(tuple):
    def __new__(cls, artists, label: str):
        obj = super().__new__(cls, artists)
        obj._label = label
        return obj

    def get_label(self) -> str:
        return str(self._label)


class _EnsembleLegendHandle(_LabeledLegendTuple):
    pass


class _EnsembleLegendHandler:
    def __init__(self, *, patch_height_frac: float = 0.86, line_inset_frac: float = 0.12):
        self._patch_height_frac = patch_height_frac
        self._line_inset_frac = line_inset_frac

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle

        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width = handlebox.width
        height = handlebox.height
        y_bottom = y0 + 0.5 * (1.0 - self._patch_height_frac) * height
        patch_height = self._patch_height_frac * height

        patch_artist = Rectangle(
            (x0, y_bottom),
            width,
            patch_height,
            facecolor=orig_handle[0].get_facecolor(),
            edgecolor=orig_handle[0].get_edgecolor(),
            linewidth=orig_handle[0].get_linewidth(),
            alpha=orig_handle[0].get_alpha(),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch_artist)

        inset = self._line_inset_frac * width
        line_artist = Line2D(
            [x0 + inset, x0 + width - inset],
            [y0 + 0.5 * height, y0 + 0.5 * height],
            color=orig_handle[1].get_color(),
            linewidth=orig_handle[1].get_linewidth(),
            transform=handlebox.get_transform(),
            solid_capstyle="round",
        )
        handlebox.add_artist(line_artist)
        return patch_artist


def _load_scf_obs_series(path: Path) -> pd.DataFrame | None:
    """Load SCF summary data, falling back to a generic fraction-series reader."""
    try:
        return _load_scf_obs(path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.debug("Falling back to generic SCF summary reader for {}: {}", path, exc)
        return load_fraction_series(path, "scf")


def _normalize_panel_name(raw: object) -> str:
    key = str(raw or "").strip().lower()
    if key in {"fws", "wsl"}:
        replacement = "WSF" if key == "fws" else "WSLA"
        raise ValueError(f"Unsupported result_overview panel: {raw!r}. Use {replacement!r} instead.")
    if key not in _PANEL_ALIASES:
        raise ValueError(f"Unsupported result_overview panel: {raw!r}")
    return _PANEL_ALIASES[key]


def _coerce_bool(raw: object, *, default: bool = True) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


def _parse_panel_specs(config_path: Path) -> list[PanelSpec]:
    cfg = read_yaml_mapping(config_path, error_cls=RuntimeError, context="Project plots config")
    raw_panels = cfg.get("panels")
    if not isinstance(raw_panels, list) or not raw_panels:
        raise ValueError(f"Missing non-empty panels list in {config_path}")

    specs: list[PanelSpec] = []
    for idx, raw in enumerate(raw_panels, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping at panels[{idx}] in {config_path}")
        panel = _normalize_panel_name(raw.get("panel"))
        station_id = raw.get("station_id")
        if panel.startswith("station-"):
            if station_id is None or str(station_id).strip() == "":
                raise ValueError(f"Missing required station_id for panels[{idx}] in {config_path}")
            station_id = str(station_id).strip()
        title = raw.get("title", raw.get("subtitle"))
        if title is not None and str(title).strip() == "":
            title = None
        specs.append(
            PanelSpec(
                panel=panel,
                title=str(title).strip() if title is not None else None,
                show_obs=_coerce_bool(raw.get("show_obs"), default=True),
                station_id=str(station_id).strip() if station_id is not None else None,
            )
        )
    return specs


def _project_custom_config_path(project_dir: Path) -> Path | None:
    candidate = (project_dir / "plots.yml").resolve()
    if not candidate.is_file():
        return None
    return candidate


def _default_custom_output(project_dir: Path) -> Path:
    out_path = project_result_overview_custom_output_path(project_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _project_time_bounds(project_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    project_yaml = find_project_yaml(project_dir)
    cfg = read_yaml_mapping(project_yaml, error_cls=RuntimeError, context="Project YAML root")
    start_raw = cfg.get("start_date")
    end_raw = cfg.get("end_date")
    if start_raw is None or end_raw is None:
        return None
    start = pd.to_datetime(start_raw)
    end = pd.to_datetime(end_raw)
    if pd.isna(start) or pd.isna(end):
        return None
    return start, end


def _custom_overview_needs_score_points(specs: list[PanelSpec] | None) -> bool:
    return any(_is_score_panel(spec.panel) for spec in (specs or []))


def _load_score_points_for_custom_overview(project_dir: Path) -> pd.DataFrame:
    score_path = project_dir / "results" / "benchmark" / "scores" / "event_scores.csv"
    if not score_path.is_file():
        raise FileNotFoundError(
            f"Missing benchmark event scores for custom score panel(s): {score_path}. "
            "Run the benchmark stage before requesting scores-crpss, scores-ner, or scores-zskill."
        )

    event_scores = pd.read_csv(score_path)
    if event_scores.empty:
        raise ValueError(f"Benchmark event scores CSV is empty: {score_path}")

    score_points = build_event_skill_plot_data(event_scores, project_dir=project_dir)
    exclude_variables = {
        str(value).strip().lower()
        for value in load_benchmark_config(project_dir).performance_scores_exclude_variables
        if str(value).strip()
    }
    if exclude_variables:
        score_points = score_points[~score_points["variable"].astype(str).str.lower().isin(exclude_variables)].copy()
    if score_points.empty:
        raise ValueError(f"Benchmark event scores contain no usable DA-date score points after exclusions: {score_path}")

    assimilation_dates = sorted({pd.Timestamp(ev.date).normalize() for ev in load_assimilation_events(project_dir)})
    return compute_event_skill_plot_positions(score_points, assimilation_dates=assimilation_dates)


def _load_setup_stations_df(project_dir: Path, setup_dir: Path) -> pd.DataFrame | None:
    meta_path = setup_dir / "meteo" / "stations.csv"
    if meta_path.is_file():
        try:
            return pd.read_csv(meta_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read station metadata {}: {}", meta_path, exc)
    try:
        steps = list_steps_sorted(project_dir)
    except FileNotFoundError:
        return None
    if not steps:
        return None
    return load_ensemble_station_table_from_steps(steps, "prior")


def _validate_station_ids(specs: list[PanelSpec], stations_df: pd.DataFrame | None) -> None:
    if stations_df is None or stations_df.empty:
        return
    known = set()
    for col in stations_df.columns:
        if col.lower().strip() in {"id", "station_id", "station", "code"}:
            known.update(stations_df[col].astype(str).str.strip().str.lower())
    if not known:
        return
    for spec in specs:
        if not spec.station_id:
            continue
        if spec.station_id.strip().lower() not in known:
            raise ValueError(f"Unknown station_id in result overview config: {spec.station_id}")


def _read_series_with_fallback(path: Path, value_col: str) -> pd.Series | None:
    if not path.is_file():
        return None
    try:
        first_row = pd.read_csv(path, nrows=1)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to inspect series {}: {}", path, exc)
        return None
    if first_row.empty:
        return None
    time_col = next((c for c in ("time", "date", "datetime") if c in first_row.columns), first_row.columns[0])
    try:
        df = read_timeseries_csv(path, time_col, [value_col])
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, ValueError) and f"Missing column '{value_col}'" in str(exc):
            return None
        logger.debug("Failed to read series {}: {}", path, exc)
        return None
    if value_col not in df.columns:
        return None
    series = df[value_col].resample("D").mean().dropna().sort_index()
    if series.empty:
        return None
    return series


def _load_station_panel_data(
    project_dir: Path,
    station_id: str,
    *,
    value_col: str,
    stations_df: pd.DataFrame | None,
) -> StationPanelData:
    point_name = f"point_{station_id}.csv"
    member_segments: dict[str, list[pd.Series]] = {}
    open_loop_segments: list[pd.Series] = []

    for step_dir in list_steps_sorted(project_dir):
        open_loop_path = step_dir / "ensembles" / "prior" / "open_loop" / "results" / point_name
        open_loop_series = _read_series_with_fallback(open_loop_path, value_col)
        if open_loop_series is not None:
            open_loop_segments.append(open_loop_series)

        for member_dir in list_member_dirs(step_dir / "ensembles", "prior"):
            series = _read_series_with_fallback(member_dir / "results" / point_name, value_col)
            if series is not None:
                member_segments.setdefault(member_dir.name, []).append(series)

    open_loop = None
    if open_loop_segments:
        stitched_open_loop = concat_series(open_loop_segments).dropna().sort_index()
        if not stitched_open_loop.empty:
            open_loop = stitched_open_loop

    members: list[pd.Series] = []
    for member_name in sorted(member_segments):
        stitched = concat_series(member_segments[member_name]).dropna().sort_index()
        if not stitched.empty:
            members.append(stitched)

    obs = None
    obs_dir_candidates = [
        project_dir / "obs" / "stations",
        project_dir.parent.parent / "obs" / "stations",
    ]
    for obs_dir in obs_dir_candidates:
        if not obs_dir.is_dir():
            continue
        station_files = {path.stem.lower(): path for path in station_observation_csvs(obs_dir)}
        obs_path = station_files.get(station_id.lower())
        if obs_path is not None:
            obs = _read_series_with_fallback(obs_path, value_col)
            if obs is not None and not obs.empty:
                break

    display_name, altitude_m, _ = format_station_label(station_id, stations_df, fallback=station_id)
    return StationPanelData(
        station_id=station_id,
        display_name=display_name,
        altitude_m=altitude_m,
        open_loop=open_loop,
        members=members,
        obs=obs,
    )


def _station_obs_frame(series: pd.Series | None, *, value_col: str) -> pd.DataFrame | None:
    if series is None or series.empty:
        return None
    return pd.DataFrame({"date": pd.to_datetime(series.index), value_col: series.values})


def _station_assimilation_dates(events: list[AssimilationEvent], panel: str) -> list[pd.Timestamp]:
    event_variable = _STATION_PANEL_EVENT_VARIABLE.get(panel)
    if event_variable is None:
        return []
    return [
        pd.Timestamp.combine(ev.date, pd.Timestamp.min.time())
        for ev in events
        if str(ev.variable).strip().lower() == event_variable
    ]


def _station_title(spec: PanelSpec, station_data: StationPanelData) -> str:
    if spec.title:
        return spec.title
    metric = "snow depth" if spec.panel == "station-sd" else "swe"
    alt_text = f" {int(station_data.altitude_m)} m" if station_data.altitude_m is not None else ""
    return (
        f"{metric} {station_data.display_name}{alt_text} - openAMUNDSEN ensemble "
        "and station observation"
    )


def _panel_title(spec: PanelSpec, station_data: StationPanelData | None = None) -> str:
    if spec.title:
        return spec.title
    if spec.panel == "ess":
        return _DEFAULT_TITLES["ess"]
    if spec.panel in _DEFAULT_TITLES:
        return _DEFAULT_TITLES[spec.panel]
    if station_data is None:
        raise ValueError(f"Missing station metadata for panel {spec.panel}")
    return _station_title(spec, station_data)


def _ess_panel_title(spec: PanelSpec, ess_data: EssPanelData | None) -> str:
    if spec.title:
        return spec.title
    ensemble_size = ess_data.ensemble_size if ess_data is not None else None
    return ess_title(ensemble_size=ensemble_size)


def _is_score_panel(panel: str) -> bool:
    return panel in {"scores-crpss", "scores-ner", "scores-zskill"}


def _score_metric_for_panel(panel: str) -> str:
    if panel == "scores-crpss":
        return "crpss"
    if panel == "scores-ner":
        return "ner"
    if panel == "scores-zskill":
        return "zskill"
    raise ValueError(f"Unsupported score panel: {panel}")


def _score_panel_points_for_metric(score_points: pd.DataFrame | None, panel: str) -> pd.DataFrame:
    if score_points is None or score_points.empty:
        return pd.DataFrame()
    metric = _score_metric_for_panel(panel)
    if metric not in score_points.columns:
        return pd.DataFrame()
    filtered = score_points.copy()
    filtered[metric] = pd.to_numeric(filtered[metric], errors="coerce")
    filtered = filtered[filtered[metric].notna()].copy()
    return filtered


def _date_bounds_frames(*frames: pd.DataFrame | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for frame in frames:
        if frame is None or frame.empty or "date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["date"]).dropna()
        if dates.empty:
            continue
        mins.append(dates.min())
        maxs.append(dates.max())
    if not mins:
        return None
    return min(mins), max(maxs)


def _wsl_prior_member_env(member_series: list[pd.Series] | None) -> pd.DataFrame | None:
    if not member_series:
        return None
    aligned = pd.concat(member_series, axis=1, join="outer")
    if aligned.empty:
        return None
    n = aligned.count(axis=1)
    center = aligned.median(axis=1, skipna=True).where(n > 0)
    value_min = aligned.min(axis=1, skipna=True).where(n > 0)
    value_max = aligned.max(axis=1, skipna=True).where(n > 0)
    out = pd.DataFrame(
        {
            "date": aligned.index,
            "value_mean": center.to_numpy(dtype=float),
            "value_min": value_min.to_numpy(dtype=float),
            "value_max": value_max.to_numpy(dtype=float),
            "n": n.to_numpy(dtype=float),
        }
    ).sort_values("date")
    return out if not out.empty else None


def _default_wsl_overview_env(project_dir: Path) -> pd.DataFrame | None:
    return _wsl_prior_member_env(
        load_member_series(
            project_dir,
            "point_wet_snow_line_roi.csv",
            "wet_snow_line",
            preserve_missing_values=True,
        )
    )


def _load_wsl_prior_coverage_frame(project_dir: Path) -> pd.DataFrame | None:
    rows: list[dict[str, object]] = []
    steps_dir = Path(project_dir) / "steps"
    if not steps_dir.is_dir():
        return None
    for step_dir in sorted(steps_dir.glob("step_*")):
        assim_dir = step_dir / "assim"
        if not assim_dir.is_dir():
            continue
        for weights_path in sorted(assim_dir.glob("weights_wet_snow_line_*.csv")):
            stamp = weights_path.stem.rsplit("_", 1)[-1]
            date = pd.to_datetime(stamp, format="%Y%m%d", errors="coerce")
            if pd.isna(date):
                logger.debug("Skipping WSLA weights file with unreadable date: {}", weights_path)
                continue
            try:
                df = pd.read_csv(weights_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read WSLA weights file {}: {}", weights_path, exc)
                continue
            summary = _wsl_prior_summary_from_weights_df(df)
            if summary is None:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(date).normalize(),
                    "value_mean": summary["median"],
                    "value_min": summary["min"],
                    "value_max": summary["max"],
                    "value_obs": summary["obs"],
                    "n": summary["n_members"],
                }
            )
    if not rows:
        return None
    out = pd.DataFrame(rows).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out if not out.empty else None


def _draw_wsl_prior_coverage_markers(ax, coverage: pd.DataFrame | None, *, color: str) -> None:
    if coverage is None or coverage.empty:
        return
    working = coverage.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    for column in ("value_min", "value_mean", "value_max"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    span_mask = working["date"].notna() & working["value_min"].notna() & working["value_max"].notna()
    if bool(span_mask.any()):
        ax.vlines(
            working.loc[span_mask, "date"],
            working.loc[span_mask, "value_min"],
            working.loc[span_mask, "value_max"],
            colors=color,
            linewidth=1.1,
            alpha=0.9,
            zorder=1.7,
        )
    center_mask = working["date"].notna() & working["value_mean"].notna()
    if bool(center_mask.any()):
        ax.scatter(
            working.loc[center_mask, "date"],
            working.loc[center_mask, "value_mean"],
            color=color,
            marker="_",
            s=60.0,
            linewidths=1.2,
            zorder=1.8,
            label="_nolegend_",
        )


def _finite_value_points(frame: pd.DataFrame | None, value_col: str) -> pd.DataFrame | None:
    """Return only rows with finite plotted values while keeping the original frame intact."""
    if frame is None or frame.empty or value_col not in frame.columns:
        return None
    values = pd.to_numeric(frame[value_col], errors="coerce")
    mask = values.notna()
    if not mask.any():
        return None
    out = frame.loc[mask].copy()
    out[value_col] = values.loc[mask]
    return out


def _date_bounds_series(*series_items: pd.Series | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for series in series_items:
        if series is None or series.empty:
            continue
        mins.append(pd.to_datetime(series.index).min())
        maxs.append(pd.to_datetime(series.index).max())
    if not mins:
        return None
    return min(mins), max(maxs)


def _pad_single_day_bounds(bounds: tuple[pd.Timestamp, pd.Timestamp] | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if bounds is None:
        return None
    start, end = bounds
    if pd.Timestamp(start) != pd.Timestamp(end):
        return bounds
    pad = pd.Timedelta(days=3)
    return pd.Timestamp(start) - pad, pd.Timestamp(end) + pad


def _series_to_frame(series: pd.Series | None, value_col: str) -> pd.DataFrame | None:
    if series is None or series.empty:
        return None
    return pd.DataFrame({"date": series.index, value_col: series.values})


def _band_frame(
    member_series: list[pd.Series] | None,
    *,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> pd.DataFrame | None:
    if not member_series:
        return None
    mean, lo, hi = envelope(member_series, q_low=q_low, q_high=q_high)
    if mean.empty:
        return None
    return pd.DataFrame(
        {
            "date": mean.index,
            "value_mean": mean.values,
            "value_min": lo.values,
            "value_max": hi.values,
        }
    )


def _panel_style(panel: str) -> dict[str, str]:
    return da_variable_style(_PANEL_VARIABLE_KEYS[panel])


def _assim_style(variable: str) -> dict[str, str]:
    meta = _ASSIM_STYLES.get(variable)
    if meta is None:
        return {"variable_key": variable, "color": "#777777", "ls": "--"}
    style = da_variable_style(variable)
    return {"variable_key": variable, "color": style["line"], "ls": str(meta["ls"])}


def _assim_labels(events: list[AssimilationEvent]) -> tuple[list[pd.Timestamp], list[str]]:
    dates: list[pd.Timestamp] = []
    labels: list[str] = []
    for idx, event in enumerate(events, start=1):
        dates.append(pd.to_datetime(event.date))
        labels.append(str(idx))
    return dates, labels


def _center_assim_event_times(events: list[AssimilationEvent]) -> list[pd.Timestamp]:
    """Place day-based result-plot DA markers at midday for visual alignment."""
    return [pd.to_datetime(event.date) + pd.Timedelta(hours=12) for event in events]


def _draw_all_assim(ax, events: list[AssimilationEvent], *, center_of_day: bool = False) -> None:
    draw_dates = _center_assim_event_times(events) if center_of_day else [pd.to_datetime(event.date) for event in events]
    for event, draw_date in zip(events, draw_dates):
        meta = _assim_style(event.variable)
        draw_assimilation_vlines(
            ax,
            [draw_date],
            color=str(meta["color"]),
            ls=str(meta["ls"]),
            lw=1.2,
            alpha=0.95,
            label="_nolegend_",
        )


def _add_assim_label_axis(ax, events: list[AssimilationEvent], idx: int, *, center_of_day: bool = False):
    if not events:
        return None
    import matplotlib.dates as mdates

    if center_of_day:
        dates = _center_assim_event_times(events)
        labels = [str(i) for i in range(1, len(events) + 1)]
    else:
        dates, labels = _assim_labels(events)

    date_index = pd.to_datetime(list(dates))
    if date_index.empty:
        return None
    x_min, x_max = ax.get_xlim()
    visible_start = pd.Timestamp(mdates.num2date(x_min)).tz_localize(None)
    visible_end = pd.Timestamp(mdates.num2date(x_max)).tz_localize(None)
    visible_items = [
        (date, label)
        for date, label in zip(date_index, labels)
        if visible_start <= pd.Timestamp(date).tz_localize(None) <= visible_end
    ]
    if not visible_items:
        return None

    label_axis = ax.twiny()
    label_axis.set_label(f"assimilation_label_axis_{idx}")
    label_axis.patch.set_alpha(0.0)
    if hasattr(label_axis, "set_in_layout"):
        label_axis.set_in_layout(False)
    label_axis.set_xlim(ax.get_xlim())
    label_axis.set_xticks([])
    label_axis.set_xlabel("")
    label_axis.yaxis.set_visible(False)
    label_axis.xaxis.set_visible(False)
    for spine in label_axis.spines.values():
        spine.set_visible(False)

    draw_assim_labels(
        label_axis,
        [item[0] for item in visible_items],
        labels=[item[1] for item in visible_items],
        max_labels=max(1, len(visible_items)),
        y_offset_pts=_ASSIM_LABEL_ROW_OFFSETS_PTS[0],
        rotation=0.0,
        row_y_offsets_pts=_ASSIM_LABEL_ROW_OFFSETS_PTS,
        min_row_spacing_days=_ASSIM_LABEL_MIN_SPACING_DAYS,
        axes_y=1.0,
        ha="center",
    )
    if label_axis is not None:
        label_axis.set_zorder(ax.get_zorder() + 1)
    return label_axis


def _build_result_overview_legend_handles(
    *,
    show_station_observation: bool,
    show_ess_threshold: bool = False,
) -> list:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_DA_OBS,
            marker="x",
            linestyle="none",
            markersize=6.2,
            markeredgewidth=1.6,
            label="observation used for data assimilation",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_DA_OBS,
            marker="o",
            linestyle="none",
            markersize=3.2,
            label="satellite observation",
        ),
        Line2D([0], [0], color="black", lw=1.8, label="open loop"),
        _EnsembleLegendHandle(
            (
                Patch(
                    facecolor="#bfc6cf",
                    edgecolor="#666666",
                    linewidth=0.9,
                    alpha=BAND_ALPHA,
                ),
                Line2D([0], [0], color="#666666", lw=1.2),
            ),
            "ensemble (min - max, mean)",
        ),
        Line2D([0], [0], color="#666666", lw=1.2, ls="--", label="data assimilation event"),
    ]
    if show_station_observation:
        handles.insert(
            4,
            Line2D(
                [0],
                [0],
                color=COLOR_DA_OBS,
                lw=LW_DA_OBS,
                label="station observation",
            ),
        )
    return handles


def _result_overview_legend_handler_map() -> dict[type, object]:
    handler_map = dict(score_legend_handler_map())
    handler_map[_EnsembleLegendHandle] = _EnsembleLegendHandler()
    return handler_map


def _build_result_overview_legends(
    fig,
    *,
    show_station_observation: bool,
    score_variables: list[str] | None = None,
    show_ess_threshold: bool = False,
) -> list:
    overview_handles = _build_result_overview_legend_handles(
        show_station_observation=show_station_observation,
        show_ess_threshold=show_ess_threshold and not score_variables,
    )

    if not score_variables:
        legend = fig.legend(
            handles=overview_handles,
            handler_map=_result_overview_legend_handler_map(),
            loc="lower left",
            bbox_to_anchor=(0.055, 0.008, 0.865, 0.06),
            bbox_transform=fig.transFigure,
            mode="expand",
            ncol=3,
            frameon=False,
            fontsize=8.0,
            handlelength=2.45,
            handleheight=1.25,
            columnspacing=1.1,
            handletextpad=0.45,
            borderaxespad=0.0,
        )
        return [legend]

    score_handles = score_legend_handles(score_variables, include_da_event=False)
    overview_legend = fig.legend(
        handles=overview_handles,
        handler_map=_result_overview_legend_handler_map(),
        loc="lower left",
        bbox_to_anchor=(0.055, 0.038, 0.865, 0.032),
        bbox_transform=fig.transFigure,
        mode="expand",
        ncol=4,
        frameon=False,
        fontsize=6.2,
        handlelength=2.4,
        handleheight=1.22,
        columnspacing=0.8,
        handletextpad=0.32,
        borderaxespad=0.0,
    )
    score_legend = fig.legend(
        handles=score_handles,
        handler_map=_result_overview_legend_handler_map(),
        loc="lower left",
        bbox_to_anchor=(0.055, 0.007, 0.865, 0.032),
        bbox_transform=fig.transFigure,
        mode="expand",
        ncol=5,
        frameon=False,
        fontsize=6.2,
        handlelength=2.4,
        handleheight=1.22,
        columnspacing=0.8,
        handletextpad=0.62,
        borderaxespad=0.0,
    )
    return [overview_legend, score_legend]


def _legend_band_bottom(fig, legends: list, *, gap: float = 0.008, minimum: float = 0.04) -> float:
    if not legends:
        return minimum
    renderer = fig.canvas.get_renderer()
    top = 0.0
    for legend in legends:
        bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        top = max(top, float(bbox.y1))
    return max(minimum, top + gap)

def _apply_result_y_ticks(ax, panel: str) -> None:
    from matplotlib.ticker import MultipleLocator

    scale = result_axis_scale(panel, float(getattr(ax.dataLim, "ymax", 0.0) or 0.0))
    if scale is None:
        return
    step, upper = scale
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2.0))


def _result_axis_scale_from_max(panel: str, data_max: float) -> tuple[float, float] | None:
    return result_axis_scale(panel, data_max, shared=True)


def _max_abs_value_frame(frame: pd.DataFrame | None, *cols: str) -> float:
    if frame is None or frame.empty:
        return 0.0
    maxima = []
    for col in cols:
        if col in frame.columns:
            maxima.append(pd.to_numeric(frame[col], errors="coerce").max())
    maxima = [float(val) for val in maxima if pd.notna(val)]
    return max(maxima) if maxima else 0.0


def _max_abs_value_series(*series_list: pd.Series | None) -> float:
    maxima: list[float] = []
    for series in series_list:
        if series is None or series.empty:
            continue
        value = pd.to_numeric(series, errors="coerce").max()
        if pd.notna(value):
            maxima.append(float(value))
    return max(maxima) if maxima else 0.0


def _shared_result_scales(
    specs: list[PanelSpec],
    *,
    roi_swe_model: pd.DataFrame | None,
    roi_swe_env: pd.DataFrame | None,
    roi_snow_depth_model: pd.DataFrame | None,
    roi_snow_depth_env: pd.DataFrame | None,
    station_panels: dict[tuple[str, str], StationPanelData],
) -> dict[str, tuple[float, float]]:
    swe_max = 0.0
    sd_max = 0.0

    for spec in specs:
        if spec.panel == "roi-swe":
            swe_max = max(swe_max, _max_abs_value_frame(roi_swe_model, "swe"))
            swe_max = max(swe_max, _max_abs_value_frame(roi_swe_env, "value_min", "value_max", "value_mean"))
        elif spec.panel == "roi-sd":
            sd_max = max(sd_max, _max_abs_value_frame(roi_snow_depth_model, "snow_depth"))
            sd_max = max(sd_max, _max_abs_value_frame(roi_snow_depth_env, "value_min", "value_max", "value_mean"))
        elif spec.panel.startswith("station-") and spec.station_id is not None:
            value_col = _STATION_PANEL_META[spec.panel]["value_col"]
            bundle = station_panels.get((spec.station_id.lower(), value_col))
            if bundle is None:
                continue
            bundle_max = _max_abs_value_series(bundle.open_loop, bundle.obs, *(bundle.members or []))
            if spec.panel == "station-swe":
                swe_max = max(swe_max, bundle_max)
            elif spec.panel == "station-sd":
                sd_max = max(sd_max, bundle_max)

    scales: dict[str, tuple[float, float]] = {}
    swe_scale = _result_axis_scale_from_max("roi-swe", swe_max)
    if swe_scale is not None:
        scales["SWE"] = swe_scale
    sd_scale = _result_axis_scale_from_max("roi-sd", sd_max)
    if sd_scale is not None:
        scales["SD"] = sd_scale
    return scales


def _apply_shared_result_scale(ax, panel: str, shared_scales: dict[str, tuple[float, float]]) -> None:
    from matplotlib.ticker import MultipleLocator

    if panel in {"roi-swe", "station-swe"}:
        scale = shared_scales.get("SWE")
    elif panel in {"roi-sd", "station-sd"}:
        scale = shared_scales.get("SD")
    else:
        scale = None

    if scale is None:
        _apply_result_y_ticks(ax, panel)
        return

    step, upper = scale
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2.0))


def _apply_fraction_ticks(ax) -> None:
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])


def _apply_ess_ticks(ax, ensemble_size: int | None, *, threshold: float | None = None) -> None:
    ticks = ess_axis_ticks(ensemble_size, threshold=threshold)
    if ticks:
        ax.set_yticks(ticks)


def _apply_time_axis_labels(axes, x_bounds: tuple[pd.Timestamp, pd.Timestamp] | None) -> None:
    import matplotlib.dates as mdates

    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%b")
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    if x_bounds is None:
        return

    tick_values = locator.tick_values(x_bounds[0].to_pydatetime(), x_bounds[1].to_pydatetime())
    tick_dates = [pd.Timestamp(mdates.num2date(val)).tz_localize(None) for val in tick_values]
    if not tick_dates:
        return

    labels: list[str] = []
    prev_year: int | None = None
    for idx, tick_dt in enumerate(tick_dates):
        if idx == 0 or tick_dt.year != prev_year:
            labels.append(tick_dt.strftime("%b\n%Y"))
        else:
            labels.append(tick_dt.strftime("%b"))
        prev_year = tick_dt.year
    axes[-1].set_xticks(tick_values)
    axes[-1].set_xticklabels(labels)
    axes[-1].tick_params(axis="x", labelsize=8.4)


def _panel_has_data(
    spec: PanelSpec,
    *,
    scf_obs: pd.DataFrame | None,
    scf_model: pd.DataFrame | None,
    wet_obs: pd.DataFrame | None,
    wet_model: pd.DataFrame | None,
    wsl_obs: pd.DataFrame | None,
    wsl_model: pd.DataFrame | None,
    scf_env: pd.DataFrame | None,
    wet_env: pd.DataFrame | None,
    wsl_env: pd.DataFrame | None,
    roi_swe_model: pd.DataFrame | None,
    roi_swe_members: list[pd.Series] | None,
    roi_snow_depth_model: pd.DataFrame | None,
    roi_snow_depth_members: list[pd.Series] | None,
    station_panels: dict[tuple[str, str], StationPanelData],
    ess_panel: EssPanelData | None,
    score_points: pd.DataFrame | None = None,
) -> bool:
    if _is_score_panel(spec.panel):
        return not _score_panel_points_for_metric(score_points, spec.panel).empty
    if spec.panel == "fSC":
        return any(frame is not None and not frame.empty for frame in (scf_obs, scf_model, scf_env))
    if spec.panel == "WSF":
        return any(frame is not None and not frame.empty for frame in (wet_obs, wet_model, wet_env))
    if spec.panel == "WSLA":
        return any(frame is not None and not frame.empty for frame in (wsl_obs, wsl_model, wsl_env))
    if spec.panel == "roi-swe":
        return (roi_swe_model is not None and not roi_swe_model.empty) or bool(roi_swe_members)
    if spec.panel == "roi-sd":
        return (roi_snow_depth_model is not None and not roi_snow_depth_model.empty) or bool(roi_snow_depth_members)
    if spec.panel == "ess":
        return bool(ess_panel and ess_panel.has_data)
    if spec.station_id is None:
        return False
    value_col = _STATION_PANEL_META[spec.panel]["value_col"]
    bundle = station_panels.get((spec.station_id.lower(), value_col))
    return bundle.has_data if bundle is not None else False


def _filter_panel_specs(
    specs: list[PanelSpec],
    *,
    strict: bool,
    scf_obs: pd.DataFrame | None,
    scf_model: pd.DataFrame | None,
    wet_obs: pd.DataFrame | None,
    wet_model: pd.DataFrame | None,
    wsl_obs: pd.DataFrame | None,
    wsl_model: pd.DataFrame | None,
    scf_env: pd.DataFrame | None,
    wet_env: pd.DataFrame | None,
    wsl_env: pd.DataFrame | None,
    roi_swe_model: pd.DataFrame | None,
    roi_swe_members: list[pd.Series] | None,
    roi_snow_depth_model: pd.DataFrame | None,
    roi_snow_depth_members: list[pd.Series] | None,
    station_panels: dict[tuple[str, str], StationPanelData],
    ess_panel: EssPanelData | None,
    score_points: pd.DataFrame | None = None,
) -> list[PanelSpec]:
    out: list[PanelSpec] = []
    for spec in specs:
        if _panel_has_data(
            spec,
            scf_obs=scf_obs,
            scf_model=scf_model,
            wet_obs=wet_obs,
            wet_model=wet_model,
            wsl_obs=wsl_obs,
            wsl_model=wsl_model,
            scf_env=scf_env,
            wet_env=wet_env,
            wsl_env=wsl_env,
            roi_swe_model=roi_swe_model,
            roi_swe_members=roi_swe_members,
            roi_snow_depth_model=roi_snow_depth_model,
            roi_snow_depth_members=roi_snow_depth_members,
            station_panels=station_panels,
            ess_panel=ess_panel,
            score_points=score_points,
        ):
            out.append(spec)
            continue
        if strict:
            station_note = f" ({spec.station_id})" if spec.station_id else ""
            raise ValueError(f"No data available for requested panel {spec.panel}{station_note}")
        logger.debug("Skipping empty result overview panel {}", spec.panel)
    return out


def plot_result_overview(
    *,
    scf_obs: pd.DataFrame | None,
    scf_model: pd.DataFrame | None,
    wet_obs: pd.DataFrame | None,
    wet_model: pd.DataFrame | None,
    scf_env: pd.DataFrame | None,
    wet_env: pd.DataFrame | None,
    output: Path,
    wsl_obs: pd.DataFrame | None = None,
    wsl_model: pd.DataFrame | None = None,
    wsl_env: pd.DataFrame | None = None,
    wsl_prior_coverage: pd.DataFrame | None = None,
    assim_events: list[AssimilationEvent] | None = None,
    mode: str = "band",
    roi_swe_model: pd.DataFrame | None = None,
    roi_swe_members: list[pd.Series] | None = None,
    roi_snow_depth_model: pd.DataFrame | None = None,
    roi_snow_depth_members: list[pd.Series] | None = None,
    panel_specs: list[PanelSpec] | None = None,
    station_panels: dict[tuple[str, str], StationPanelData] | None = None,
    ess_panel: EssPanelData | None = None,
    score_points: pd.DataFrame | None = None,
    strict_panels: bool = False,
    x_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    backend: str = "Agg",
) -> None:
    """Render the result overview into one PNG."""
    import matplotlib

    matplotlib.use(backend or "Agg")
    set_matplotlib_text_black(matplotlib)
    import matplotlib.pyplot as plt

    specs = panel_specs or list(_DEFAULT_PANELS)
    station_panels = station_panels or {}
    score_points = score_points.copy() if score_points is not None else None
    mode = (mode or "band").lower()
    if mode not in {"band", "members"}:
        mode = "band"

    events = list(assim_events or [])
    if score_points is not None and not score_points.empty and "plot_x" not in score_points.columns:
        score_dates = sorted({pd.Timestamp(ev.date).normalize() for ev in events})
        if not score_dates:
            score_dates = sorted(pd.to_datetime(score_points["assimilation_date"]).dt.normalize().unique())
        if score_dates:
            score_points = compute_event_skill_plot_positions(score_points, assimilation_dates=score_dates)

    specs = _filter_panel_specs(
        specs,
        strict=strict_panels,
        scf_obs=scf_obs,
        scf_model=scf_model,
        wet_obs=wet_obs,
        wet_model=wet_model,
        wsl_obs=wsl_obs,
        wsl_model=wsl_model,
        scf_env=scf_env,
        wet_env=wet_env,
        wsl_env=wsl_env,
        roi_swe_model=roi_swe_model,
        roi_swe_members=roi_swe_members,
        roi_snow_depth_model=roi_snow_depth_model,
        roi_snow_depth_members=roi_snow_depth_members,
        station_panels=station_panels,
        ess_panel=ess_panel,
        score_points=score_points,
    )
    if not specs:
        raise ValueError("No data available to plot.")

    height_ratios = [
        0.5 if spec.panel == "ess" else OVERVIEW_SCORE_PANEL_HEIGHT_FACTOR if _is_score_panel(spec.panel) else 1.0
        for spec in specs
    ]
    total_height_units = sum(height_ratios)
    fig, axes = plt.subplots(
        len(specs),
        1,
        figsize=(FIGWIDTH_OVERVIEW_PAPER, FIGHEIGHT_OVERVIEW_ROW * total_height_units),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if len(specs) == 1:
        axes = [axes]
    title_artists: list[tuple[object, object]] = []

    roi_swe_env = _band_frame(roi_swe_members)
    roi_snow_depth_env = _band_frame(roi_snow_depth_members)
    shared_scales = _shared_result_scales(
        specs,
        roi_swe_model=roi_swe_model,
        roi_swe_env=roi_swe_env,
        roi_snow_depth_model=roi_snow_depth_model,
        roi_snow_depth_env=roi_snow_depth_env,
        station_panels=station_panels,
    )
    data_x_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None
    label_axes: list[tuple[object, object]] = []
    show_station_observation = False
    score_panel_indices = [idx for idx, spec in enumerate(specs) if _is_score_panel(spec.panel)]
    score_legend_variables = (
        sorted(score_points["variable"].astype(str).unique(), key=score_variable_sort_key)
        if score_panel_indices and score_points is not None and not score_points.empty
        else []
    )
    show_ess_threshold = any(spec.panel == "ess" and ess_panel is not None and ess_panel.threshold is not None for spec in specs)

    for idx, (ax, spec) in enumerate(zip(axes, specs)):
        letter = ascii_lowercase[idx] if idx < len(ascii_lowercase) else str(idx + 1)
        station_data: StationPanelData | None = None
        current_ess_panel = ess_panel if spec.panel == "ess" else None
        if spec.panel.startswith("station-") and spec.station_id is not None:
            value_col = _STATION_PANEL_META[spec.panel]["value_col"]
            station_data = station_panels[(spec.station_id.lower(), value_col)]
        panel_style = None if _is_score_panel(spec.panel) else _panel_style(spec.panel)

        title_artist = ax.set_title(
            f"({letter}) {(_ess_panel_title(spec, current_ess_panel) if spec.panel == 'ess' else _panel_title(spec, station_data))}",
            loc="left",
            fontsize=9.4,
            pad=result_title_pad(bool(events)),
        )
        title_artists.append((ax, title_artist))
        center_assim = spec.panel in {"roi-swe", "roi-sd", "station-swe", "station-sd"}
        if not _is_score_panel(spec.panel):
            _draw_all_assim(ax, events, center_of_day=center_assim)

        if _is_score_panel(spec.panel):
            score_metric = _score_metric_for_panel(spec.panel)
            metric_points = _score_panel_points_for_metric(score_points, spec.panel)
            score_variables = sorted(metric_points["variable"].astype(str).unique(), key=score_variable_sort_key)
            draw_score_metric_panel(
                ax,
                points=metric_points,
                metric=score_metric,
                variables=score_variables,
                assimilation_events=events,
            )
            ax.set_ylabel(_PANEL_YLABELS[spec.panel], fontsize=8.6)
            ax.set_ylim(*score_metric_ylim(metric_points, score_metric))
            bounds = _date_bounds_frames(pd.DataFrame({"date": pd.to_datetime(metric_points["assimilation_date"])}))
        elif spec.panel == "fSC":
            scf_obs_points = _finite_value_points(scf_obs, "scf")
            if mode == "band" and scf_env is not None and not scf_env.empty:
                ax.fill_between(
                    scf_env["date"],
                    scf_env["value_min"],
                    scf_env["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    label="_nolegend_",
                )
                ax.plot(
                    scf_env["date"],
                    scf_env["value_mean"],
                    "-",
                    color=panel_style["line"],
                    lw=LW_MEAN,
                    alpha=0.9,
                    label="_nolegend_",
                )
            if scf_model is not None and not scf_model.empty:
                ax.plot(scf_model["date"], scf_model["scf"], "-", color="black", lw=LW_OPEN, label="_nolegend_")
            if spec.show_obs and scf_obs_points is not None and not scf_obs_points.empty:
                ax.plot(
                    scf_obs_points["date"],
                    scf_obs_points["scf"],
                    linestyle="none",
                    marker="o",
                    ms=2.8,
                    color=COLOR_DA_OBS,
                    label="_nolegend_",
                )
                scf_dates = [pd.to_datetime(ev.date) for ev in events if ev.variable == "scf"]
                if scf_dates:
                    draw_assimilation_markers(
                        ax,
                        dates=scf_dates,
                        obs=scf_obs_points,
                        value_col="scf",
                        color=COLOR_DA_OBS,
                        label="_nolegend_",
                        size=SIZE_DA_OBS * 0.8,
                        linewidth=LW_DA_OBS,
                        draw_vlines=False,
                    )
            ax.set_ylabel(_PANEL_YLABELS[spec.panel], fontsize=8.6)
            ax.set_ylim(0, 1)
            apply_fraction_grid(ax, y_step=0.2)
            _apply_fraction_ticks(ax)
            bounds = _date_bounds_frames(scf_obs, scf_model, scf_env)
        elif spec.panel == "WSF":
            wet_obs_points = _finite_value_points(wet_obs, "wet_snow_fraction")
            if mode == "band" and wet_env is not None and not wet_env.empty:
                ax.fill_between(
                    wet_env["date"],
                    wet_env["value_min"],
                    wet_env["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    label="_nolegend_",
                )
                ax.plot(
                    wet_env["date"],
                    wet_env["value_mean"],
                    "-",
                    color=panel_style["line"],
                    lw=LW_MEAN,
                    alpha=0.9,
                    label="_nolegend_",
                )
            if wet_model is not None and not wet_model.empty:
                ax.plot(
                    wet_model["date"],
                    wet_model["wet_snow_fraction"],
                    "-",
                    color="black",
                    lw=LW_OPEN,
                    label="_nolegend_",
                )
            if spec.show_obs and wet_obs_points is not None and not wet_obs_points.empty:
                ax.plot(
                    wet_obs_points["date"],
                    wet_obs_points["wet_snow_fraction"],
                    linestyle="none",
                    marker="o",
                    ms=2.8,
                    color=COLOR_DA_OBS,
                    label="_nolegend_",
                )
                wet_dates = [pd.to_datetime(ev.date) for ev in events if ev.variable == "wet_snow"]
                if wet_dates:
                    draw_assimilation_markers(
                        ax,
                        dates=wet_dates,
                        obs=wet_obs_points,
                        value_col="wet_snow_fraction",
                        color=COLOR_DA_OBS,
                        label="_nolegend_",
                        size=SIZE_DA_OBS * 0.8,
                        linewidth=LW_DA_OBS,
                        draw_vlines=False,
                    )
            ax.set_ylabel(_PANEL_YLABELS[spec.panel], fontsize=8.6)
            ax.set_ylim(0, 1)
            apply_fraction_grid(ax, y_step=0.2)
            _apply_fraction_ticks(ax)
            bounds = _date_bounds_frames(wet_obs, wet_model, wet_env)
        elif spec.panel == "WSLA":
            wsl_obs_points = _finite_value_points(wsl_obs, "wet_snow_line")
            if mode == "band" and wsl_env is not None and not wsl_env.empty:
                ax.fill_between(
                    wsl_env["date"],
                    wsl_env["value_min"],
                    wsl_env["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    label="_nolegend_",
                )
                ax.plot(
                    wsl_env["date"],
                    wsl_env["value_mean"],
                    "-",
                    color=panel_style["line"],
                    lw=LW_MEAN,
                    alpha=0.95,
                    label="_nolegend_",
                )
            _draw_wsl_prior_coverage_markers(ax, wsl_prior_coverage, color=panel_style["line"])
            if wsl_model is not None and not wsl_model.empty:
                ax.plot(
                    wsl_model["date"],
                    wsl_model["wet_snow_line"],
                    "-",
                    color="black",
                    lw=LW_OPEN,
                    label="_nolegend_",
                )
            if spec.show_obs and wsl_obs_points is not None and not wsl_obs_points.empty:
                ax.plot(
                    wsl_obs_points["date"],
                    wsl_obs_points["wet_snow_line"],
                    linestyle="none",
                    marker="o",
                    ms=2.8,
                    color=COLOR_DA_OBS,
                    label="_nolegend_",
                )
                wsl_dates = [pd.to_datetime(ev.date) for ev in events if ev.variable == "wet_snow_line"]
                if wsl_dates:
                    draw_assimilation_markers(
                        ax,
                        dates=wsl_dates,
                        obs=wsl_obs_points,
                        value_col="wet_snow_line",
                        color=COLOR_DA_OBS,
                        label="_nolegend_",
                        size=SIZE_DA_OBS * 0.8,
                        linewidth=LW_DA_OBS,
                        draw_vlines=False,
                    )
            ax.set_ylabel(_PANEL_YLABELS[spec.panel], fontsize=8.6)
            apply_fraction_grid(ax, y_step=None)
            bounds = _date_bounds_frames(wsl_obs, wsl_model, wsl_env, wsl_prior_coverage)
        elif spec.panel == "roi-swe":
            if roi_swe_env is not None and not roi_swe_env.empty:
                ax.fill_between(
                    roi_swe_env["date"],
                    roi_swe_env["value_min"],
                    roi_swe_env["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    label="_nolegend_",
                )
                ax.plot(
                    roi_swe_env["date"],
                    roi_swe_env["value_mean"],
                    "-",
                    color=panel_style["line"],
                    lw=LW_MEAN,
                    alpha=0.95,
                    label="_nolegend_",
                )
            if roi_swe_model is not None and not roi_swe_model.empty:
                ax.plot(
                    roi_swe_model["date"],
                    roi_swe_model["swe"],
                    "-",
                    color="black",
                    lw=LW_OPEN,
                    label="_nolegend_",
                )
            ax.set_ylabel(_PANEL_YLABELS[spec.panel], fontsize=8.6)
            apply_fraction_grid(ax, y_step=None)
            _apply_shared_result_scale(ax, spec.panel, shared_scales)
            bounds = _date_bounds_frames(roi_swe_model, roi_swe_env)
        elif spec.panel == "roi-sd":
            if roi_snow_depth_env is not None and not roi_snow_depth_env.empty:
                ax.fill_between(
                    roi_snow_depth_env["date"],
                    roi_snow_depth_env["value_min"],
                    roi_snow_depth_env["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    label="_nolegend_",
                )
                ax.plot(
                    roi_snow_depth_env["date"],
                    roi_snow_depth_env["value_mean"],
                    "-",
                    color=panel_style["line"],
                    lw=LW_MEAN,
                    alpha=0.95,
                    label="_nolegend_",
                )
            if roi_snow_depth_model is not None and not roi_snow_depth_model.empty:
                ax.plot(
                    roi_snow_depth_model["date"],
                    roi_snow_depth_model["snow_depth"],
                    "-",
                    color="black",
                    lw=LW_OPEN,
                    label="_nolegend_",
                )
            ax.set_ylabel(_PANEL_YLABELS[spec.panel], fontsize=8.6)
            apply_fraction_grid(ax, y_step=None)
            _apply_shared_result_scale(ax, spec.panel, shared_scales)
            bounds = _date_bounds_frames(roi_snow_depth_model, roi_snow_depth_env)
        elif spec.panel == "ess":
            if current_ess_panel is None or current_ess_panel.series is None or current_ess_panel.series.empty:
                raise ValueError("Missing ESS panel data")
            ess_series = current_ess_panel.series
            ax.plot(
                ess_series["date"],
                ess_series["ess"],
                marker="o",
                ms=3.2,
                lw=0.0,
                ls="none",
                color="#000000",
                zorder=25,
            )
            ax.set_ylabel(_PANEL_YLABELS[spec.panel], fontsize=8.6)
            if current_ess_panel.ensemble_size is not None and current_ess_panel.ensemble_size > 0:
                ax.set_ylim(0.0, float(current_ess_panel.ensemble_size))
                _apply_ess_ticks(
                    ax,
                    current_ess_panel.ensemble_size,
                    threshold=current_ess_panel.threshold,
                )
            if current_ess_panel.threshold is not None:
                from matplotlib.lines import Line2D

                ax.axhline(current_ess_panel.threshold, color="#d62728", lw=0.9, ls="--", zorder=10)
                ax.legend(
                    [Line2D([0], [0], color="#d62728", lw=0.9, ls="--")],
                    ["ESS threshold"],
                    loc="lower right",
                    bbox_to_anchor=(1.0, 1.28),
                    frameon=False,
                    fontsize=7.0,
                    handlelength=1.8,
                    borderaxespad=0.0,
                )
            apply_fraction_grid(ax, y_step=None)
            bounds = _date_bounds_frames(ess_series)
        else:
            if station_data is None:
                raise ValueError(f"Missing station panel data for {spec.panel}")
            env_frame = _band_frame(station_data.members, q_low=0.0, q_high=1.0)
            if env_frame is not None and not env_frame.empty:
                ax.fill_between(
                    env_frame["date"],
                    env_frame["value_min"],
                    env_frame["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    label="_nolegend_",
                    zorder=2,
                )
                ax.plot(
                    env_frame["date"],
                    env_frame["value_mean"],
                    "-",
                    color=panel_style["line"],
                    alpha=0.95,
                    label="_nolegend_",
                    lw=LW_MEAN,
                    zorder=4,
                )
            if station_data.open_loop is not None and not station_data.open_loop.empty:
                ax.plot(
                    station_data.open_loop.index,
                    station_data.open_loop.values,
                    "-",
                    color="black",
                    lw=LW_OPEN,
                    label="_nolegend_",
                    zorder=5,
                )
            if spec.show_obs and station_data.obs is not None and not station_data.obs.empty:
                value_col = _STATION_PANEL_META[spec.panel]["value_col"]
                ax.plot(
                    station_data.obs.index,
                    station_data.obs.values,
                    "-",
                    color=COLOR_DA_OBS,
                    lw=LW_DA_OBS,
                    label="_nolegend_",
                    zorder=6,
                )
                draw_assimilation_markers(
                    ax,
                    dates=_station_assimilation_dates(events, spec.panel),
                    obs=_station_obs_frame(station_data.obs, value_col=value_col),
                    value_col=value_col,
                    color=COLOR_DA_OBS,
                    label="_nolegend_",
                    size=SIZE_DA_OBS * 0.8,
                    linewidth=LW_DA_OBS,
                    zorder=7,
                    draw_vlines=False,
                )
                show_station_observation = True
            ax.set_ylabel(_PANEL_YLABELS[spec.panel], fontsize=8.6)
            apply_fraction_grid(ax, y_step=None)
            _apply_shared_result_scale(ax, spec.panel, shared_scales)
            bounds = _date_bounds_series(
                station_data.open_loop,
                station_data.obs,
                *(station_data.members or []),
            )

        if bounds is not None:
            if data_x_bounds is None:
                data_x_bounds = bounds
            else:
                data_x_bounds = (min(data_x_bounds[0], bounds[0]), max(data_x_bounds[1], bounds[1]))

    effective_x_bounds = _pad_single_day_bounds(x_bounds or data_x_bounds)
    if effective_x_bounds is not None:
        for ax in axes:
            ax.set_xlim(*effective_x_bounds)
    if events:
        for idx, ax in enumerate(axes):
            center_assim = specs[idx].panel in {"roi-swe", "roi-sd", "station-swe", "station-sd"}
            label_axis = _add_assim_label_axis(ax, events, idx, center_of_day=center_assim)
            if label_axis is not None:
                label_axes.append((ax, label_axis))
    _apply_time_axis_labels(axes, effective_x_bounds)
    for ax, label_axis in label_axes:
        label_axis.set_xlim(ax.get_xlim())

    axes[-1].set_xlabel("")
    fig.tight_layout(rect=(-0.02, 0.04, 0.985, 1.0), h_pad=0.74)
    fig.align_ylabels(axes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    global_left_disp: float | None = None
    for ax, _title_artist in title_artists:
        tick_labels = [label for label in ax.get_yticklabels() if label.get_text()]
        if not tick_labels:
            continue
        left_disp = min(label.get_window_extent(renderer).x0 for label in tick_labels) - 6.0
        if global_left_disp is None or left_disp < global_left_disp:
            global_left_disp = left_disp
    if global_left_disp is None:
        global_left_disp = min(ax.bbox.x0 for ax in axes)
    for ax, title_artist in title_artists:
        x_axes = ax.transAxes.inverted().transform((global_left_disp, ax.bbox.y1))[0]
        title_artist.set_x(x_axes)
    legends = _build_result_overview_legends(
        fig,
        show_station_observation=show_station_observation,
        score_variables=score_legend_variables,
        show_ess_threshold=show_ess_threshold,
    )
    fig.canvas.draw()
    legend_bottom = _legend_band_bottom(fig, legends, gap=0.008, minimum=0.04)
    fig.tight_layout(rect=(-0.02, legend_bottom, 0.985, 1.0), h_pad=0.74)
    fig.align_ylabels(axes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    global_left_disp = None
    for ax, _title_artist in title_artists:
        tick_labels = [label for label in ax.get_yticklabels() if label.get_text()]
        if not tick_labels:
            continue
        left_disp = min(label.get_window_extent(renderer).x0 for label in tick_labels) - 6.0
        if global_left_disp is None or left_disp < global_left_disp:
            global_left_disp = left_disp
    if global_left_disp is None:
        global_left_disp = min(ax.bbox.x0 for ax in axes)
    for ax, title_artist in title_artists:
        x_axes = ax.transAxes.inverted().transform((global_left_disp, ax.bbox.y1))[0]
        title_artist.set_x(x_axes)
    force_figure_text_black(fig, axes)
    save_figure_png(fig, output)
    plt.close(fig)


def cli_main(argv: list[str] | None = None, *, configure_logger: bool = True) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="oa-da-plot-result-overview",
        description="Plot the setup result overview for fractions, ROI aggregates, and optional station panels.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory (setup/projects/project_YYYY_YYYY)")
    parser.add_argument("--setup-dir", type=Path, help="Setup directory (default: project_dir/../..)")
    parser.add_argument("--scf-obs-csv", type=Path, help="Path to scf_summary.csv (obs)")
    parser.add_argument("--wet-obs-csv", type=Path, help="Path to wet_snow_summary.csv (obs)")
    parser.add_argument("--wsl-obs-csv", type=Path, help="Path to wet_snow_line_diagnostics.csv (obs)")
    parser.add_argument("--scf-model-csv", type=Path, help="Model SCF CSV (date/time + scf)")
    parser.add_argument("--wet-model-csv", type=Path, help="Model WSF CSV (date/time + wet_snow_fraction)")
    parser.add_argument("--wsl-model-csv", type=Path, help="Model WSLA CSV (date/time + wet_snow_line)")
    parser.add_argument("--scf-env-csv", type=Path, help="SCF envelope CSV (value_min/value_max/value_mean)")
    parser.add_argument("--wet-env-csv", type=Path, help="WSF envelope CSV (value_min/value_max/value_mean)")
    parser.add_argument("--wsl-env-csv", type=Path, help="WSLA envelope CSV (value_min/value_max/value_mean)")
    parser.add_argument("--output", type=Path, help="Output PNG path (default: <project>/results/plots/results/result_overview.png)")
    parser.add_argument("--custom-config", type=Path, help="Custom panel YAML (default: <project-dir>/plots.yml)")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    parser.add_argument("--mode", choices=["band", "members"], default="band", help="Plot mode: band (default) or members")
    parser.add_argument("--backend", default="Agg", help="Matplotlib backend (default: Agg)")
    args = parser.parse_args(argv)

    if configure_logger:
        configure_cli_logger(args.log_level)

    project_dir = Path(args.project_dir)
    setup_dir = Path(args.setup_dir) if args.setup_dir else project_dir.parent.parent
    scf_obs_path = Path(args.scf_obs_csv) if args.scf_obs_csv else resolve_fraction_summary_path(setup_dir, project_dir, "scf_summary.csv")
    wet_obs_path = Path(args.wet_obs_csv) if args.wet_obs_csv else resolve_fraction_summary_path(setup_dir, project_dir, "wet_snow_summary.csv")
    wsl_obs_path = (
        Path(args.wsl_obs_csv)
        if args.wsl_obs_csv
        else resolve_fraction_summary_path(setup_dir, project_dir, "wet_snow_line_diagnostics.csv")
    )
    scf_env_path = Path(args.scf_env_csv) if args.scf_env_csv else project_fraction_envelope_path(project_dir, "scf")
    wet_env_path = Path(args.wet_env_csv) if args.wet_env_csv else project_fraction_envelope_path(project_dir, "wet_snow")
    wsl_env_path = Path(args.wsl_env_csv) if args.wsl_env_csv else project_fraction_envelope_path(project_dir, "wet_snow_line")

    scf_obs = _load_scf_obs_series(scf_obs_path)
    wet_obs = load_fraction_series(wet_obs_path, "wet_snow_fraction")
    wsl_obs = load_fraction_series(wsl_obs_path, "wet_snow_line", preserve_missing_values=True)
    scf_model = load_fraction_series(Path(args.scf_model_csv), "scf") if args.scf_model_csv else None
    wet_model = load_fraction_series(Path(args.wet_model_csv), "wet_snow_fraction") if args.wet_model_csv else None
    wsl_model = (
        load_fraction_series(Path(args.wsl_model_csv), "wet_snow_line", preserve_missing_values=True)
        if args.wsl_model_csv
        else None
    )
    if scf_model is None:
        scf_model = load_open_loop_fraction_series(project_dir, "point_scf_roi.csv", "scf")
    if wet_model is None:
        wet_model = load_open_loop_fraction_series(project_dir, "point_wet_snow_roi.csv", "wet_snow_fraction")
    if wsl_model is None:
        wsl_model = load_open_loop_fraction_series(
            project_dir,
            "point_wet_snow_line_roi.csv",
            "wet_snow_line",
            preserve_missing_values=True,
        )
    roi_swe_model = load_open_loop_fraction_series(project_dir, "point_swe_roi.csv", "swe")
    roi_swe_members = load_member_series(project_dir, "point_swe_roi.csv", "swe")
    roi_snow_depth_model = load_open_loop_fraction_series(project_dir, "point_snow_depth_roi.csv", "snow_depth")
    roi_snow_depth_members = load_member_series(project_dir, "point_snow_depth_roi.csv", "snow_depth")
    scf_env = load_fraction_series(scf_env_path, "value_mean")
    if scf_env is not None and not scf_env.empty and {"value_min", "value_max"}.issubset(scf_env.columns) is False:
        scf_env = None
    wet_env = load_fraction_series(wet_env_path, "value_mean")
    if wet_env is not None and not wet_env.empty and {"value_min", "value_max"}.issubset(wet_env.columns) is False:
        wet_env = None
    if args.wsl_env_csv:
        wsl_env = load_fraction_series(wsl_env_path, "value_mean")
        if wsl_env is not None and not wsl_env.empty and {"value_min", "value_max"}.issubset(wsl_env.columns) is False:
            wsl_env = None
    else:
        try:
            wsl_env = _default_wsl_overview_env(project_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prior-member WSLA overview series failed, plotting without ensemble WSLA band: {}", exc)
            wsl_env = None
    wsl_prior_coverage = _load_wsl_prior_coverage_frame(project_dir)

    if scf_obs is None or scf_obs.empty:
        logger.warning("SCF obs not found at {} - plotting without obs points", scf_obs_path)
    if wet_obs is None or wet_obs.empty:
        logger.warning("Wet-snow obs not found at {} - plotting without obs points", wet_obs_path)
    if wsl_obs is None or wsl_obs.empty:
        logger.warning("Wet-snow-line obs not found at {} - plotting without obs points", wsl_obs_path)

    try:
        assim_events = load_assimilation_events(project_dir)
    except (FileNotFoundError, ValueError):
        assim_events = []

    try:
        ess_df = load_setup_ess_series(project_dir)
        ess_ensemble_size = int(ess_df["n"].iloc[0]) if "n" in ess_df.columns and not ess_df.empty else None
        ess_panel = EssPanelData(
            series=ess_df,
            ensemble_size=ess_ensemble_size,
            threshold=load_setup_ess_threshold(project_dir, ensemble_size=ess_ensemble_size),
        )
    except FileNotFoundError:
        ess_panel = EssPanelData(series=None, ensemble_size=None, threshold=None)

    stations_df = _load_setup_stations_df(project_dir, setup_dir)
    project_time_bounds = _project_time_bounds(project_dir)

    if all(
        x is None or x.empty
        for x in (scf_obs, wet_obs, scf_model, wet_model, scf_env, wet_env, roi_swe_model, roi_snow_depth_model, ess_panel.series)
    ) and all(x is None or x.empty for x in (wsl_obs, wsl_model, wsl_env, wsl_prior_coverage)) and not roi_swe_members and not roi_snow_depth_members:
        logger.error("No data available to plot. Provide at least one obs/model series.")
        return 1

    custom_config_path = (
        Path(abspath_relative_to(project_dir, args.custom_config)).resolve()
        if args.custom_config
        else _project_custom_config_path(project_dir)
    )
    custom_specs: list[PanelSpec] | None = None
    custom_station_panels: dict[tuple[str, str], StationPanelData] = {}
    custom_score_points: pd.DataFrame | None = None
    if custom_config_path is not None:
        try:
            custom_specs = _parse_panel_specs(custom_config_path)
            _validate_station_ids(custom_specs, stations_df)
            requested_station_keys = {
                (spec.station_id.lower(), _STATION_PANEL_META[spec.panel]["value_col"])
                for spec in custom_specs
                if spec.station_id is not None and spec.panel in _STATION_PANEL_META
            }
            custom_station_panels = {
                key: _load_station_panel_data(project_dir, key[0], value_col=key[1], stations_df=stations_df)
                for key in sorted(requested_station_keys)
            }
            if _custom_overview_needs_score_points(custom_specs):
                custom_score_points = _load_score_points_for_custom_overview(project_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load custom result overview config {}: {}", custom_config_path, exc)
            return 1

    try:
        if args.custom_config is not None:
            if custom_specs is None:
                raise ValueError(f"Custom config could not be loaded: {custom_config_path}")
            custom_output = args.output if args.output is not None else _default_custom_output(project_dir)
            plot_result_overview(
                scf_obs=scf_obs,
                scf_model=scf_model,
                wet_obs=wet_obs,
                wet_model=wet_model,
                wsl_obs=wsl_obs,
                wsl_model=wsl_model,
                scf_env=scf_env,
                wet_env=wet_env,
                wsl_env=wsl_env,
                wsl_prior_coverage=wsl_prior_coverage,
                output=custom_output,
                assim_events=assim_events,
                mode=str(args.mode or "band"),
                roi_swe_model=roi_swe_model,
                roi_swe_members=roi_swe_members,
                roi_snow_depth_model=roi_snow_depth_model,
                roi_snow_depth_members=roi_snow_depth_members,
                panel_specs=custom_specs,
                station_panels=custom_station_panels,
                ess_panel=ess_panel,
                score_points=custom_score_points,
                strict_panels=True,
                x_bounds=project_time_bounds,
                backend=args.backend,
            )
            logger.info("Wrote custom plot: {}", custom_output)
        else:
            default_output = default_result_overview_output(project_dir, args.output)
            plot_result_overview(
                scf_obs=scf_obs,
                scf_model=scf_model,
                wet_obs=wet_obs,
                wet_model=wet_model,
                wsl_obs=wsl_obs,
                wsl_model=wsl_model,
                scf_env=scf_env,
                wet_env=wet_env,
                wsl_env=wsl_env,
                wsl_prior_coverage=wsl_prior_coverage,
                output=default_output,
                assim_events=assim_events,
                mode=str(args.mode or "band"),
                roi_swe_model=roi_swe_model,
                roi_swe_members=roi_swe_members,
                roi_snow_depth_model=roi_snow_depth_model,
                roi_snow_depth_members=roi_snow_depth_members,
                ess_panel=ess_panel,
                x_bounds=project_time_bounds,
                backend=args.backend,
            )
            logger.info("Wrote plot: {}", default_output)

            if custom_specs is not None:
                custom_output = _default_custom_output(project_dir)
                plot_result_overview(
                    scf_obs=scf_obs,
                    scf_model=scf_model,
                    wet_obs=wet_obs,
                    wet_model=wet_model,
                    wsl_obs=wsl_obs,
                    wsl_model=wsl_model,
                    scf_env=scf_env,
                    wet_env=wet_env,
                    wsl_env=wsl_env,
                    wsl_prior_coverage=wsl_prior_coverage,
                    output=custom_output,
                    assim_events=assim_events,
                    mode=str(args.mode or "band"),
                    roi_swe_model=roi_swe_model,
                    roi_swe_members=roi_swe_members,
                    roi_snow_depth_model=roi_snow_depth_model,
                    roi_snow_depth_members=roi_snow_depth_members,
                    panel_specs=custom_specs,
                    station_panels=custom_station_panels,
                    ess_panel=ess_panel,
                    score_points=custom_score_points,
                    strict_panels=True,
                    x_bounds=project_time_bounds,
                    backend=args.backend,
                )
                logger.info("Wrote custom plot: {}", custom_output)
    except ModuleNotFoundError as exc:
        logger.error("matplotlib is required to plot: {}", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Plotting failed: {}", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
