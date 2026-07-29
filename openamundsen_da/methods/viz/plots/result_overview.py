"""Plot the setup-level result overview for fraction, ROI, and station series."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from string import ascii_lowercase

import pandas as pd
from loguru import logger

from openamundsen_da.benchmark.pipeline.core import load_benchmark_config
from openamundsen_da.methods.viz.plots.benchmark.core import (
    apply_score_tick_labels,
    build_event_skill_plot_data,
    compute_event_skill_plot_positions,
    draw_score_metric_panel,
    score_legend_handles,
    score_legend_handler_map,
    score_metric_ylim,
    score_variable_sort_key,
    score_variable_color,
    variable_label,
)
from openamundsen_da.io.paths import (
    abspath_relative_to,
    find_project_yaml,
    list_member_dirs,
    list_steps_sorted,
    project_fraction_envelope_path,
)
from openamundsen_da.methods.viz.station_meta import load_ensemble_station_table_from_steps
from openamundsen_da.methods.viz.common import (
    PosterRenderStyle,
    force_figure_text_black,
    save_figure_png,
    scaled_module_attributes,
    set_matplotlib_text_black,
    temporary_module_attributes,
)
from openamundsen_da.methods.viz.plots.theme import (
    BAND_ALPHA,
    COLOR_DA_OBS,
    FIGHEIGHT_OVERVIEW_ROW,
    FIGWIDTH_OVERVIEW_PAPER,
    OVERVIEW_AXIS_LABEL_SIZE,
    OVERVIEW_STANDARD_PANEL_HEIGHT_FACTOR,
    OVERVIEW_XTICK_SIZE,
    OVERVIEW_YTICK_SIZE,
    LS_STATION_OBS,
    LW_DA_OBS,
    SIZE_DA_OBS,
    da_variable_style,
)
from openamundsen_da.methods.viz.plots.common import (
    apply_fraction_grid,
    apply_month_interval_axis_labels,
    draw_adaptive_assim_labels,
    draw_assimilation_markers,
    draw_assimilation_vlines,
    format_station_label,
    result_axis_scale,
)
from openamundsen_da.methods.viz.fraction_series import (
    default_result_overview_output,
    load_fraction_series,
    load_member_series,
    load_open_loop_fraction_series,
    load_weighted_member_envelope,
)
from openamundsen_da.methods.viz.plots.assimilation.ess_timeline import (
    ess_axis_ticks,
    ess_title,
    load_setup_ess_series,
    load_setup_ess_threshold,
)
from openamundsen_da.methods.viz.wet_snow_fields import wsl_prior_summary_from_weights_df
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
    envelope: pd.DataFrame | None = None

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


@dataclass
class _ResultOverviewLegendState:
    da_observation: bool = False
    satellite_observation: bool = False
    open_loop: bool = False
    ensemble_summary: bool = False
    station_observation: bool = False
    da_event: bool = False


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
    "fSC": "fSCA",
    "WSF": "WSF",
    "WSLA": "Elevation [m]",
    "roi-swe": "SWE [mm]",
    "roi-sd": "Snow depth [m]",
    "station-sd": "Snow depth [m]",
    "station-swe": "SWE [mm]",
    "ess": "ESS",
    "scores-crpss": "CRPSS",
    "scores-ner": "NER",
    "scores-zskill": "zSkill",
}

_PANEL_TITLE_X = 0.0
_PANEL_TITLE_Y_OFFSET = 0.0
_PANEL_TITLE_Y_OFFSET_WITH_ASSIM_LABELS = 0.007
_PANEL_TITLE_PAD = 2.0
_PANEL_TITLE_PAD_WITH_ASSIM_LABELS = 9.0
_RESULT_OVERVIEW_DATA_LW = 1.35
_RESULT_OVERVIEW_MATCHED_EVENT_LW = 1.35
_RESULT_OVERVIEW_TITLE_SIZE = 8.0
_RESULT_OVERVIEW_LABEL_SIZE = OVERVIEW_AXIS_LABEL_SIZE
_RESULT_OVERVIEW_TICK_SIZE = OVERVIEW_YTICK_SIZE
_RESULT_OVERVIEW_XTICK_SIZE = OVERVIEW_XTICK_SIZE
_RESULT_OVERVIEW_LEGEND_SIZE = 5.4
_RESULT_OVERVIEW_SCORE_LEGEND_SIZE = 4.8
_RESULT_OVERVIEW_FIGURE_LEGEND_SIZE = 8.0
_RESULT_OVERVIEW_SPLIT_FIGURE_LEGEND_SIZE = 6.2
_RESULT_OVERVIEW_DA_LABEL_SIZE = 5.5
_RESULT_OVERVIEW_PANEL_BOX_LW: float | None = None
_RESULT_OVERVIEW_LEGEND_FRAME_ALPHA = 0.74
_RESULT_OVERVIEW_SAVE_PAD_INCHES = 0.015
_RESULT_OVERVIEW_OBS_MARKER_SIZE = 2.8
_RESULT_OVERVIEW_ESS_MARKER_SIZE = 3.2
_SCORE_LEGEND_SPACER_SCALE = 0.25
_ESS_THRESHOLD_COLOR = "black"
_ESS_THRESHOLD_LW = 0.9
_ALTITUDE_MAJOR_TICK_STEPS_M = (100.0, 200.0, 250.0, 500.0, 1000.0, 2000.0)
_ALTITUDE_MAX_MAJOR_INTERVALS = 5.0

_DEFAULT_TITLES = {
    "fSC": "Fractional snow covered area (fSCA)",
    "WSF": "Wet snow fraction",
    "WSLA": "Wet snow line altitude",
    "roi-swe": "Mean SWE",
    "roi-sd": "Mean snow depth",
    "ess": "Effective sample size",
    "scores-crpss": "Continuous ranked probability skill score (CRPSS)",
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
_DA_EVENT_STANDARD_COLOR = "#777777"
_DA_EVENT_MATCHED_COLOR = "#000000"

_STATION_PANEL_EVENT_VARIABLE = {
    "station-sd": "station_hs",
    "station-swe": "station_swe",
}

_ASSIM_LABEL_ROW_OFFSETS_PTS = [0.35, 6.5]
_ASSIM_LABEL_MIN_SPACING_DAYS = 18.0
_PANEL_YLABEL_FONT_SIZE = 8.4


@contextmanager
def _scaled_result_overview_style(style: PosterRenderStyle | float):
    if isinstance(style, (int, float)):
        style = PosterRenderStyle(scale=float(style))
    names = (
        "_RESULT_OVERVIEW_DATA_LW",
        "_RESULT_OVERVIEW_MATCHED_EVENT_LW",
        "_RESULT_OVERVIEW_TITLE_SIZE",
        "_RESULT_OVERVIEW_LABEL_SIZE",
        "_RESULT_OVERVIEW_TICK_SIZE",
        "_RESULT_OVERVIEW_XTICK_SIZE",
        "_RESULT_OVERVIEW_LEGEND_SIZE",
        "_RESULT_OVERVIEW_SCORE_LEGEND_SIZE",
        "_RESULT_OVERVIEW_FIGURE_LEGEND_SIZE",
        "_RESULT_OVERVIEW_SPLIT_FIGURE_LEGEND_SIZE",
        "_RESULT_OVERVIEW_DA_LABEL_SIZE",
        "_RESULT_OVERVIEW_OBS_MARKER_SIZE",
        "_RESULT_OVERVIEW_ESS_MARKER_SIZE",
        "_ESS_THRESHOLD_LW",
        "_PANEL_YLABEL_FONT_SIZE",
        "SIZE_DA_OBS",
        "LW_DA_OBS",
    )
    with scaled_module_attributes(sys.modules[__name__], names, style.scale):
        overrides: dict[str, object] = {}
        if style.typography is not None:
            typography = style.typography
            overrides.update(
                {
                    "_RESULT_OVERVIEW_TITLE_SIZE": typography.title_pt,
                    "_RESULT_OVERVIEW_LABEL_SIZE": typography.label_pt,
                    "_PANEL_YLABEL_FONT_SIZE": typography.label_pt,
                    "_RESULT_OVERVIEW_TICK_SIZE": typography.support_pt,
                    "_RESULT_OVERVIEW_XTICK_SIZE": typography.support_pt,
                    "_RESULT_OVERVIEW_LEGEND_SIZE": typography.support_pt,
                    "_RESULT_OVERVIEW_SCORE_LEGEND_SIZE": typography.support_pt,
                    "_RESULT_OVERVIEW_FIGURE_LEGEND_SIZE": typography.support_pt,
                    "_RESULT_OVERVIEW_SPLIT_FIGURE_LEGEND_SIZE": typography.support_pt,
                    "_RESULT_OVERVIEW_DA_LABEL_SIZE": typography.support_pt,
                }
            )
        if style.linework is not None:
            overrides["_RESULT_OVERVIEW_PANEL_BOX_LW"] = style.linework.panel_box_pt
        if overrides:
            with temporary_module_attributes(sys.modules[__name__], overrides):
                yield
        else:
            yield


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


def _project_panel_config_path(project_dir: Path) -> Path | None:
    candidate = (project_dir / "plots.yml").resolve()
    if not candidate.is_file():
        return None
    return candidate


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


def _configured_overview_needs_score_points(specs: list[PanelSpec] | None) -> bool:
    return any(_is_score_panel(spec.panel) for spec in (specs or []))


def _load_score_points_for_configured_overview(project_dir: Path) -> pd.DataFrame:
    score_path = project_dir / "results" / "benchmark" / "scores" / "event_scores.csv"
    if not score_path.is_file():
        raise FileNotFoundError(
            f"Missing benchmark event scores for configured score panel(s): {score_path}. "
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


def _weighted_envelope_or_none(
    project_dir: Path,
    filename: str,
    value_col: str,
    **kwargs,
) -> pd.DataFrame | None:
    try:
        return load_weighted_member_envelope(project_dir, filename, value_col, **kwargs)
    except FileNotFoundError:
        logger.warning(
            "No PF prior ledger available for {}; using the legacy materialized-member summary",
            filename,
        )
        return None


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
        envelope=_weighted_envelope_or_none(
            project_dir,
            point_name,
            value_col,
            daily_mean=True,
        ),
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
    metric = "Snow depth" if spec.panel == "station-sd" else "SWE"
    alt_text = f" {int(station_data.altitude_m)} m" if station_data.altitude_m is not None else ""
    return f"{metric} ({station_data.display_name}{alt_text})"


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
    center = aligned.mean(axis=1, skipna=True).where(n > 0)
    value_min = aligned.min(axis=1, numeric_only=True).where(n > 0)
    value_max = aligned.max(axis=1, numeric_only=True).where(n > 0)
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
    return _weighted_envelope_or_none(
        project_dir,
        "point_wet_snow_line_roi.csv",
        "wet_snow_line",
        preserve_missing_values=True,
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
            summary = wsl_prior_summary_from_weights_df(df)
            if summary is None:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(date).normalize(),
                    "value_mean": summary["mean"],
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


def _frame_has_finite_value(frame: pd.DataFrame | None, value_col: str) -> bool:
    if frame is None or frame.empty or value_col not in frame.columns:
        return False
    return bool(pd.to_numeric(frame[value_col], errors="coerce").notna().any())


def _series_has_finite_value(series: pd.Series | None) -> bool:
    if series is None or series.empty:
        return False
    return bool(pd.to_numeric(series, errors="coerce").notna().any())


def _frame_has_finite_band(frame: pd.DataFrame | None) -> bool:
    if frame is None or frame.empty:
        return False
    return (
        _frame_has_finite_value(frame, "value_min")
        or _frame_has_finite_value(frame, "value_mean")
        or _frame_has_finite_value(frame, "value_max")
    )


def _has_matching_assimilation_observation(
    dates: list[pd.Timestamp],
    obs: pd.DataFrame | None,
    *,
    value_col: str,
) -> bool:
    if obs is None or obs.empty or not dates or value_col not in obs.columns or "date" not in obs.columns:
        return False
    try:
        target = pd.to_datetime(dates).normalize()
        obs_dates = pd.to_datetime(obs["date"], errors="coerce")
    except Exception:
        return False
    values = pd.to_numeric(obs[value_col], errors="coerce")
    mask = obs_dates.dt.normalize().isin(target).fillna(False) & values.notna()
    return bool(mask.any())


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


def _ensemble_legend_handle(panel_style: dict[str, str]):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return _EnsembleLegendHandle(
        (
            Patch(facecolor=panel_style["fill"], edgecolor="none", linewidth=0.0, alpha=BAND_ALPHA),
            Line2D([0], [0], color=panel_style["line"], lw=_RESULT_OVERVIEW_DATA_LW),
        ),
        "ensemble (with mean)",
    )


def _open_loop_legend_handle():
    from matplotlib.lines import Line2D

    return Line2D([0], [0], color="black", lw=_RESULT_OVERVIEW_DATA_LW, label="open loop")


def _satellite_obs_legend_handle():
    from matplotlib.lines import Line2D

    return Line2D(
        [0],
        [0],
        color=COLOR_DA_OBS,
        marker="o",
        linestyle="none",
        markersize=3.2,
        label="satellite observation",
    )


def _assimilated_obs_legend_handle():
    from matplotlib.lines import Line2D

    return Line2D(
        [0],
        [0],
        color=COLOR_DA_OBS,
        marker="x",
        linestyle="none",
        markersize=5.2,
        markeredgewidth=1.35,
        label="assimilated observation",
    )


def _station_obs_legend_handle():
    from matplotlib.lines import Line2D

    return Line2D(
        [0],
        [0],
        color=COLOR_DA_OBS,
        lw=_RESULT_OVERVIEW_DATA_LW,
        ls=LS_STATION_OBS,
        label="station observation",
    )


def _add_panel_local_legend(ax, handles: list, *, loc: str = "upper left") -> None:
    if not handles:
        return
    legend = ax.legend(
        handles=handles,
        handler_map=_result_overview_legend_handler_map(),
        loc=loc,
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=_RESULT_OVERVIEW_LEGEND_FRAME_ALPHA,
        fontsize=_RESULT_OVERVIEW_LEGEND_SIZE,
        handlelength=1.7,
        handleheight=1.0,
        handletextpad=0.35,
        columnspacing=0.7,
        labelspacing=0.18,
        borderaxespad=0.35,
    )
    legend.get_frame().set_linewidth(0.0)
    legend.set_zorder(50)


def _add_ess_threshold_inline_label(ax, threshold: float) -> None:
    ymin, ymax = ax.get_ylim()
    offset = 0.035 * max(abs(ymax - ymin), 1.0)
    y = max(ymin, float(threshold) - offset)
    ax.text(
        0.02,
        y,
        "ESS threshold",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="top",
        fontsize=_RESULT_OVERVIEW_LEGEND_SIZE,
        color=_ESS_THRESHOLD_COLOR,
        zorder=50,
    )


def _compact_legend_spacer_rows(legend, spacer_indices: tuple[int, ...]) -> None:
    """Reduce selected one-column legend rows without changing other spacing."""
    handle_box = getattr(legend, "_legend_handle_box", None)
    if handle_box is None:
        return
    columns = handle_box.get_children()
    if len(columns) != 1:
        return
    rows = columns[0].get_children()
    texts = legend.get_texts()
    spacer_height = _RESULT_OVERVIEW_SCORE_LEGEND_SIZE * _SCORE_LEGEND_SPACER_SCALE
    for index in spacer_indices:
        if index >= len(rows) or index >= len(texts):
            continue
        row_children = rows[index].get_children()
        if row_children:
            drawing_area = row_children[0]
            drawing_area.height = spacer_height
            drawing_area.ydescent = 0.0
        texts[index].set_fontsize(spacer_height)


def _add_score_panel_legend(ax, variables: list[str]) -> None:
    if not variables:
        return
    from matplotlib.lines import Line2D

    def _score_variable_label(variable: str) -> str:
        if str(variable) == "scf":
            return "fSCA"
        return variable_label(variable)

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=4.2,
            markerfacecolor=score_variable_color(variable),
            markeredgecolor=score_variable_color(variable),
            color=score_variable_color(variable),
            label=_score_variable_label(variable),
        )
        for variable in variables
    ]
    spacer_indices = (len(handles), len(handles) + 3)
    handles.extend(
        [
            Line2D([0], [0], linestyle="none", marker="none", alpha=0.0, label=" "),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=4.2,
                markerfacecolor="#000000",
                markeredgecolor="#000000",
                color="#000000",
                label="posterior",
            ),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor="#000000",
                color="#000000",
                label="prior",
            ),
            Line2D([0], [0], linestyle="none", marker="none", alpha=0.0, label=" "),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor="#000000",
                color="#000000",
                label="assimilation fit",
            ),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="s",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor="#000000",
                color="#000000",
                label="semi-independent",
            ),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="^",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor="#000000",
                color="#000000",
                label="independent",
            ),
        ]
    )
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=_RESULT_OVERVIEW_LEGEND_FRAME_ALPHA,
        fontsize=_RESULT_OVERVIEW_SCORE_LEGEND_SIZE,
        handlelength=1.2,
        handleheight=1.0,
        handletextpad=0.3,
        columnspacing=0.55,
        labelspacing=0.12,
        borderaxespad=0.35,
    )
    _compact_legend_spacer_rows(legend, spacer_indices)
    legend.get_frame().set_linewidth(0.0)
    legend.set_zorder(50)


def _apply_result_axis_text(axes, specs: list[PanelSpec]) -> None:
    for ax, spec in zip(axes, specs, strict=True):
        ax.set_ylabel(_PANEL_YLABELS.get(spec.panel, ""), fontsize=_RESULT_OVERVIEW_LABEL_SIZE, labelpad=2.0)
        ax.tick_params(axis="y", labelsize=_RESULT_OVERVIEW_TICK_SIZE, pad=4.0)
        if _RESULT_OVERVIEW_PANEL_BOX_LW is not None:
            for spine in ax.spines.values():
                spine.set_linewidth(_RESULT_OVERVIEW_PANEL_BOX_LW)
        for label in ax.get_yticklabels():
            label.set_rotation(90)
            label.set_rotation_mode("anchor")
            label.set_ha("center")
            label.set_va("center")


def _add_in_panel_assim_labels(ax, events: list[AssimilationEvent], *, center_of_day: bool = False) -> None:
    if not events:
        return
    import matplotlib.dates as mdates

    dates = _center_assim_event_times(events) if center_of_day else [pd.to_datetime(event.date) for event in events]
    x_min, x_max = ax.get_xlim()
    x_min_dt = pd.Timestamp(mdates.num2date(x_min)).tz_localize(None)
    x_max_dt = pd.Timestamp(mdates.num2date(x_max)).tz_localize(None)
    placed: list[pd.Timestamp] = []
    y_levels = [0.13, 0.24, 0.35]
    span_days = max((x_max_dt - x_min_dt).total_seconds() / 86400.0, 1.0)
    shift = pd.Timedelta(days=min(2.0, max(0.1, span_days * 0.015)))
    edge_guard = pd.Timedelta(days=min(9.0, max(0.2, span_days * 0.12)))
    for idx, date in enumerate(pd.to_datetime(dates), start=1):
        date = pd.Timestamp(date).tz_localize(None)
        if not (x_min_dt <= date <= x_max_dt):
            continue
        close_count = sum(abs((date - previous).days) < 14 for previous in placed)
        y_axes = y_levels[close_count % len(y_levels)]
        placed.append(date)
        if date + edge_guard <= x_max_dt:
            x_date = date + shift
            ha = "left"
        else:
            x_date = max(date - shift, x_min_dt)
            ha = "right"
        ax.text(
            x_date,
            y_axes,
            f"DA {idx}",
            transform=ax.get_xaxis_transform(),
            ha=ha,
            va="center",
            fontsize=_RESULT_OVERVIEW_DA_LABEL_SIZE,
            color="#000000",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": _DA_EVENT_STANDARD_COLOR,
                "linewidth": 0.55,
                "alpha": 0.92,
            },
            zorder=55,
            clip_on=True,
        )


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


def _band_frame(
    member_series: list[pd.Series] | None,
    *,
    q_low: float = 0.0,
    q_high: float = 1.0,
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
        return {"variable_key": variable}
    return {"variable_key": variable, "ls": str(meta["ls"])}


def _assim_labels(events: list[AssimilationEvent]) -> tuple[list[pd.Timestamp], list[str]]:
    dates: list[pd.Timestamp] = []
    labels: list[str] = []
    for idx, event in enumerate(events, start=1):
        dates.append(pd.to_datetime(event.date))
        labels.append(f"DA {idx}")
    return dates, labels


def _center_assim_event_times(events: list[AssimilationEvent]) -> list[pd.Timestamp]:
    """Place day-based result-plot DA markers at midday for visual alignment."""
    return [pd.to_datetime(event.date) + pd.Timedelta(hours=12) for event in events]


def _draw_all_assim(
    ax,
    events: list[AssimilationEvent],
    *,
    center_of_day: bool = False,
    matched_variable: str | None = None,
) -> None:
    draw_dates = _center_assim_event_times(events) if center_of_day else [pd.to_datetime(event.date) for event in events]
    matched_token = str(matched_variable).strip().lower() if matched_variable is not None else None
    for event, draw_date in zip(events, draw_dates):
        meta = _assim_style(event.variable)
        is_matching_panel_event = matched_token is not None and str(meta["variable_key"]).strip().lower() == matched_token
        draw_assimilation_vlines(
            ax,
            [draw_date],
            color=_DA_EVENT_MATCHED_COLOR if is_matching_panel_event else _DA_EVENT_STANDARD_COLOR,
            ls="--",
            lw=_RESULT_OVERVIEW_MATCHED_EVENT_LW if is_matching_panel_event else 1.0,
            alpha=0.95,
            label="_nolegend_",
        )


def _add_assim_label_axis(ax, events: list[AssimilationEvent], idx: int, *, center_of_day: bool = False):
    if not events:
        return None
    import matplotlib.dates as mdates

    if center_of_day:
        dates = _center_assim_event_times(events)
        labels = [f"DA {i}" for i in range(1, len(events) + 1)]
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

    draw_adaptive_assim_labels(
        label_axis,
        [item[0] for item in visible_items],
        labels=[item[1] for item in visible_items],
        avoid_artists=[ax._left_title],
        max_labels=max(1, len(visible_items)),
        y_offset_pts=_ASSIM_LABEL_ROW_OFFSETS_PTS[0],
        rotation=0.0,
        row_y_offsets_pts=_ASSIM_LABEL_ROW_OFFSETS_PTS,
        min_row_spacing_days=_ASSIM_LABEL_MIN_SPACING_DAYS,
        axes_y=1.0,
        ha="center",
        fontsize=_RESULT_OVERVIEW_LEGEND_SIZE,
    )
    if label_axis is not None:
        label_axis.set_zorder(ax.get_zorder() + 1)
    return label_axis


def _build_result_overview_legend_handles(
    *,
    legend_state: _ResultOverviewLegendState,
    show_ess_threshold: bool = False,
) -> list:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    if legend_state.da_observation:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLOR_DA_OBS,
                marker="x",
                linestyle="none",
                markersize=5.6,
                markeredgewidth=1.45,
                label="assimilated observation",
            )
        )
    if legend_state.satellite_observation:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLOR_DA_OBS,
                marker="o",
                linestyle="none",
                markersize=3.2,
                label="satellite observation",
            )
        )
    if legend_state.open_loop:
        handles.append(Line2D([0], [0], color="black", lw=_RESULT_OVERVIEW_DATA_LW, label="open loop"))
    if legend_state.ensemble_summary:
        handles.append(
            _EnsembleLegendHandle(
                (
                    Patch(
                        facecolor="#bfc6cf",
                        edgecolor="#666666",
                        linewidth=0.9,
                        alpha=BAND_ALPHA,
                    ),
                    Line2D([0], [0], color="#666666", lw=_RESULT_OVERVIEW_DATA_LW),
                ),
                "ensemble (with mean)",
            )
        )
    if legend_state.station_observation:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLOR_DA_OBS,
                lw=_RESULT_OVERVIEW_DATA_LW,
                ls=LS_STATION_OBS,
                label="station observation",
            ),
        )
    if legend_state.da_event:
        handles.append(
            Line2D([0], [0], color=_DA_EVENT_STANDARD_COLOR, lw=1.2, ls="--", label="data assimilation event")
        )
    return handles


def _result_overview_legend_handler_map() -> dict[type, object]:
    handler_map = dict(score_legend_handler_map())
    handler_map[_EnsembleLegendHandle] = _EnsembleLegendHandler()
    return handler_map


def _build_result_overview_legends(
    fig,
    *,
    legend_state: _ResultOverviewLegendState,
    score_variables: list[str] | None = None,
    show_ess_threshold: bool = False,
) -> list:
    overview_handles = _build_result_overview_legend_handles(
        legend_state=legend_state,
        show_ess_threshold=show_ess_threshold and not score_variables,
    )

    if not score_variables:
        if not overview_handles:
            return []
        legend = fig.legend(
            handles=overview_handles,
            handler_map=_result_overview_legend_handler_map(),
            loc="lower left",
            bbox_to_anchor=(0.055, 0.018, 0.865, 0.06),
            bbox_transform=fig.transFigure,
            mode="expand",
            ncol=3,
            frameon=False,
            fontsize=_RESULT_OVERVIEW_FIGURE_LEGEND_SIZE,
            handlelength=2.45,
            handleheight=1.25,
            columnspacing=1.1,
            handletextpad=0.45,
            borderaxespad=0.0,
        )
        return [legend]

    score_handles = score_legend_handles(score_variables, include_da_event=False)
    legends = []
    if overview_handles:
        overview_legend = fig.legend(
            handles=overview_handles,
            handler_map=_result_overview_legend_handler_map(),
            loc="lower left",
            bbox_to_anchor=(0.055, 0.054, 0.865, 0.032),
            bbox_transform=fig.transFigure,
            mode="expand",
            ncol=min(4, len(overview_handles)),
            frameon=False,
            fontsize=_RESULT_OVERVIEW_SPLIT_FIGURE_LEGEND_SIZE,
            handlelength=2.4,
            handleheight=1.22,
            columnspacing=0.8,
            handletextpad=0.32,
            borderaxespad=0.0,
        )
        legends.append(overview_legend)
    score_legend = fig.legend(
        handles=score_handles,
        handler_map=_result_overview_legend_handler_map(),
        loc="lower left",
        bbox_to_anchor=(0.055, 0.018, 0.865, 0.032),
        bbox_transform=fig.transFigure,
        mode="expand",
        ncol=5,
        frameon=False,
        fontsize=_RESULT_OVERVIEW_SPLIT_FIGURE_LEGEND_SIZE,
        handlelength=2.4,
        handleheight=1.22,
        columnspacing=0.8,
        handletextpad=0.62,
        borderaxespad=0.0,
    )
    legends.append(score_legend)
    if overview_handles:
        fig.canvas.draw()
        score_bbox = score_legend.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted()
        )
        overview_legend.set_bbox_to_anchor((0.055, float(score_bbox.y1) + 0.008, 0.865, 0.032), transform=fig.transFigure)
    return legends


def _legend_band_bottom(fig, legends: list, *, gap: float = 0.008, minimum: float = 0.04) -> float:
    if not legends:
        return minimum
    renderer = fig.canvas.get_renderer()
    top = 0.0
    for legend in legends:
        bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        top = max(top, float(bbox.y1))
    return max(minimum, top + gap)


def _altitude_major_tick_step(ymin: float, ymax: float) -> float | None:
    span = abs(float(ymax) - float(ymin))
    if span <= 0.0:
        return None
    for step in _ALTITUDE_MAJOR_TICK_STEPS_M:
        if span / step <= _ALTITUDE_MAX_MAJOR_INTERVALS:
            return step
    return _ALTITUDE_MAJOR_TICK_STEPS_M[-1]


def _apply_altitude_y_ticks(ax) -> None:
    from matplotlib.ticker import MultipleLocator

    step = _altitude_major_tick_step(*ax.get_ylim())
    if step is None:
        return
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2.0))


def _label_lower_clean_1000m_y_ticks(ax) -> None:
    from matplotlib.ticker import FuncFormatter

    ymin, ymax = sorted(float(value) for value in ax.get_ylim())
    tol = max(abs(ymax - ymin), 1.0) * 1e-9
    ticks = [float(tick) for tick in ax.get_yticks() if ymin - tol <= float(tick) <= ymax + tol]
    if not ticks:
        return
    candidates = [
        tick
        for tick in ticks
        if abs(tick % 1000.0) <= tol or abs((tick % 1000.0) - 1000.0) <= tol
    ]
    if len(candidates) > 2:
        candidates = candidates[:2]
    selected = set(candidates)

    def _format_tick(value: float, _pos: int) -> str:
        return f"{value:g}" if any(abs(float(value) - tick) <= tol for tick in selected) else ""

    ax.yaxis.set_major_formatter(FuncFormatter(_format_tick))


def _label_every_second_dense_y_ticks_from_bottom(ax, *, max_visible_labels: int = 4) -> None:
    from matplotlib.ticker import FuncFormatter

    if max_visible_labels < 2:
        raise ValueError("max_visible_labels must be at least 2")
    ymin, ymax = sorted(float(value) for value in ax.get_ylim())
    tol = max(abs(ymax - ymin), 1.0) * 1e-9
    ticks = [float(tick) for tick in ax.get_yticks() if ymin - tol <= float(tick) <= ymax + tol]
    if len(ticks) <= max_visible_labels:
        selected_ticks = ticks
    else:
        selected_ticks = ticks[0::2][:max_visible_labels]
    if len(selected_ticks) > 2 and abs(selected_ticks[-1] - ymax) <= tol:
        selected_ticks = selected_ticks[:-1]
    selected = set(selected_ticks)

    def _format_tick(value: float, _pos: int) -> str:
        if not any(abs(float(value) - tick) <= tol for tick in selected):
            return ""
        return f"{value:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(_format_tick))


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
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], labels=["", "", "0.5", "", "1"])


def _apply_ess_ticks(ax, ensemble_size: int | None, *, threshold: float | None = None) -> None:
    ticks = ess_axis_ticks(ensemble_size, threshold=threshold)
    if ticks:
        ax.set_yticks(ticks)


def _add_ess_threshold_legend(ax) -> None:
    from matplotlib.lines import Line2D

    legend = ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=_ESS_THRESHOLD_COLOR,
                lw=_ESS_THRESHOLD_LW,
                ls="--",
                label="ESS threshold",
            )
        ],
        loc="upper right",
        frameon=False,
        fontsize=_RESULT_OVERVIEW_SPLIT_FIGURE_LEGEND_SIZE,
        handlelength=1.8,
        handletextpad=0.35,
        labelspacing=0.2,
        borderpad=0.0,
        borderaxespad=0.35,
    )
    legend.set_zorder(40)


def _align_panel_titles_to_axes(title_artists: list[tuple[object, object, bool]]) -> None:
    for _ax, title_artist, has_top_assim_labels in title_artists:
        title_artist.set_x(_PANEL_TITLE_X)
        title_offset = _PANEL_TITLE_Y_OFFSET_WITH_ASSIM_LABELS if has_top_assim_labels else _PANEL_TITLE_Y_OFFSET
        title_artist.set_y(1.0 + title_offset)


def _hide_title_overlapping_y_tick_labels(fig, title_artists: list[tuple[object, object, bool]]) -> None:
    renderer = fig.canvas.get_renderer()
    for ax, title_artist, _has_top_assim_labels in title_artists:
        title_bbox = title_artist.get_window_extent(renderer)
        for tick in ax.yaxis.get_major_ticks():
            for label in (tick.label1, tick.label2):
                if label.get_text() and label.get_visible() and title_bbox.overlaps(label.get_window_extent(renderer)):
                    label.set_visible(False)


def _thin_final_result_y_tick_labels(axes, specs: list[PanelSpec]) -> None:
    for ax, spec in zip(axes, specs, strict=True):
        if spec.panel == "roi-swe":
            _label_every_second_dense_y_ticks_from_bottom(ax, max_visible_labels=4)


def _apply_time_axis_labels(
    axes,
    x_bounds: tuple[pd.Timestamp, pd.Timestamp] | None,
    *,
    align_first_xtick_left: bool = False,
) -> None:
    apply_month_interval_axis_labels(
        axes,
        x_bounds,
        labelsize=_RESULT_OVERVIEW_XTICK_SIZE,
        align_first_label_left=align_first_xtick_left,
    )


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
    roi_swe_env: pd.DataFrame | None = None,
    roi_snow_depth_model: pd.DataFrame | None = None,
    roi_snow_depth_members: list[pd.Series] | None = None,
    roi_snow_depth_env: pd.DataFrame | None = None,
    panel_specs: list[PanelSpec] | None = None,
    station_panels: dict[tuple[str, str], StationPanelData] | None = None,
    ess_panel: EssPanelData | None = None,
    score_points: pd.DataFrame | None = None,
    strict_panels: bool = False,
    x_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    backend: str = "Agg",
    target_size_in: tuple[float, float] | None = None,
    style_scale: float = 1.0,
    poster_style: PosterRenderStyle | None = None,
    layout_h_pad: float = 0.32,
    layout_hspace: float | None = None,
    panel_height_factor: float = 1.0,
    align_first_xtick_left: bool = False,
) -> None:
    """Render the result overview into one PNG."""
    if panel_height_factor <= 0.0:
        raise ValueError("panel_height_factor must be > 0")
    style = poster_style or PosterRenderStyle(scale=style_scale)
    if style.scale != 1.0 or style.typography is not None or style.linework is not None:
        with _scaled_result_overview_style(style):
            return plot_result_overview(
                scf_obs=scf_obs,
                scf_model=scf_model,
                wet_obs=wet_obs,
                wet_model=wet_model,
                scf_env=scf_env,
                wet_env=wet_env,
                output=output,
                wsl_obs=wsl_obs,
                wsl_model=wsl_model,
                wsl_env=wsl_env,
                wsl_prior_coverage=wsl_prior_coverage,
                assim_events=assim_events,
                mode=mode,
                roi_swe_model=roi_swe_model,
                roi_swe_members=roi_swe_members,
                roi_swe_env=roi_swe_env,
                roi_snow_depth_model=roi_snow_depth_model,
                roi_snow_depth_members=roi_snow_depth_members,
                roi_snow_depth_env=roi_snow_depth_env,
                panel_specs=panel_specs,
                station_panels=station_panels,
                ess_panel=ess_panel,
                score_points=score_points,
                strict_panels=strict_panels,
                x_bounds=x_bounds,
                backend=backend,
                target_size_in=target_size_in,
                style_scale=1.0,
                poster_style=PosterRenderStyle(),
                layout_h_pad=layout_h_pad,
                layout_hspace=layout_hspace,
                panel_height_factor=panel_height_factor,
                align_first_xtick_left=align_first_xtick_left,
            )

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

    height_ratios = [OVERVIEW_STANDARD_PANEL_HEIGHT_FACTOR * panel_height_factor for _spec in specs]
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
    title_artists: list[tuple[object, object, bool]] = []

    roi_swe_env = roi_swe_env if roi_swe_env is not None else _band_frame(roi_swe_members)
    roi_snow_depth_env = roi_snow_depth_env if roi_snow_depth_env is not None else _band_frame(roi_snow_depth_members)
    shared_scales = _shared_result_scales(
        specs,
        roi_swe_model=roi_swe_model,
        roi_swe_env=roi_swe_env,
        roi_snow_depth_model=roi_snow_depth_model,
        roi_snow_depth_env=roi_snow_depth_env,
        station_panels=station_panels,
    )
    data_x_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None
    legend_state = _ResultOverviewLegendState()

    for idx, (ax, spec) in enumerate(zip(axes, specs)):
        letter = ascii_lowercase[idx] if idx < len(ascii_lowercase) else str(idx + 1)
        station_data: StationPanelData | None = None
        current_ess_panel = ess_panel if spec.panel == "ess" else None
        if spec.panel.startswith("station-") and spec.station_id is not None:
            value_col = _STATION_PANEL_META[spec.panel]["value_col"]
            station_data = station_panels[(spec.station_id.lower(), value_col)]
        panel_style = None if _is_score_panel(spec.panel) else _panel_style(spec.panel)

        has_top_assim_labels = idx == 0 and bool(events)
        title_artist = ax.set_title(
            f"({letter}) {(_ess_panel_title(spec, current_ess_panel) if spec.panel == 'ess' else _panel_title(spec, station_data))}",
            loc="left",
            fontsize=_RESULT_OVERVIEW_TITLE_SIZE,
            pad=_PANEL_TITLE_PAD_WITH_ASSIM_LABELS if has_top_assim_labels else _PANEL_TITLE_PAD,
        )
        title_artists.append((ax, title_artist, has_top_assim_labels))
        center_assim = spec.panel in {"roi-swe", "roi-sd", "station-swe", "station-sd"}
        if not _is_score_panel(spec.panel):
            matched_variable = _PANEL_VARIABLE_KEYS.get(spec.panel) if spec.panel != "ess" else None
            _draw_all_assim(ax, events, center_of_day=center_assim, matched_variable=matched_variable)
            legend_state.da_event = legend_state.da_event or bool(events)

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
            _add_score_panel_legend(ax, score_variables)
            legend_state.da_event = legend_state.da_event or bool(events)
            ax.set_ylabel("")
            ax.set_ylim(*score_metric_ylim(metric_points, score_metric))
            apply_score_tick_labels(ax)
            bounds = _date_bounds_frames(pd.DataFrame({"date": pd.to_datetime(metric_points["assimilation_date"])}))
        elif spec.panel == "fSC":
            scf_obs_points = _finite_value_points(scf_obs, "scf")
            local_has_ensemble = False
            local_has_open_loop = False
            local_has_satellite_obs = False
            local_has_assimilated_obs = False
            if mode == "band" and _frame_has_finite_band(scf_env):
                legend_state.ensemble_summary = True
                local_has_ensemble = True
                ax.fill_between(
                    scf_env["date"],
                    scf_env["value_min"],
                    scf_env["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    edgecolor="none",
                    linewidth=0.0,
                    label="_nolegend_",
                )
                ax.plot(
                    scf_env["date"],
                    scf_env["value_mean"],
                    "-",
                    color=panel_style["line"],
                    lw=_RESULT_OVERVIEW_DATA_LW,
                    alpha=0.9,
                    label="_nolegend_",
                )
            if _frame_has_finite_value(scf_model, "scf"):
                legend_state.open_loop = True
                local_has_open_loop = True
                ax.plot(scf_model["date"], scf_model["scf"], "-", color="black", lw=_RESULT_OVERVIEW_DATA_LW, label="_nolegend_")
            if spec.show_obs and scf_obs_points is not None and not scf_obs_points.empty:
                legend_state.satellite_observation = True
                local_has_satellite_obs = True
                ax.plot(
                    scf_obs_points["date"],
                    scf_obs_points["scf"],
                    linestyle="none",
                    marker="o",
                    ms=_RESULT_OVERVIEW_OBS_MARKER_SIZE,
                    color=COLOR_DA_OBS,
                    label="_nolegend_",
                )
                scf_dates = [pd.to_datetime(ev.date) for ev in events if ev.variable == "scf"]
                if scf_dates:
                    local_has_assimilated_obs = _has_matching_assimilation_observation(
                        scf_dates,
                        scf_obs_points,
                        value_col="scf",
                    )
                    legend_state.da_observation = legend_state.da_observation or local_has_assimilated_obs
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
            ax.set_ylabel("")
            ax.set_ylim(0, 1)
            apply_fraction_grid(ax, y_step=0.25)
            _apply_fraction_ticks(ax)
            _add_panel_local_legend(
                ax,
                [
                    *([_satellite_obs_legend_handle()] if local_has_satellite_obs else []),
                    *([_assimilated_obs_legend_handle()] if local_has_assimilated_obs else []),
                    *([_open_loop_legend_handle()] if local_has_open_loop else []),
                    *([_ensemble_legend_handle(panel_style)] if local_has_ensemble else []),
                ],
                loc="upper left",
            )
            bounds = _date_bounds_frames(scf_obs, scf_model, scf_env)
        elif spec.panel == "WSF":
            wet_obs_points = _finite_value_points(wet_obs, "wet_snow_fraction")
            local_has_ensemble = False
            local_has_open_loop = False
            local_has_satellite_obs = False
            local_has_assimilated_obs = False
            if mode == "band" and _frame_has_finite_band(wet_env):
                legend_state.ensemble_summary = True
                local_has_ensemble = True
                ax.fill_between(
                    wet_env["date"],
                    wet_env["value_min"],
                    wet_env["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    edgecolor="none",
                    linewidth=0.0,
                    label="_nolegend_",
                )
                ax.plot(
                    wet_env["date"],
                    wet_env["value_mean"],
                    "-",
                    color=panel_style["line"],
                    lw=_RESULT_OVERVIEW_DATA_LW,
                    alpha=0.9,
                    label="_nolegend_",
                )
            if _frame_has_finite_value(wet_model, "wet_snow_fraction"):
                legend_state.open_loop = True
                local_has_open_loop = True
                ax.plot(
                    wet_model["date"],
                    wet_model["wet_snow_fraction"],
                    "-",
                    color="black",
                    lw=_RESULT_OVERVIEW_DATA_LW,
                    label="_nolegend_",
                )
            if spec.show_obs and wet_obs_points is not None and not wet_obs_points.empty:
                legend_state.satellite_observation = True
                local_has_satellite_obs = True
                ax.plot(
                    wet_obs_points["date"],
                    wet_obs_points["wet_snow_fraction"],
                    linestyle="none",
                    marker="o",
                    ms=_RESULT_OVERVIEW_OBS_MARKER_SIZE,
                    color=COLOR_DA_OBS,
                    label="_nolegend_",
                )
                wet_dates = [pd.to_datetime(ev.date) for ev in events if ev.variable == "wet_snow"]
                if wet_dates:
                    local_has_assimilated_obs = _has_matching_assimilation_observation(
                        wet_dates,
                        wet_obs_points,
                        value_col="wet_snow_fraction",
                    )
                    legend_state.da_observation = legend_state.da_observation or local_has_assimilated_obs
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
            ax.set_ylabel("")
            ax.set_ylim(0, 1)
            apply_fraction_grid(ax, y_step=0.25)
            _apply_fraction_ticks(ax)
            _add_panel_local_legend(
                ax,
                [
                    *([_satellite_obs_legend_handle()] if local_has_satellite_obs else []),
                    *([_assimilated_obs_legend_handle()] if local_has_assimilated_obs else []),
                    *([_open_loop_legend_handle()] if local_has_open_loop else []),
                    *([_ensemble_legend_handle(panel_style)] if local_has_ensemble else []),
                ],
                loc="upper left",
            )
            bounds = _date_bounds_frames(wet_obs, wet_model, wet_env)
        elif spec.panel == "WSLA":
            wsl_obs_points = _finite_value_points(wsl_obs, "wet_snow_line")
            local_has_ensemble = False
            local_has_open_loop = False
            local_has_satellite_obs = False
            local_has_assimilated_obs = False
            if mode == "band" and _frame_has_finite_band(wsl_env):
                legend_state.ensemble_summary = True
                local_has_ensemble = True
                ax.fill_between(
                    wsl_env["date"],
                    wsl_env["value_min"],
                    wsl_env["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    edgecolor="none",
                    linewidth=0.0,
                    label="_nolegend_",
                )
                ax.plot(
                    wsl_env["date"],
                    wsl_env["value_mean"],
                    "-",
                    color=panel_style["line"],
                    lw=_RESULT_OVERVIEW_DATA_LW,
                    alpha=0.95,
                    label="_nolegend_",
                )
            if _frame_has_finite_band(wsl_prior_coverage):
                legend_state.ensemble_summary = True
                local_has_ensemble = True
            _draw_wsl_prior_coverage_markers(ax, wsl_prior_coverage, color=panel_style["line"])
            if _frame_has_finite_value(wsl_model, "wet_snow_line"):
                legend_state.open_loop = True
                local_has_open_loop = True
                ax.plot(
                    wsl_model["date"],
                    wsl_model["wet_snow_line"],
                    "-",
                    color="black",
                    lw=_RESULT_OVERVIEW_DATA_LW,
                    label="_nolegend_",
                )
            if spec.show_obs and wsl_obs_points is not None and not wsl_obs_points.empty:
                legend_state.satellite_observation = True
                local_has_satellite_obs = True
                ax.plot(
                    wsl_obs_points["date"],
                    wsl_obs_points["wet_snow_line"],
                    linestyle="none",
                    marker="o",
                    ms=_RESULT_OVERVIEW_OBS_MARKER_SIZE,
                    color=COLOR_DA_OBS,
                    label="_nolegend_",
                )
                wsl_dates = [pd.to_datetime(ev.date) for ev in events if ev.variable == "wet_snow_line"]
                if wsl_dates:
                    local_has_assimilated_obs = _has_matching_assimilation_observation(
                        wsl_dates,
                        wsl_obs_points,
                        value_col="wet_snow_line",
                    )
                    legend_state.da_observation = legend_state.da_observation or local_has_assimilated_obs
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
            ax.set_ylabel("")
            apply_fraction_grid(ax, y_step=None)
            _apply_altitude_y_ticks(ax)
            _label_lower_clean_1000m_y_ticks(ax)
            _add_panel_local_legend(
                ax,
                [
                    *([_open_loop_legend_handle()] if local_has_open_loop else []),
                    *([_ensemble_legend_handle(panel_style)] if local_has_ensemble else []),
                    *([_satellite_obs_legend_handle()] if local_has_satellite_obs else []),
                    *([_assimilated_obs_legend_handle()] if local_has_assimilated_obs else []),
                ],
                loc="lower left",
            )
            bounds = _date_bounds_frames(wsl_obs, wsl_model, wsl_env, wsl_prior_coverage)
        elif spec.panel == "roi-swe":
            if _frame_has_finite_band(roi_swe_env):
                legend_state.ensemble_summary = True
                ax.fill_between(
                    roi_swe_env["date"],
                    roi_swe_env["value_min"],
                    roi_swe_env["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    edgecolor="none",
                    linewidth=0.0,
                    label="_nolegend_",
                )
                ax.plot(
                    roi_swe_env["date"],
                    roi_swe_env["value_mean"],
                    "-",
                    color=panel_style["line"],
                    lw=_RESULT_OVERVIEW_DATA_LW,
                    alpha=0.95,
                    label="_nolegend_",
                )
            if _frame_has_finite_value(roi_swe_model, "swe"):
                legend_state.open_loop = True
                ax.plot(
                    roi_swe_model["date"],
                    roi_swe_model["swe"],
                    "-",
                    color="black",
                    lw=_RESULT_OVERVIEW_DATA_LW,
                    label="_nolegend_",
                )
            ax.set_ylabel("")
            apply_fraction_grid(ax, y_step=None)
            _apply_shared_result_scale(ax, spec.panel, shared_scales)
            _label_every_second_dense_y_ticks_from_bottom(ax, max_visible_labels=4)
            bounds = _date_bounds_frames(roi_swe_model, roi_swe_env)
        elif spec.panel == "roi-sd":
            if _frame_has_finite_band(roi_snow_depth_env):
                legend_state.ensemble_summary = True
                ax.fill_between(
                    roi_snow_depth_env["date"],
                    roi_snow_depth_env["value_min"],
                    roi_snow_depth_env["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    edgecolor="none",
                    linewidth=0.0,
                    label="_nolegend_",
                )
                ax.plot(
                    roi_snow_depth_env["date"],
                    roi_snow_depth_env["value_mean"],
                    "-",
                    color=panel_style["line"],
                    lw=_RESULT_OVERVIEW_DATA_LW,
                    alpha=0.95,
                    label="_nolegend_",
                )
            if _frame_has_finite_value(roi_snow_depth_model, "snow_depth"):
                legend_state.open_loop = True
                ax.plot(
                    roi_snow_depth_model["date"],
                    roi_snow_depth_model["snow_depth"],
                    "-",
                    color="black",
                    lw=_RESULT_OVERVIEW_DATA_LW,
                    label="_nolegend_",
                )
            ax.set_ylabel("")
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
                ms=_RESULT_OVERVIEW_ESS_MARKER_SIZE,
                lw=0.0,
                ls="none",
                color="#000000",
                zorder=25,
            )
            ax.set_ylabel("")
            if current_ess_panel.ensemble_size is not None and current_ess_panel.ensemble_size > 0:
                ax.set_ylim(0.0, float(current_ess_panel.ensemble_size))
                _apply_ess_ticks(
                    ax,
                    current_ess_panel.ensemble_size,
                    threshold=current_ess_panel.threshold,
                )
            if current_ess_panel.threshold is not None:
                ax.axhline(
                    current_ess_panel.threshold,
                    color=_ESS_THRESHOLD_COLOR,
                    lw=_ESS_THRESHOLD_LW,
                    ls="--",
                    zorder=10,
                )
                _add_ess_threshold_inline_label(ax, current_ess_panel.threshold)
            apply_fraction_grid(ax, y_step=None)
            bounds = _date_bounds_frames(ess_series)
        else:
            if station_data is None:
                raise ValueError(f"Missing station panel data for {spec.panel}")
            env_frame = station_data.envelope if station_data.envelope is not None else _band_frame(station_data.members)
            local_has_ensemble = False
            local_has_open_loop = False
            local_has_station_obs = False
            local_has_assimilated_obs = False
            if _frame_has_finite_band(env_frame):
                legend_state.ensemble_summary = True
                local_has_ensemble = True
                ax.fill_between(
                    env_frame["date"],
                    env_frame["value_min"],
                    env_frame["value_max"],
                    color=panel_style["fill"],
                    alpha=BAND_ALPHA,
                    edgecolor="none",
                    linewidth=0.0,
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
                    lw=_RESULT_OVERVIEW_DATA_LW,
                    zorder=4,
                )
            if _series_has_finite_value(station_data.open_loop):
                legend_state.open_loop = True
                local_has_open_loop = True
                ax.plot(
                    station_data.open_loop.index,
                    station_data.open_loop.values,
                    "-",
                    color="black",
                    lw=_RESULT_OVERVIEW_DATA_LW,
                    label="_nolegend_",
                    zorder=5,
                )
            if spec.show_obs and _series_has_finite_value(station_data.obs):
                value_col = _STATION_PANEL_META[spec.panel]["value_col"]
                legend_state.station_observation = True
                local_has_station_obs = True
                ax.plot(
                    station_data.obs.index,
                    station_data.obs.values,
                    LS_STATION_OBS,
                    color=COLOR_DA_OBS,
                    lw=_RESULT_OVERVIEW_DATA_LW,
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
                local_has_assimilated_obs = _has_matching_assimilation_observation(
                    _station_assimilation_dates(events, spec.panel),
                    _station_obs_frame(station_data.obs, value_col=value_col),
                    value_col=value_col,
                )
                legend_state.da_observation = legend_state.da_observation or local_has_assimilated_obs
            ax.set_ylabel("")
            apply_fraction_grid(ax, y_step=None)
            _apply_shared_result_scale(ax, spec.panel, shared_scales)
            _add_panel_local_legend(
                ax,
                [
                    *([_open_loop_legend_handle()] if local_has_open_loop else []),
                    *([_ensemble_legend_handle(panel_style)] if local_has_ensemble else []),
                    *([_station_obs_legend_handle()] if local_has_station_obs else []),
                    *([_assimilated_obs_legend_handle()] if local_has_assimilated_obs else []),
                ],
                loc="upper left",
            )
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
    if events and len(axes) > 0:
        center_assim = specs[0].panel in {"roi-swe", "roi-sd", "station-swe", "station-sd"}
        _add_assim_label_axis(axes[0], events, 0, center_of_day=center_assim)
    _apply_time_axis_labels(axes, effective_x_bounds, align_first_xtick_left=align_first_xtick_left)

    axes[-1].set_xlabel("")
    fig.tight_layout(rect=(0.0, 0.025, 0.985, 1.0), h_pad=layout_h_pad)
    if layout_hspace is not None:
        fig.subplots_adjust(hspace=layout_hspace)
    fig.align_ylabels(axes)
    fig.canvas.draw()
    _align_panel_titles_to_axes(title_artists)
    del legend_state
    fig.tight_layout(rect=(0.0, 0.025, 0.985, 1.0), h_pad=layout_h_pad)
    if layout_hspace is not None:
        fig.subplots_adjust(hspace=layout_hspace)
    fig.align_ylabels(axes)
    fig.canvas.draw()
    _align_panel_titles_to_axes(title_artists)
    fig.canvas.draw()
    _thin_final_result_y_tick_labels(axes, specs)
    _apply_result_axis_text(axes, specs)
    _hide_title_overlapping_y_tick_labels(fig, title_artists)
    fig.canvas.draw()
    force_figure_text_black(fig, axes)
    save_figure_png(
        fig,
        output,
        bbox_inches="tight",
        pad_inches=_RESULT_OVERVIEW_SAVE_PAD_INCHES,
        target_size_in=target_size_in,
    )
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
    parser.add_argument("--custom-config", type=Path, help="Panel YAML (default: <project-dir>/plots.yml)")
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
    roi_swe_env = _weighted_envelope_or_none(project_dir, "point_swe_roi.csv", "swe")
    roi_snow_depth_model = load_open_loop_fraction_series(project_dir, "point_snow_depth_roi.csv", "snow_depth")
    roi_snow_depth_members = load_member_series(project_dir, "point_snow_depth_roi.csv", "snow_depth")
    roi_snow_depth_env = _weighted_envelope_or_none(
        project_dir,
        "point_snow_depth_roi.csv",
        "snow_depth",
    )
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

    try:
        assim_events = load_assimilation_events(project_dir)
    except (FileNotFoundError, ValueError):
        assim_events = []
    event_variables = {event.variable for event in assim_events}

    if scf_obs is None or scf_obs.empty:
        logger.warning("SCF obs not found at {} - plotting without obs points", scf_obs_path)
    if "wet_snow" in event_variables and (wet_obs is None or wet_obs.empty):
        logger.warning("Wet-snow obs not found at {} - plotting without obs points", wet_obs_path)
    if "wet_snow_line" in event_variables and (wsl_obs is None or wsl_obs.empty):
        logger.warning("Wet-snow-line obs not found at {} - plotting without obs points", wsl_obs_path)

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

    panel_config_path = (
        Path(abspath_relative_to(project_dir, args.custom_config)).resolve()
        if args.custom_config
        else _project_panel_config_path(project_dir)
    )
    configured_specs: list[PanelSpec] | None = None
    configured_station_panels: dict[tuple[str, str], StationPanelData] = {}
    configured_score_points: pd.DataFrame | None = None
    if panel_config_path is not None:
        try:
            configured_specs = _parse_panel_specs(panel_config_path)
            _validate_station_ids(configured_specs, stations_df)
            requested_station_keys = {
                (spec.station_id.lower(), _STATION_PANEL_META[spec.panel]["value_col"])
                for spec in configured_specs
                if spec.station_id is not None and spec.panel in _STATION_PANEL_META
            }
            configured_station_panels = {
                key: _load_station_panel_data(project_dir, key[0], value_col=key[1], stations_df=stations_df)
                for key in sorted(requested_station_keys)
            }
            if _configured_overview_needs_score_points(configured_specs):
                configured_score_points = _load_score_points_for_configured_overview(project_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load result overview config {}: {}", panel_config_path, exc)
            return 1

    try:
        output = default_result_overview_output(project_dir, args.output)
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
            output=output,
            assim_events=assim_events,
            mode=str(args.mode or "band"),
            roi_swe_model=roi_swe_model,
            roi_swe_members=roi_swe_members,
            roi_swe_env=roi_swe_env,
            roi_snow_depth_model=roi_snow_depth_model,
            roi_snow_depth_members=roi_snow_depth_members,
            roi_snow_depth_env=roi_snow_depth_env,
            panel_specs=configured_specs,
            station_panels=configured_station_panels,
            ess_panel=ess_panel,
            score_points=configured_score_points,
            strict_panels=configured_specs is not None,
            x_bounds=project_time_bounds,
            backend=args.backend,
        )
        logger.info("Wrote plot: {}", output)
    except ModuleNotFoundError as exc:
        logger.error("matplotlib is required to plot: {}", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Plotting failed: {}", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
