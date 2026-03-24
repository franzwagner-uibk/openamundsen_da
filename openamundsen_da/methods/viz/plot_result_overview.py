"""Plot the setup-level result overview for fraction, ROI, and station series."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from string import ascii_lowercase

import pandas as pd
from loguru import logger

from openamundsen_da.io.paths import (
    abspath_relative_to,
    find_project_yaml,
    list_member_dirs,
    list_steps_sorted,
)
from openamundsen_da.methods.viz._ensemble_meta import load_stations_table_from_steps
from openamundsen_da.methods.viz._style import (
    COLOR_DA_OBS,
    LW_MEMBER,
    LW_MEAN,
    LW_OPEN,
    SIZE_DA_OBS,
    LW_DA_OBS,
)
from openamundsen_da.methods.viz._utils import (
    apply_fraction_grid,
    draw_assim_labels,
    draw_assimilation_markers,
    draw_assimilation_vlines,
    format_station_label,
)
from openamundsen_da.methods.viz.fraction_series import (
    default_fraction_obs_path,
    default_result_overview_output,
    load_fraction_series,
    load_member_series,
    load_open_loop_fraction_series,
)
from openamundsen_da.observer.plot_scf_summary import _load_summary as _load_scf_obs
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


_PANEL_ALIASES = {
    "fsc": "fSC",
    "fws": "fWS",
    "roi-swe": "roi-swe",
    "roi-sd": "roi-sd",
    "station-sd": "station-sd",
    "station-swe": "station-swe",
}

_DEFAULT_PANELS = [
    PanelSpec(panel="fSC"),
    PanelSpec(panel="fWS"),
    PanelSpec(panel="roi-swe"),
    PanelSpec(panel="roi-sd"),
]

_PANEL_YLABELS = {
    "fSC": "snow cover fraction",
    "fWS": "wet snow fraction",
    "roi-swe": "swe [mm]",
    "roi-sd": "snow depth [m]",
    "station-sd": "snow depth [m]",
    "station-swe": "swe [mm]",
}

_DEFAULT_TITLES = {
    "fSC": "snow cover fraction (roi) - openAMUNDSEN ensemble and satellite observations",
    "fWS": "wet snow fraction (roi) - openAMUNDSEN ensemble and satellite observations",
    "roi-swe": "mean swe (roi) - openAMUNDSEN ensemble and open loop",
    "roi-sd": "mean snow depth (roi) - openAMUNDSEN ensemble and open loop",
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

_VARIABLE_STYLES = {
    "fSC": {"fill": "#9ec5ff", "line": "#2f6fb5"},
    "fWS": {"fill": "#9bd8bf", "line": "#2c8a64"},
    "SWE": {"fill": "#ccb8f2", "line": "#7a58b5"},
    "SD": {"fill": "#f3c38e", "line": "#cf7a20"},
}

_PANEL_VARIABLE_KEYS = {
    "fSC": "fSC",
    "fWS": "fWS",
    "roi-swe": "SWE",
    "roi-sd": "SD",
    "station-swe": "SWE",
    "station-sd": "SD",
}

_ASSIM_STYLES = {
    "scf": {"variable_key": "fSC", "ls": "--"},
    "wet_snow": {"variable_key": "fWS", "ls": "--"},
    "station_hs": {"variable_key": "SD", "ls": "--"},
    "station_swe": {"variable_key": "SWE", "ls": "--"},
}

_ASSIM_LABEL_ROW_OFFSETS_PTS = [2.0, 8.0]
_ASSIM_LABEL_MIN_SPACING_DAYS = 18.0
def _load_scf_obs_series(path: Path) -> pd.DataFrame | None:
    """Load SCF summary data, falling back to a generic fraction-series reader."""
    try:
        return _load_scf_obs(path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.debug("Falling back to generic SCF summary reader for {}: {}", path, exc)
        return load_fraction_series(path, "scf")


def _normalize_panel_name(raw: object) -> str:
    key = str(raw or "").strip().lower()
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
    cfg = read_yaml_mapping(config_path, error_cls=RuntimeError, context="Custom result overview config")
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
    candidate = (project_dir / "result_overview_custom.yml").resolve()
    if not candidate.is_file():
        return None
    return candidate


def _default_custom_output(project_dir: Path) -> Path:
    out_dir = project_dir / "plots" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "result_overview_custom.png"


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
    return load_stations_table_from_steps(steps, "prior")


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
    if spec.panel in _DEFAULT_TITLES:
        return _DEFAULT_TITLES[spec.panel]
    if station_data is None:
        raise ValueError(f"Missing station metadata for panel {spec.panel}")
    return _station_title(spec, station_data)


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
    return _VARIABLE_STYLES[_PANEL_VARIABLE_KEYS[panel]]


def _assim_style(variable: str) -> dict[str, str]:
    meta = _ASSIM_STYLES.get(variable)
    if meta is None:
        return {"variable_key": variable, "color": "#777777", "ls": "--"}
    variable_key = str(meta["variable_key"])
    style = _VARIABLE_STYLES[variable_key]
    return {"variable_key": variable_key, "color": style["line"], "ls": str(meta["ls"])}


def _assim_labels(events: list[AssimilationEvent]) -> tuple[list[pd.Timestamp], list[str]]:
    dates: list[pd.Timestamp] = []
    labels: list[str] = []
    for idx, event in enumerate(events, start=1):
        dates.append(pd.to_datetime(event.date))
        labels.append(str(idx))
    return dates, labels


def _draw_all_assim(ax, events: list[AssimilationEvent]) -> None:
    for event in events:
        meta = _assim_style(event.variable)
        draw_assimilation_vlines(
            ax,
            [pd.to_datetime(event.date)],
            color=str(meta["color"]),
            ls=str(meta["ls"]),
            lw=1.2,
            alpha=0.95,
            label="_nolegend_",
        )


def _add_assim_label_axis(ax, events: list[AssimilationEvent], idx: int):
    import matplotlib.dates as mdates

    if not events:
        return None

    x_min, x_max = sorted(ax.get_xlim())
    visible_start = pd.Timestamp(mdates.num2date(x_min)).tz_localize(None)
    visible_end = pd.Timestamp(mdates.num2date(x_max)).tz_localize(None)
    dates, labels = _assim_labels(events)
    visible_items = [
        (date, label)
        for date, label in zip(dates, labels)
        if visible_start <= date <= visible_end
    ]
    if not visible_items:
        return None

    label_axis = ax.twiny()
    label_axis.set_label(f"assimilation_label_axis_{idx}")
    label_axis.patch.set_alpha(0.0)
    label_axis.set_zorder(ax.get_zorder() + 1)
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
        fontsize=6.0,
        color="#000000",
        rotation=0.0,
        va="bottom",
        row_y_offsets_pts=_ASSIM_LABEL_ROW_OFFSETS_PTS,
        min_row_spacing_days=_ASSIM_LABEL_MIN_SPACING_DAYS,
        axes_y=1.0,
        ha="center",
        x_offset_pts=0.0,
    )
    return label_axis


def _build_result_overview_legend(fig) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_DA_OBS,
            marker="x",
            linestyle="none",
            markersize=6.2,
            markeredgewidth=1.6,
            label="satellite observation used in DA",
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
        Line2D([0], [0], color="#666666", lw=LW_MEAN, label="ensemble mean"),
        Line2D([0], [0], color=COLOR_DA_OBS, lw=LW_DA_OBS, label="station observation"),
        Line2D([0], [0], color="#666666", lw=1.2, ls="--", label="DA event"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.008, 0.865, 0.06),
        bbox_transform=fig.transFigure,
        mode="expand",
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=1.6,
        columnspacing=1.1,
        handletextpad=0.45,
        borderaxespad=0.0,
    )


def _apply_result_y_ticks(ax, panel: str) -> None:
    import math
    from matplotlib.ticker import MultipleLocator

    data_max = max(0.0, float(getattr(ax.dataLim, "ymax", 0.0) or 0.0))
    if panel in {"roi-swe", "station-swe"}:
        step_options = [50.0, 100.0]
    elif panel in {"roi-sd", "station-sd"}:
        step_options = [0.25, 0.5, 1.0]
    else:
        return

    step = next((candidate for candidate in step_options if data_max <= candidate * 4.0), step_options[-1])
    upper = step * 4.0 if data_max <= step * 4.0 else math.ceil(data_max / step) * step
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2.0))


def _apply_fraction_ticks(ax) -> None:
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])


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
    scf_env: pd.DataFrame | None,
    wet_env: pd.DataFrame | None,
    roi_swe_model: pd.DataFrame | None,
    roi_swe_members: list[pd.Series] | None,
    roi_snow_depth_model: pd.DataFrame | None,
    roi_snow_depth_members: list[pd.Series] | None,
    station_panels: dict[tuple[str, str], StationPanelData],
) -> bool:
    if spec.panel == "fSC":
        return any(frame is not None and not frame.empty for frame in (scf_obs, scf_model, scf_env))
    if spec.panel == "fWS":
        return any(frame is not None and not frame.empty for frame in (wet_obs, wet_model, wet_env))
    if spec.panel == "roi-swe":
        return (roi_swe_model is not None and not roi_swe_model.empty) or bool(roi_swe_members)
    if spec.panel == "roi-sd":
        return (roi_snow_depth_model is not None and not roi_snow_depth_model.empty) or bool(roi_snow_depth_members)
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
    scf_env: pd.DataFrame | None,
    wet_env: pd.DataFrame | None,
    roi_swe_model: pd.DataFrame | None,
    roi_swe_members: list[pd.Series] | None,
    roi_snow_depth_model: pd.DataFrame | None,
    roi_snow_depth_members: list[pd.Series] | None,
    station_panels: dict[tuple[str, str], StationPanelData],
) -> list[PanelSpec]:
    out: list[PanelSpec] = []
    for spec in specs:
        if _panel_has_data(
            spec,
            scf_obs=scf_obs,
            scf_model=scf_model,
            wet_obs=wet_obs,
            wet_model=wet_model,
            scf_env=scf_env,
            wet_env=wet_env,
            roi_swe_model=roi_swe_model,
            roi_swe_members=roi_swe_members,
            roi_snow_depth_model=roi_snow_depth_model,
            roi_snow_depth_members=roi_snow_depth_members,
            station_panels=station_panels,
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
    assim_events: list[AssimilationEvent] | None = None,
    mode: str = "band",
    roi_swe_model: pd.DataFrame | None = None,
    roi_swe_members: list[pd.Series] | None = None,
    roi_snow_depth_model: pd.DataFrame | None = None,
    roi_snow_depth_members: list[pd.Series] | None = None,
    panel_specs: list[PanelSpec] | None = None,
    station_panels: dict[tuple[str, str], StationPanelData] | None = None,
    strict_panels: bool = False,
    x_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> None:
    """Render the result overview into one PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = panel_specs or list(_DEFAULT_PANELS)
    station_panels = station_panels or {}
    mode = (mode or "band").lower()
    if mode not in {"band", "members"}:
        mode = "band"

    specs = _filter_panel_specs(
        specs,
        strict=strict_panels,
        scf_obs=scf_obs,
        scf_model=scf_model,
        wet_obs=wet_obs,
        wet_model=wet_model,
        scf_env=scf_env,
        wet_env=wet_env,
        roi_swe_model=roi_swe_model,
        roi_swe_members=roi_swe_members,
        roi_snow_depth_model=roi_snow_depth_model,
        roi_snow_depth_members=roi_snow_depth_members,
        station_panels=station_panels,
    )
    if not specs:
        raise ValueError("No data available to plot.")

    fig, axes = plt.subplots(len(specs), 1, figsize=(7.2876875, 1.71236835 * len(specs)), sharex=True)
    if len(specs) == 1:
        axes = [axes]
    title_artists: list[tuple[object, object]] = []

    events = list(assim_events or [])
    roi_swe_env = _band_frame(roi_swe_members)
    roi_snow_depth_env = _band_frame(roi_snow_depth_members)
    data_x_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None
    label_axes: list[tuple[object, object]] = []

    for idx, (ax, spec) in enumerate(zip(axes, specs)):
        letter = ascii_lowercase[idx] if idx < len(ascii_lowercase) else str(idx + 1)
        station_data: StationPanelData | None = None
        if spec.panel.startswith("station-") and spec.station_id is not None:
            value_col = _STATION_PANEL_META[spec.panel]["value_col"]
            station_data = station_panels[(spec.station_id.lower(), value_col)]
        panel_style = _panel_style(spec.panel)

        title_artist = ax.set_title(
            f"({letter}) {_panel_title(spec, station_data)}",
            loc="left",
            fontsize=9.4,
            pad=16.0 if events else 9.0,
        )
        title_artists.append((ax, title_artist))
        _draw_all_assim(ax, events)

        if spec.panel == "fSC":
            if mode == "band" and scf_env is not None and not scf_env.empty:
                ax.fill_between(
                    scf_env["date"],
                    scf_env["value_min"],
                    scf_env["value_max"],
                    color=panel_style["fill"],
                    alpha=0.6,
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
            if spec.show_obs and scf_obs is not None and not scf_obs.empty:
                ax.plot(
                    scf_obs["date"],
                    scf_obs["scf"],
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
                        obs=scf_obs,
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
        elif spec.panel == "fWS":
            if mode == "band" and wet_env is not None and not wet_env.empty:
                ax.fill_between(
                    wet_env["date"],
                    wet_env["value_min"],
                    wet_env["value_max"],
                    color=panel_style["fill"],
                    alpha=0.6,
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
            if spec.show_obs and wet_obs is not None and not wet_obs.empty:
                ax.plot(
                    wet_obs["date"],
                    wet_obs["wet_snow_fraction"],
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
                        obs=wet_obs,
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
        elif spec.panel == "roi-swe":
            if roi_swe_env is not None and not roi_swe_env.empty:
                ax.fill_between(
                    roi_swe_env["date"],
                    roi_swe_env["value_min"],
                    roi_swe_env["value_max"],
                    color=panel_style["fill"],
                    alpha=0.35,
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
            _apply_result_y_ticks(ax, spec.panel)
            bounds = _date_bounds_frames(roi_swe_model, roi_swe_env)
        elif spec.panel == "roi-sd":
            if roi_snow_depth_env is not None and not roi_snow_depth_env.empty:
                ax.fill_between(
                    roi_snow_depth_env["date"],
                    roi_snow_depth_env["value_min"],
                    roi_snow_depth_env["value_max"],
                    color=panel_style["fill"],
                    alpha=0.35,
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
            _apply_result_y_ticks(ax, spec.panel)
            bounds = _date_bounds_frames(roi_snow_depth_model, roi_snow_depth_env)
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
                    alpha=0.35,
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
                ax.plot(
                    station_data.obs.index,
                    station_data.obs.values,
                    "-",
                    color=COLOR_DA_OBS,
                    lw=LW_DA_OBS,
                    label="_nolegend_",
                    zorder=6,
                )
            ax.set_ylabel(_PANEL_YLABELS[spec.panel], fontsize=8.6)
            apply_fraction_grid(ax, y_step=None)
            _apply_result_y_ticks(ax, spec.panel)
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

    effective_x_bounds = x_bounds or data_x_bounds
    if effective_x_bounds is not None:
        for ax in axes:
            ax.set_xlim(*effective_x_bounds)
    if events:
        for idx, ax in enumerate(axes):
            label_axis = _add_assim_label_axis(ax, events, idx)
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
    for ax, title_artist in title_artists:
        tick_labels = [label for label in ax.get_yticklabels() if label.get_text()]
        if not tick_labels:
            continue
        left_disp = min(label.get_window_extent(renderer).x0 for label in tick_labels) - 6.0
        x_axes = ax.transAxes.inverted().transform((left_disp, ax.bbox.y1))[0]
        title_artist.set_x(x_axes)
    _build_result_overview_legend(fig)
    fig.savefig(output, dpi=150)
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
    parser.add_argument("--scf-model-csv", type=Path, help="Model SCF CSV (date/time + scf)")
    parser.add_argument("--wet-model-csv", type=Path, help="Model wet-snow CSV (date/time + wet_snow_fraction)")
    parser.add_argument("--scf-env-csv", type=Path, help="SCF envelope CSV (value_min/value_max/value_mean)")
    parser.add_argument("--wet-env-csv", type=Path, help="Wet-snow envelope CSV (value_min/value_max/value_mean)")
    parser.add_argument("--output", type=Path, help="Output PNG path (default: <project>/plots/results/result_overview.png)")
    parser.add_argument("--custom-config", type=Path, help="Custom panel YAML (default: <project-dir>/result_overview_custom.yml)")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    parser.add_argument("--mode", choices=["band", "members"], default="band", help="Plot mode: band (default) or members")
    args = parser.parse_args(argv)

    if configure_logger:
        configure_cli_logger(args.log_level)

    project_dir = Path(args.project_dir)
    setup_dir = Path(args.setup_dir) if args.setup_dir else project_dir.parent.parent
    project_name = project_dir.name

    scf_obs_path = Path(args.scf_obs_csv) if args.scf_obs_csv else default_fraction_obs_path(setup_dir, project_name, "scf_summary.csv")
    wet_obs_path = Path(args.wet_obs_csv) if args.wet_obs_csv else default_fraction_obs_path(setup_dir, project_name, "wet_snow_summary.csv")
    scf_env_path = Path(args.scf_env_csv) if args.scf_env_csv else (project_dir / "point_scf_roi_envelope.csv")
    wet_env_path = Path(args.wet_env_csv) if args.wet_env_csv else (project_dir / "point_wet_snow_roi_envelope.csv")

    scf_obs = _load_scf_obs_series(scf_obs_path)
    wet_obs = load_fraction_series(wet_obs_path, "wet_snow_fraction")
    scf_model = load_fraction_series(Path(args.scf_model_csv), "scf") if args.scf_model_csv else None
    wet_model = load_fraction_series(Path(args.wet_model_csv), "wet_snow_fraction") if args.wet_model_csv else None
    if scf_model is None:
        scf_model = load_open_loop_fraction_series(project_dir, "point_scf_roi.csv", "scf")
    if wet_model is None:
        wet_model = load_open_loop_fraction_series(project_dir, "point_wet_snow_roi.csv", "wet_snow_fraction")
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

    if scf_obs is None or scf_obs.empty:
        logger.warning("SCF obs not found at {} - plotting without obs points", scf_obs_path)
    if wet_obs is None or wet_obs.empty:
        logger.warning("Wet-snow obs not found at {} - plotting without obs points", wet_obs_path)

    try:
        assim_events = load_assimilation_events(project_dir)
    except (FileNotFoundError, ValueError):
        assim_events = []

    stations_df = _load_setup_stations_df(project_dir, setup_dir)
    project_time_bounds = _project_time_bounds(project_dir)

    if all(
        x is None or x.empty
        for x in (scf_obs, wet_obs, scf_model, wet_model, scf_env, wet_env, roi_swe_model, roi_snow_depth_model)
    ) and not roi_swe_members and not roi_snow_depth_members:
        logger.error("No data available to plot. Provide at least one obs/model series.")
        return 1

    custom_config_path = (
        Path(abspath_relative_to(project_dir, args.custom_config)).resolve()
        if args.custom_config
        else _project_custom_config_path(project_dir)
    )
    custom_specs: list[PanelSpec] | None = None
    custom_station_panels: dict[tuple[str, str], StationPanelData] = {}
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
                scf_env=scf_env,
                wet_env=wet_env,
                output=custom_output,
                assim_events=assim_events,
                mode=str(args.mode or "band"),
                roi_swe_model=roi_swe_model,
                roi_swe_members=roi_swe_members,
                roi_snow_depth_model=roi_snow_depth_model,
                roi_snow_depth_members=roi_snow_depth_members,
                panel_specs=custom_specs,
                station_panels=custom_station_panels,
                strict_panels=True,
                x_bounds=project_time_bounds,
            )
            logger.info("Wrote custom plot: {}", custom_output)
        else:
            default_output = default_result_overview_output(project_dir, args.output)
            plot_result_overview(
                scf_obs=scf_obs,
                scf_model=scf_model,
                wet_obs=wet_obs,
                wet_model=wet_model,
                scf_env=scf_env,
                wet_env=wet_env,
                output=default_output,
                assim_events=assim_events,
                mode=str(args.mode or "band"),
                roi_swe_model=roi_swe_model,
                roi_swe_members=roi_swe_members,
                roi_snow_depth_model=roi_snow_depth_model,
                roi_snow_depth_members=roi_snow_depth_members,
                x_bounds=project_time_bounds,
            )
            logger.info("Wrote plot: {}", default_output)

            if custom_specs is not None:
                custom_output = _default_custom_output(project_dir)
                plot_result_overview(
                    scf_obs=scf_obs,
                    scf_model=scf_model,
                    wet_obs=wet_obs,
                    wet_model=wet_model,
                    scf_env=scf_env,
                    wet_env=wet_env,
                    output=custom_output,
                    assim_events=assim_events,
                    mode=str(args.mode or "band"),
                    roi_swe_model=roi_swe_model,
                    roi_swe_members=roi_swe_members,
                    roi_snow_depth_model=roi_snow_depth_model,
                    roi_snow_depth_members=roi_snow_depth_members,
                    panel_specs=custom_specs,
                    station_panels=custom_station_panels,
                    strict_panels=True,
                    x_bounds=project_time_bounds,
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
