"""Lean benchmark plots for openAMUNDSEN-DA projects."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import MultipleLocator

from openamundsen_da.methods.viz._style import (
    FIGHEIGHT_OVERVIEW_ROW,
    FIGWIDTH_OVERVIEW_PAPER,
    LEGEND_NCOL,
    STANDALONE_SCORE_FIGURE_ROW_UNITS,
)
from openamundsen_da.methods.viz._utils import (
    CRPSS_AXIS_POLICY,
    align_figure_title_to_plot_block,
    apply_fraction_grid,
    bounded_metric_range,
    draw_assim_labels,
    draw_assimilation_vlines,
    force_figure_text_black,
    save_figure_png,
    set_matplotlib_text_black,
)
from openamundsen_da.util.da_events import load_assimilation_events

from ..common import ensure_dir, variable_label, variable_style


_VARIABLE_ORDER = ("scf", "wet_snow", "station_hs", "station_swe")
_POINT_TYPE_ORDER = ("prior", "posterior")
_STREAM_ORDER = ("assimilation_fit", "semi_independent", "independent")
_STREAM_MARKERS = {
    "assimilation_fit": "o",
    "semi_independent": "s",
    "independent": "^",
}
_STREAM_LABELS = {
    "assimilation_fit": "assimilation fit",
    "semi_independent": "semi-independent",
    "independent": "independent",
}
_FIGURE_TITLE = "Data assimilation performance scores"
_MARKER_EDGE_COLOR = "#000000"
_MARKER_EDGE_WIDTH = 0.5
_ASSIM_LABEL_ROW_OFFSETS_PTS = [2.0, 8.0]
_ASSIM_LABEL_MIN_SPACING_DAYS = 18.0


class _LabeledLegendTuple(tuple):
    def __new__(cls, artists, label: str):
        obj = super().__new__(cls, artists)
        obj._label = label
        return obj

    def get_label(self) -> str:
        return str(self._label)


class _StageLegendHandle(_LabeledLegendTuple):
    pass


class _StageLegendHandler(HandlerBase):
    def __init__(self, *, x_fracs: tuple[float, float, float] = (0.04, 0.5, 0.96), **kwargs):
        super().__init__(**kwargs)
        self._x_fracs = x_fracs

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        x0 = -xdescent
        y0 = -ydescent
        y_mid = y0 + 0.5 * height
        artists = []
        for artist, x_frac in zip(orig_handle, self._x_fracs):
            marker_artist = Line2D(
                [x0 + x_frac * width],
                [y_mid],
                linestyle="none",
                marker=artist.get_marker(),
                markersize=artist.get_markersize(),
                markerfacecolor=artist.get_markerfacecolor(),
                markeredgecolor=artist.get_markeredgecolor(),
                markeredgewidth=artist.get_markeredgewidth(),
                color=artist.get_color(),
            )
            marker_artist.set_transform(trans)
            artists.append(marker_artist)
        return artists


def _project_assimilation_events(project_dir: Path):
    return sorted(load_assimilation_events(project_dir), key=lambda event: (pd.Timestamp(event.date), str(event.variable)))


def _project_assimilation_dates(project_dir: Path) -> list[pd.Timestamp]:
    return sorted({pd.Timestamp(ev.date).normalize() for ev in _project_assimilation_events(project_dir)})


def _sort_variable(variable: str) -> tuple[int, str]:
    token = str(variable)
    try:
        return (_VARIABLE_ORDER.index(token), token)
    except ValueError:
        return (len(_VARIABLE_ORDER), token)


def score_variable_color(variable: str) -> str:
    return variable_style(str(variable))["line"]


def _clean_plot_dir(plots_dir: Path) -> None:
    stale_items = (
        plots_dir / "core",
        plots_dir / "extended",
        plots_dir / "benchmark_event_skill.png",
        plots_dir / "benchmark_event_skill.svg",
    )
    for stale in stale_items:
        if stale.exists():
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()


def clean_plot_outputs(project_dir: Path, plots_dir: Path) -> None:
    _clean_plot_dir(plots_dir)
    legacy_root = project_dir / "plots" / "benchmark"
    if legacy_root.exists():
        shutil.rmtree(legacy_root)


def _normalized_date_series(frame: pd.DataFrame) -> pd.Series:
    if "date" in frame.columns:
        return pd.to_datetime(frame["date"]).dt.normalize()
    return pd.to_datetime(frame["timestamp"]).dt.normalize()


def _reduce_daily_means(frame: pd.DataFrame, *, point_type: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["variable", "stream", "assimilation_date", "point_type", "crpss", "ner"])
    working = frame.copy()
    working["assimilation_date"] = _normalized_date_series(working)
    aggregated = (
        working.groupby(["variable", "stream", "assimilation_date"], sort=True, dropna=False)[["crpss", "ner"]]
        .mean()
        .reset_index()
    )
    aggregated["point_type"] = point_type
    return aggregated[["variable", "stream", "assimilation_date", "point_type", "crpss", "ner"]]


def build_event_skill_plot_data(event_scores: pd.DataFrame, *, project_dir: Path) -> pd.DataFrame:
    """Build one-row-per-variable-date-point-type data for the headline benchmark plot."""
    if event_scores.empty:
        return pd.DataFrame(columns=["variable", "stream", "assimilation_date", "point_type", "crpss", "ner"])

    assimilation_dates = _project_assimilation_dates(project_dir)
    if not assimilation_dates:
        return pd.DataFrame(columns=["variable", "stream", "assimilation_date", "point_type", "crpss", "ner"])
    valid_dates = pd.DatetimeIndex(assimilation_dates)

    analysis = event_scores[
        (event_scores["score_set"] == "analysis")
        & (event_scores["stream"].isin(_STREAM_ORDER))
        & (event_scores["representation"].isin(["prior", "posterior"]))
    ].copy()
    if not analysis.empty:
        analysis["assimilation_date"] = _normalized_date_series(analysis)
        analysis = analysis[analysis["assimilation_date"].isin(valid_dates)].copy()
    prior = _reduce_daily_means(analysis[analysis["representation"] == "prior"], point_type="prior")
    posterior = _reduce_daily_means(analysis[analysis["representation"] == "posterior"], point_type="posterior")

    merged = pd.concat([prior, posterior], ignore_index=True)
    if merged.empty:
        return merged
    merged = merged.sort_values(
        by=["assimilation_date", "variable", "stream", "point_type"],
        key=lambda series: (
            series.map(_sort_variable)
            if series.name == "variable"
            else series.map(lambda value: _STREAM_ORDER.index(value))
            if series.name == "stream"
            else series.map(lambda value: _POINT_TYPE_ORDER.index(value))
            if series.name == "point_type"
            else series
        ),
    ).reset_index(drop=True)
    return merged


def _scaled_timedelta(delta: pd.Timedelta, factor: float) -> pd.Timedelta:
    return pd.to_timedelta(delta.total_seconds() * factor, unit="s")


def _apply_result_like_time_axis_labels(axes, x_bounds: tuple[pd.Timestamp, pd.Timestamp] | None) -> None:
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


def _assim_style(variable: str) -> dict[str, str]:
    token = str(variable or "").strip().lower()
    if token in {"scf", "wet_snow", "station_hs", "station_swe"}:
        style = variable_style(token)
        return {"variable_key": token, "color": style["line"], "ls": "--"}
    return {"variable_key": token, "color": "#777777", "ls": "--"}


def _date_half_span(assimilation_dates: list[pd.Timestamp], idx: int) -> pd.Timedelta:
    gaps: list[pd.Timedelta] = []
    if idx > 0:
        gaps.append(assimilation_dates[idx] - assimilation_dates[idx - 1])
    if idx < len(assimilation_dates) - 1:
        gaps.append(assimilation_dates[idx + 1] - assimilation_dates[idx])
    if not gaps:
        return pd.Timedelta(hours=18)
    min_gap = min(gaps)
    return min(pd.Timedelta(hours=30), max(pd.Timedelta(hours=6), _scaled_timedelta(min_gap, 0.22)))


def _date_padding(assimilation_dates: list[pd.Timestamp], *, is_start: bool) -> pd.Timedelta:
    if len(assimilation_dates) == 1:
        return pd.Timedelta(days=5)
    gap = (
        assimilation_dates[1] - assimilation_dates[0]
        if is_start
        else assimilation_dates[-1] - assimilation_dates[-2]
    )
    return min(pd.Timedelta(days=5), max(pd.Timedelta(days=2), _scaled_timedelta(gap, 0.5)))


def compute_event_skill_plot_positions(points: pd.DataFrame, *, assimilation_dates: list[pd.Timestamp]) -> pd.DataFrame:
    if points.empty:
        out = points.copy()
        out["plot_x"] = pd.Series(dtype="datetime64[ns]")
        return out

    working = points.copy()
    working["assimilation_date"] = pd.to_datetime(working["assimilation_date"]).dt.normalize()
    assimilation_dates = [pd.Timestamp(ts).normalize() for ts in assimilation_dates]

    position_parts: list[pd.DataFrame] = []
    for idx, date_value in enumerate(assimilation_dates):
        date_rows = working[working["assimilation_date"] == date_value].copy()
        if date_rows.empty:
            continue
        date_rows["_variable_order"] = date_rows["variable"].map(_sort_variable)
        date_rows["_stream_order"] = date_rows["stream"].map(lambda value: _STREAM_ORDER.index(str(value)))
        date_rows["_point_type_order"] = date_rows["point_type"].map(lambda value: _POINT_TYPE_ORDER.index(str(value)))
        date_rows = date_rows.sort_values(["_variable_order", "_stream_order", "_point_type_order"]).reset_index(drop=True)
        half_span = _date_half_span(assimilation_dates, idx)
        if len(date_rows) == 1:
            offsets = np.array([0.0], dtype=float)
        else:
            offsets = np.linspace(-half_span.total_seconds(), half_span.total_seconds(), len(date_rows))
        date_rows["plot_x"] = [
            date_value + pd.to_timedelta(float(offset_seconds), unit="s")
            for offset_seconds in offsets
        ]
        position_parts.append(date_rows.drop(columns=["_variable_order", "_stream_order", "_point_type_order"]))

    if not position_parts:
        working["plot_x"] = pd.Series(dtype="datetime64[ns]")
        return working
    return pd.concat(position_parts, ignore_index=True)


def _metric_ylim(points: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = pd.to_numeric(points[metric], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-0.2, 0.2)
    lo = float(values.min())
    hi = float(values.max())
    span = max(0.2, hi - lo)
    margin = 0.08 * span
    return lo - margin, hi + margin


def score_metric_ylim(points: pd.DataFrame, metric: str) -> tuple[float, float]:
    if metric == "crpss":
        values = pd.to_numeric(points[metric], errors="coerce").to_numpy(dtype=float)
        lower, upper, _ = bounded_metric_range(values, policy=CRPSS_AXIS_POLICY)
        return lower, upper
    return _metric_ylim(points, metric)


def _assim_labels(dates: list[pd.Timestamp]) -> tuple[list[pd.Timestamp], list[str]]:
    return list(dates), [str(idx) for idx in range(1, len(dates) + 1)]


def _add_assim_label_axis(ax, assimilation_dates: list[pd.Timestamp], idx: int):
    if not assimilation_dates:
        return None

    x_min, x_max = sorted(ax.get_xlim())
    visible_start = pd.Timestamp(mdates.num2date(x_min)).tz_localize(None)
    visible_end = pd.Timestamp(mdates.num2date(x_max)).tz_localize(None)
    dates, labels = _assim_labels(assimilation_dates)
    visible_items = [
        (date_value, label)
        for date_value, label in zip(dates, labels)
        if visible_start <= date_value <= visible_end
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


def add_score_assim_label_axis(ax, assimilation_dates: list[pd.Timestamp], idx: int):
    return _add_assim_label_axis(ax, assimilation_dates, idx)


def _draw_metric_panel(
    ax,
    *,
    points: pd.DataFrame,
    metric: str,
    variables: list[str],
    assimilation_events: list,
) -> None:
    _apply_score_axis(ax, points, metric)
    ax.axhline(0.0, color="#6f6f6f", lw=0.9, ls="--", zorder=1)
    for event in assimilation_events:
        meta = _assim_style(str(event.variable))
        draw_assimilation_vlines(
            ax,
            [pd.Timestamp(event.date).normalize()],
            color=str(meta["color"]),
            ls=str(meta["ls"]),
            lw=1.2,
            alpha=0.95,
            label="_nolegend_",
            zorder=2,
        )

    for point_type in _POINT_TYPE_ORDER:
        point_rows = points[points["point_type"] == point_type]
        if point_rows.empty:
            continue
        for stream in _STREAM_ORDER:
            stream_rows = point_rows[point_rows["stream"] == stream]
            if stream_rows.empty:
                continue
            for variable in variables:
                var_rows = stream_rows[stream_rows["variable"] == variable]
                if var_rows.empty:
                    continue
                score_color = score_variable_color(variable)
                facecolors = "white" if point_type == "prior" else score_color
                edgecolors = score_color
                ax.scatter(
                    pd.to_datetime(var_rows["plot_x"]),
                    var_rows[metric],
                    s=30.0,
                    marker=_STREAM_MARKERS[stream],
                    facecolors=facecolors,
                    edgecolors=edgecolors,
                    linewidths=_MARKER_EDGE_WIDTH,
                    zorder=5,
                )


def _apply_score_axis(ax, points: pd.DataFrame, metric: str) -> None:
    lower, upper = score_metric_ylim(points, metric)
    step = 0.5
    if metric == "crpss":
        values = pd.to_numeric(points[metric], errors="coerce").to_numpy(dtype=float)
        lower, upper, step = bounded_metric_range(values, policy=CRPSS_AXIS_POLICY)
    if np.isclose(lower, upper):
        upper = lower + step
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    apply_fraction_grid(ax, y_step=None)


def draw_score_metric_panel(
    ax,
    *,
    points: pd.DataFrame,
    metric: str,
    variables: list[str],
    assimilation_events: list,
) -> None:
    _draw_metric_panel(
        ax,
        points=points,
        metric=metric,
        variables=variables,
        assimilation_events=assimilation_events,
    )


def _stage_legend_handle(point_type: str) -> _StageLegendHandle:
    is_posterior = point_type == "posterior"
    marker_face = "#000000" if is_posterior else "white"
    marker_edge = "#000000"
    artists = tuple(
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=_STREAM_MARKERS[stream],
            markersize=5.5,
            markerfacecolor=marker_face,
            markeredgecolor=marker_edge,
            markeredgewidth=_MARKER_EDGE_WIDTH,
            color="black",
        )
        for stream in _STREAM_ORDER
    )
    return _StageLegendHandle(artists, point_type)


def score_legend_handler_map() -> dict[type, HandlerBase]:
    return {_StageLegendHandle: _StageLegendHandler()}


def score_legend_handles(variables: list[str], *, include_da_event: bool = True) -> list:
    handles: list = []
    for variable in variables:
        score_color = score_variable_color(variable)
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=5.5,
                markerfacecolor=score_color,
                markeredgecolor=score_color,
                markeredgewidth=_MARKER_EDGE_WIDTH,
                color=score_color,
                label=variable_label(variable),
            )
        )
    extra_handles = [
            _stage_legend_handle("prior"),
            _stage_legend_handle("posterior"),
            *[
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker=_STREAM_MARKERS[stream],
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgecolor=_MARKER_EDGE_COLOR,
                    markeredgewidth=_MARKER_EDGE_WIDTH,
                    color="black",
                    label=_STREAM_LABELS[stream],
                )
                for stream in _STREAM_ORDER
            ],
    ]
    if include_da_event:
        extra_handles.append(
            Line2D(
                [0],
                [0],
                color="#777777",
                lw=1.2,
                ls="--",
                label="data assimilation event",
            )
        )
    handles.extend(extra_handles)
    return handles


def score_variable_sort_key(variable: str) -> tuple[int, str]:
    return _sort_variable(variable)


def _align_title_to_plot_block(fig, axes: tuple) -> None:
    align_figure_title_to_plot_block(fig, axes)


def align_score_title_to_plot_block(fig, axes: tuple) -> None:
    _align_title_to_plot_block(fig, axes)


def _write_event_skill_figure(
    out_path: Path,
    *,
    event_scores: pd.DataFrame,
    project_dir: Path,
) -> Path | None:
    points = build_event_skill_plot_data(event_scores, project_dir=project_dir)
    if points.empty:
        return None

    assimilation_events = _project_assimilation_events(project_dir)
    assimilation_dates = sorted({pd.Timestamp(event.date).normalize() for event in assimilation_events})
    if not assimilation_dates:
        return None
    start_pad = _date_padding(assimilation_dates, is_start=True)
    end_pad = _date_padding(assimilation_dates, is_start=False)
    x_min = assimilation_dates[0] - start_pad
    x_max = assimilation_dates[-1] + end_pad

    variables = sorted(points["variable"].astype(str).unique(), key=_sort_variable)
    points = compute_event_skill_plot_positions(points, assimilation_dates=assimilation_dates)

    set_matplotlib_text_black(matplotlib)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIGWIDTH_OVERVIEW_PAPER, FIGHEIGHT_OVERVIEW_ROW * STANDALONE_SCORE_FIGURE_ROW_UNITS),
        sharex=True,
        squeeze=False,
    )
    ax_crpss, ax_ner = axes[:, 0]

    _draw_metric_panel(
        ax_crpss,
        points=points,
        metric="crpss",
        variables=variables,
        assimilation_events=assimilation_events,
    )
    _draw_metric_panel(
        ax_ner,
        points=points,
        metric="ner",
        variables=variables,
        assimilation_events=assimilation_events,
    )

    ax_crpss.set_ylabel("CRPSS")
    ax_ner.set_ylabel("NER")
    ax_ner.set_xlabel("")
    ax_crpss.set_title("")
    ax_ner.set_title("")

    for ax in (ax_crpss, ax_ner):
        ax.set_xlim(x_min, x_max)
    _apply_result_like_time_axis_labels((ax_crpss, ax_ner), (x_min, x_max))

    label_axes = []
    for idx, ax in enumerate((ax_crpss, ax_ner)):
        label_axis = _add_assim_label_axis(ax, assimilation_dates, idx)
        if label_axis is not None:
            label_axes.append(label_axis)

    legend_handles = score_legend_handles(variables)
    fig.legend(
        handles=legend_handles,
        handler_map=score_legend_handler_map(),
        loc="lower left",
        bbox_to_anchor=(0.055, 0.006, 0.88, 0.052),
        bbox_transform=fig.transFigure,
        mode="expand",
        ncol=min(max(4, LEGEND_NCOL + 1), len(legend_handles)),
        frameon=False,
        fontsize=8.0,
        handlelength=2.55,
        columnspacing=1.1,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(-0.015, 0.058, 0.992, 0.998), h_pad=0.72)
    fig.align_ylabels((ax_crpss, ax_ner))
    fig.suptitle(_FIGURE_TITLE, ha="left", fontsize=10.2)
    _align_title_to_plot_block(fig, (ax_crpss, ax_ner, *label_axes))
    force_figure_text_black(fig, (ax_crpss, ax_ner, *label_axes))
    ensure_dir(out_path.parent)
    save_figure_png(fig, out_path)
    plt.close(fig)
    return out_path


def write_plots(
    plots_dir: Path,
    *,
    event_scores: pd.DataFrame,
    project_dir: Path,
    case_scores: pd.DataFrame | None = None,
    reliability: pd.DataFrame | None = None,
) -> dict[str, Path]:
    del case_scores, reliability
    plots_dir = Path(plots_dir)
    clean_plot_outputs(project_dir, plots_dir)

    outputs: dict[str, Path] = {}
    out_path = plots_dir / "performance_scores.png"
    written = _write_event_skill_figure(out_path, event_scores=event_scores, project_dir=project_dir)
    if written is not None:
        outputs["performance_scores"] = written
    return outputs


__all__ = [
    "add_score_assim_label_axis",
    "align_score_title_to_plot_block",
    "build_event_skill_plot_data",
    "clean_plot_outputs",
    "compute_event_skill_plot_positions",
    "draw_score_metric_panel",
    "score_legend_handles",
    "score_legend_handler_map",
    "score_metric_ylim",
    "score_variable_sort_key",
    "write_plots",
]
