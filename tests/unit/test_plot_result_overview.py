from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import openamundsen_da.methods.viz.plots.result_overview as plot_mod
from openamundsen_da.methods.viz.plots.theme import BAND_ALPHA, da_variable_style
from openamundsen_da.methods.viz.plots.result_overview import (
    PanelSpec,
    StationPanelData,
    _project_custom_config_path,
    _load_station_panel_data,
    _parse_panel_specs,
    plot_result_overview,
)


def _series(values: list[float]) -> pd.Series:
    dates = pd.to_datetime(["2023-01-01", "2023-01-02"])
    return pd.Series(values, index=dates)


def _frame(col: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2023-01-02"]), col: values})


def _score_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": "scf",
                "stream": "assimilation_fit",
                "assimilation_date": "2023-01-02",
                "point_type": "prior",
                "crpss": 0.10,
                "ner": 0.05,
                "zskill": float("nan"),
            },
            {
                "variable": "scf",
                "stream": "assimilation_fit",
                "assimilation_date": "2023-01-02",
                "point_type": "posterior",
                "crpss": 0.35,
                "ner": 0.22,
                "zskill": float("nan"),
            },
            {
                "variable": "station_swe",
                "stream": "semi_independent",
                "assimilation_date": "2023-01-03",
                "point_type": "prior",
                "crpss": 0.18,
                "ner": 0.11,
                "zskill": -0.08,
            },
            {
                "variable": "station_swe",
                "stream": "semi_independent",
                "assimilation_date": "2023-01-03",
                "point_type": "posterior",
                "crpss": 0.26,
                "ner": 0.16,
                "zskill": 0.24,
            },
        ]
    )


def _panel_axes(fig) -> list:
    return [ax for ax in fig.axes if not ax.get_label().startswith("assimilation_label_axis")]


def _figure_legend_labels(fig, index: int = 0) -> list[str]:
    return [text.get_text() for text in fig.legends[index].get_texts()]


def _assert_figure_legends_clear_axes(fig) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bboxes = [
        legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        for legend in fig.legends
    ]
    axes_bboxes = [ax.get_position() for ax in _panel_axes(fig)]
    for legend_bbox in legend_bboxes:
        assert all(not legend_bbox.overlaps(ax_bbox) for ax_bbox in axes_bboxes)
    for idx, legend_bbox in enumerate(legend_bboxes):
        assert all(not legend_bbox.overlaps(other_bbox) for other_bbox in legend_bboxes[idx + 1 :])


def test_default_wsl_overview_env_uses_prior_member_median_minmax_and_preserves_gaps(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    member_001 = project_dir / "steps" / "step_00" / "ensembles" / "prior" / "member_001" / "results"
    member_002 = project_dir / "steps" / "step_00" / "ensembles" / "prior" / "member_002" / "results"
    member_001.mkdir(parents=True, exist_ok=True)
    member_002.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "time": ["2023-04-29", "2023-05-11", "2023-05-15"],
            "wet_snow_line": [2400.0, float("nan"), 2600.0],
        }
    ).to_csv(member_001 / "point_wet_snow_line_roi.csv", index=False)
    pd.DataFrame(
        {
            "time": ["2023-04-29", "2023-05-11", "2023-05-15"],
            "wet_snow_line": [2500.0, float("nan"), 2700.0],
        }
    ).to_csv(member_002 / "point_wet_snow_line_roi.csv", index=False)

    env = plot_mod._default_wsl_overview_env(project_dir)

    assert env is not None
    assert list(env["date"]) == list(pd.to_datetime(["2023-04-29", "2023-05-11", "2023-05-15"]))
    assert env.iloc[0]["value_mean"] == 2450.0
    assert env.iloc[0]["value_min"] == 2400.0
    assert env.iloc[0]["value_max"] == 2500.0
    assert pd.isna(env.iloc[1]["value_mean"])
    assert pd.isna(env.iloc[1]["value_min"])
    assert pd.isna(env.iloc[1]["value_max"])
    assert env.iloc[2]["value_mean"] == 2650.0
    assert env.iloc[2]["value_min"] == 2600.0
    assert env.iloc[2]["value_max"] == 2700.0
    assert list(env["n"]) == [2.0, 0.0, 2.0]


def test_load_wsl_prior_coverage_frame_uses_value_model_from_weights_csv(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    weights_dir = project_dir / "steps" / "step_01" / "assim"
    weights_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"member_id": "member_001", "value_model": 2400.0, "value_obs": 2550.0, "weight": 0.4},
            {"member_id": "member_002", "value_model": 2600.0, "value_obs": 2550.0, "weight": 0.6},
        ]
    ).to_csv(weights_dir / "weights_wet_snow_line_20230511.csv", index=False)

    frame = plot_mod._load_wsl_prior_coverage_frame(project_dir)

    assert frame is not None
    assert list(frame["date"]) == [pd.Timestamp("2023-05-11")]
    assert list(frame["value_mean"]) == [2500.0]
    assert list(frame["value_min"]) == [2400.0]
    assert list(frame["value_max"]) == [2600.0]
    assert list(frame["value_obs"]) == [2550.0]
    assert list(frame["n"]) == [2]


def test_plot_result_overview_uses_four_panels_when_roi_series_exist(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    recorded: dict[str, int] = {}
    original_subplots = plt.subplots
    original_close = plt.close

    def _spy_subplots(nrows, *args, **kwargs):
        recorded["nrows"] = nrows
        return original_subplots(nrows, *args, **kwargs)

    monkeypatch.setattr(plt, "subplots", _spy_subplots)
    monkeypatch.setattr(plt, "close", lambda fig=None: None)

    out_path = tmp_path / "result_overview.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=_frame("scf", [0.2, 0.4]),
        wet_obs=None,
        wet_model=_frame("wet_snow_fraction", [0.1, 0.2]),
        scf_env=None,
        wet_env=None,
        roi_swe_model=_frame("swe", [10.0, 12.0]),
        roi_swe_members=[_series([8.0, 11.0]), _series([9.0, 13.0])],
        roi_snow_depth_model=_frame("snow_depth", [0.3, 0.4]),
        roi_snow_depth_members=[_series([0.2, 0.3]), _series([0.4, 0.5])],
        output=out_path,
    )

    assert recorded["nrows"] == 4
    axes = _panel_axes(plt.gcf())
    assert [ax.get_ylabel() for ax in axes] == ["", "", "", ""]
    assert axes[0].get_title(loc="left") == "(a) Snow cover fraction (roi)"
    assert axes[1].get_title(loc="left") == "(b) Wet snow fraction (roi)"
    assert axes[2].lines[0].get_color() == da_variable_style("station_swe")["line"]
    assert axes[2].lines[0].get_linewidth() == pytest.approx(plot_mod._RESULT_OVERVIEW_DATA_LW)
    assert axes[2].lines[1].get_color() == "black"
    assert axes[2].lines[1].get_linewidth() == pytest.approx(plot_mod._RESULT_OVERVIEW_DATA_LW)
    assert axes[3].lines[0].get_color() == da_variable_style("station_hs")["line"]
    assert axes[3].lines[0].get_linewidth() == pytest.approx(plot_mod._RESULT_OVERVIEW_DATA_LW)
    for ax in axes:
        assert ax._left_title.get_position()[0] == pytest.approx(0.0)
        assert ax._left_title.get_position()[1] == pytest.approx(1.0)
    assert isinstance(axes[2].yaxis.get_major_locator(), mticker.MultipleLocator)
    assert isinstance(axes[3].yaxis.get_major_locator(), mticker.MultipleLocator)
    swe_ticks = axes[2].yaxis.get_major_locator().tick_values(0.0, 200.0)
    sd_ticks = axes[3].yaxis.get_major_locator().tick_values(0.0, 2.0)
    assert swe_ticks[1] - swe_ticks[0] == 50.0
    assert sd_ticks[1] - sd_ticks[0] == 0.25
    assert plt.gcf()._suptitle is None
    assert plt.gcf().get_size_inches()[0] == plot_mod.FIGWIDTH_OVERVIEW_PAPER
    assert plt.gcf().get_size_inches()[1] == pytest.approx(
        plot_mod.FIGHEIGHT_OVERVIEW_ROW * plot_mod.OVERVIEW_STANDARD_PANEL_HEIGHT_FACTOR * 4.0
    )
    assert len(plt.gcf().legends) == 1
    assert out_path.is_file()
    original_close(plt.gcf())


def test_plot_result_overview_keeps_two_panels_without_roi_series(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    recorded: dict[str, int] = {}
    original_subplots = plt.subplots

    def _spy_subplots(nrows, *args, **kwargs):
        recorded["nrows"] = nrows
        return original_subplots(nrows, *args, **kwargs)

    monkeypatch.setattr(plt, "subplots", _spy_subplots)

    out_path = tmp_path / "result_overview.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=_frame("scf", [0.2, 0.4]),
        wet_obs=None,
        wet_model=_frame("wet_snow_fraction", [0.1, 0.2]),
        scf_env=None,
        wet_env=None,
        output=out_path,
    )

    assert recorded["nrows"] == 2
    assert out_path.is_file()


def test_plot_result_overview_adds_wsl_panel_when_wsl_series_exist(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    recorded: dict[str, int] = {}
    original_subplots = plt.subplots
    original_close = plt.close

    def _spy_subplots(nrows, *args, **kwargs):
        recorded["nrows"] = nrows
        return original_subplots(nrows, *args, **kwargs)

    monkeypatch.setattr(plt, "subplots", _spy_subplots)
    monkeypatch.setattr(plt, "close", lambda fig=None: None)

    out_path = tmp_path / "result_overview.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=_frame("scf", [0.2, 0.4]),
        wet_obs=None,
        wet_model=_frame("wet_snow_fraction", [0.1, 0.2]),
        wsl_obs=_frame("wet_snow_line", [2400.0, 2450.0]),
        wsl_model=_frame("wet_snow_line", [2380.0, 2440.0]),
        scf_env=None,
        wet_env=None,
        wsl_env=_frame("value_mean", [2390.0, 2430.0]).assign(value_min=[2360.0, 2400.0], value_max=[2420.0, 2460.0]),
        output=out_path,
    )

    assert recorded["nrows"] == 3
    axes = _panel_axes(plt.gcf())
    assert [ax.get_ylabel() for ax in axes] == ["", "", ""]
    assert out_path.is_file()
    original_close(plt.gcf())


def test_plot_result_overview_ylabels_do_not_overlap_with_stacked_custom_panels(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=_frame("scf", [0.2, 0.4]),
            wet_obs=None,
            wet_model=_frame("wet_snow_fraction", [0.1, 0.2]),
            wsl_obs=None,
            wsl_model=_frame("wet_snow_line", [2380.0, 2440.0]),
            scf_env=None,
            wet_env=None,
            wsl_env=_frame("value_mean", [2390.0, 2430.0]).assign(
                value_min=[2360.0, 2400.0],
                value_max=[2420.0, 2460.0],
            ),
            output=out_path,
            panel_specs=[
                PanelSpec(panel="fSC"),
                PanelSpec(panel="WSF"),
                PanelSpec(panel="WSLA"),
                PanelSpec(panel="station-sd", station_id="proviantdepot"),
                PanelSpec(panel="ess"),
                PanelSpec(panel="scores-crpss"),
            ],
            station_panels={
                ("proviantdepot", "snow_depth"): StationPanelData(
                    station_id="proviantdepot",
                    display_name="Proviantdepot",
                    altitude_m=2659.0,
                    open_loop=_series([0.4, 0.5]),
                    members=[_series([0.3, 0.45]), _series([0.35, 0.55])],
                    obs=_series([0.32, 0.53]),
                )
            },
            ess_panel=plot_mod.EssPanelData(
                series=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                        "ess": [22.0, 25.0],
                    }
                ),
                ensemble_size=30,
                threshold=21.0,
            ),
            score_points=_score_points(),
            assim_events=[
                plot_mod.AssimilationEvent(
                    date=pd.Timestamp("2023-01-02").date(),
                    variable="scf",
                    product="SNOWCOVER",
                ),
                plot_mod.AssimilationEvent(
                    date=pd.Timestamp("2023-01-03").date(),
                    variable="station_hs",
                    product="STATION",
                ),
            ],
            strict_panels=True,
        )

        fig = plt.gcf()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes = _panel_axes(fig)
        assert [ax.get_ylabel() for ax in axes] == ["", "", "", "", "", ""]
        ytick_bboxes = [
            bbox
            for ax in axes
            for label in ax.get_yticklabels()
            if label.get_text() and label.get_visible() and (bbox := label.get_window_extent(renderer)).width > 0
        ]
        for upper_bbox, lower_bbox in zip(ytick_bboxes, ytick_bboxes[1:]):
            assert not upper_bbox.overlaps(lower_bbox)
        for ax in axes:
            for label in ax.get_yticklabels():
                if label.get_text() and label.get_visible():
                    assert not ax._left_title.get_window_extent(renderer).overlaps(label.get_window_extent(renderer))
        assert out_path.is_file()
    finally:
        plt.close = original_close
        original_close(plt.gcf())


def test_plot_result_overview_marks_wet_snow_line_events_on_wsl_panel(monkeypatch, tmp_path: Path) -> None:
    marker_calls: list[tuple[list[pd.Timestamp], pd.DataFrame, str | None]] = []

    def _record_markers(ax, *, dates, obs, **kwargs) -> None:
        marker_calls.append((list(pd.to_datetime(dates)), obs.copy(), kwargs.get("marker")))

    monkeypatch.setattr(plot_mod, "draw_assimilation_markers", _record_markers)

    event_date = pd.Timestamp("2023-04-29")
    out_path = tmp_path / "result_overview.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=None,
        wet_obs=None,
        wet_model=None,
        wsl_obs=pd.DataFrame({"date": [event_date], "wet_snow_line": [2450.0]}),
        wsl_model=pd.DataFrame({"date": [event_date], "wet_snow_line": [2430.0]}),
        scf_env=None,
        wet_env=None,
        wsl_env=None,
        output=out_path,
        assim_events=[plot_mod.AssimilationEvent(date=event_date.date(), variable="wet_snow_line", product="WETSNOW")],
    )

    assert len(marker_calls) == 1
    assert marker_calls[0][0] == [event_date]
    assert list(marker_calls[0][1]["wet_snow_line"]) == [2450.0]
    assert marker_calls[0][2] is None
    assert out_path.is_file()


def test_plot_result_overview_does_not_mark_wet_snow_line_events_on_fws_panel(
    monkeypatch,
    tmp_path: Path,
) -> None:
    marker_calls: list[tuple[list[pd.Timestamp], pd.DataFrame, str | None]] = []

    def _record_markers(ax, *, dates, obs, **kwargs) -> None:
        marker_calls.append((list(pd.to_datetime(dates)), obs.copy(), kwargs.get("marker")))

    monkeypatch.setattr(plot_mod, "draw_assimilation_markers", _record_markers)

    event_date = pd.Timestamp("2023-04-29")
    out_path = tmp_path / "result_overview.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=None,
        wet_obs=pd.DataFrame({"date": [event_date], "wet_snow_fraction": [0.35]}),
        wet_model=pd.DataFrame({"date": [event_date], "wet_snow_fraction": [0.30]}),
        wsl_obs=None,
        wsl_model=None,
        scf_env=None,
        wet_env=None,
        wsl_env=None,
        output=out_path,
        assim_events=[plot_mod.AssimilationEvent(date=event_date.date(), variable="wet_snow_line", product="WETSNOW")],
    )

    assert marker_calls == []
    assert out_path.is_file()


def test_plot_result_overview_wsl_keeps_model_gaps_and_omits_missing_obs_points(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import matplotlib.axes

    captured_plots: list[dict[str, object]] = []
    captured_marker_calls: list[dict[str, object]] = []
    original_plot = matplotlib.axes.Axes.plot

    def _spy_plot(self, x, y, *args, **kwargs):
        captured_plots.append(
            {
                "color": kwargs.get("color"),
                "marker": kwargs.get("marker"),
                "linestyle": kwargs.get("linestyle"),
                "y": list(pd.Series(y)),
            }
        )
        return original_plot(self, x, y, *args, **kwargs)

    def _record_markers(ax, *, dates, obs, **kwargs) -> None:
        captured_marker_calls.append(
            {
                "dates": list(pd.to_datetime(dates)),
                "obs": obs.copy(),
                "marker": kwargs.get("marker"),
            }
        )

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)
    monkeypatch.setattr(plot_mod, "draw_assimilation_markers", _record_markers)

    dates = pd.to_datetime(["2023-04-29", "2023-05-11", "2023-05-15"])
    out_path = tmp_path / "result_overview.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=None,
        wet_obs=None,
        wet_model=None,
        wsl_obs=pd.DataFrame({"date": dates[:2], "wet_snow_line": [2450.0, float("nan")]}),
        wsl_model=pd.DataFrame({"date": dates, "wet_snow_line": [2430.0, float("nan"), 2520.0]}),
        scf_env=None,
        wet_env=None,
        wsl_env=None,
        output=out_path,
        assim_events=[
            plot_mod.AssimilationEvent(date=dates[0].date(), variable="wet_snow_line", product="WETSNOW"),
            plot_mod.AssimilationEvent(date=dates[1].date(), variable="wet_snow_line", product="WETSNOW"),
        ],
    )

    wsl_line_calls = [call for call in captured_plots if call["color"] == "black"]
    wsl_obs_calls = [call for call in captured_plots if call["color"] == plot_mod.COLOR_DA_OBS and call["marker"] == "o"]

    assert any(pd.isna(value) for value in wsl_line_calls[0]["y"])
    assert len(wsl_obs_calls) == 1
    assert wsl_obs_calls[0]["y"] == [2450.0]
    assert len(captured_marker_calls) == 1
    assert captured_marker_calls[0]["dates"] == [dates[0], dates[1]]
    assert list(captured_marker_calls[0]["obs"]["wet_snow_line"]) == [2450.0]
    assert captured_marker_calls[0]["marker"] is None
    assert out_path.is_file()


def test_plot_result_overview_wsl_draws_prior_coverage_markers_from_weights_frame(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import matplotlib.axes

    captured_vlines: list[dict[str, object]] = []
    captured_scatter: list[dict[str, object]] = []
    original_vlines = matplotlib.axes.Axes.vlines
    original_scatter = matplotlib.axes.Axes.scatter

    def _spy_vlines(self, x, ymin, ymax, *args, **kwargs):
        captured_vlines.append(
            {
                "x": list(pd.to_datetime(pd.Index(x))),
                "ymin": list(pd.Series(ymin, dtype=float)),
                "ymax": list(pd.Series(ymax, dtype=float)),
            }
        )
        return original_vlines(self, x, ymin, ymax, *args, **kwargs)

    def _spy_scatter(self, x, y, *args, **kwargs):
        captured_scatter.append(
            {
                "x": list(pd.to_datetime(pd.Index(x))),
                "y": list(pd.Series(y, dtype=float)),
                "marker": kwargs.get("marker"),
            }
        )
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "vlines", _spy_vlines)
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", _spy_scatter)

    event_date = pd.Timestamp("2023-05-11")
    out_path = tmp_path / "result_overview.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=None,
        wet_obs=None,
        wet_model=None,
        wsl_obs=None,
        wsl_model=pd.DataFrame({"date": [event_date], "wet_snow_line": [2500.0]}),
        scf_env=None,
        wet_env=None,
        wsl_env=pd.DataFrame({"date": [event_date], "value_mean": [2800.0], "value_min": [2750.0], "value_max": [2850.0]}),
        wsl_prior_coverage=pd.DataFrame(
            {"date": [event_date], "value_mean": [2100.0], "value_min": [2000.0], "value_max": [2200.0]}
        ),
        output=out_path,
    )

    assert len(captured_vlines) == 1
    assert captured_vlines[0]["x"] == [event_date]
    assert captured_vlines[0]["ymin"] == [2000.0]
    assert captured_vlines[0]["ymax"] == [2200.0]
    prior_center_calls = [call for call in captured_scatter if call["marker"] == "_"]
    assert len(prior_center_calls) == 1
    assert prior_center_calls[0]["x"] == [event_date]
    assert prior_center_calls[0]["y"] == [2100.0]
    assert out_path.is_file()


def test_plot_result_overview_fraction_panels_use_point_two_y_step(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=_frame("scf", [0.2, 0.4]),
            wet_obs=None,
            wet_model=_frame("wet_snow_fraction", [0.1, 0.2]),
            scf_env=None,
            wet_env=None,
            output=out_path,
        )

        axes = _panel_axes(plt.gcf())
        scf_ticks = list(axes[0].get_yticks())
        wet_ticks = list(axes[1].get_yticks())
        assert scf_ticks == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert wet_ticks == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert [label.get_text() for label in axes[0].get_yticklabels() if label.get_text()] == ["0.5", "1"]
        assert [label.get_text() for label in axes[1].get_yticklabels() if label.get_text()] == ["0.5", "1"]
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_wsla_panel_uses_nice_altitude_y_step(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=None,
            wet_obs=None,
            wet_model=None,
            wsl_obs=None,
            wsl_model=_frame("wet_snow_line", [2100.0, 3900.0]),
            scf_env=None,
            wet_env=None,
            wsl_env=_frame("value_mean", [2300.0, 3400.0]).assign(
                value_min=[1900.0, 3100.0],
                value_max=[2600.0, 4100.0],
            ),
            output=out_path,
            panel_specs=[PanelSpec(panel="WSLA")],
            strict_panels=True,
        )

        axes = _panel_axes(plt.gcf())
        locator = axes[0].yaxis.get_major_locator()
        assert isinstance(locator, mticker.MultipleLocator)
        ticks = locator.tick_values(*axes[0].get_ylim())
        assert ticks[1] - ticks[0] == pytest.approx(500.0)
        assert [label.get_text() for label in axes[0].get_yticklabels() if label.get_text()] == ["2000", "3000"]
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_wsla_labels_lower_clean_1000m_ticks_only() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, ax = plt.subplots()
    try:
        ax.set_ylim(1900.0, 4100.0)
        ax.yaxis.set_major_locator(MultipleLocator(500.0))

        plot_mod._label_lower_clean_1000m_y_ticks(ax)
        fig.canvas.draw()

        labels = [label.get_text() for label in ax.get_yticklabels()]
        assert [label for label in labels if label] == ["2000", "3000"]
        assert "2500" not in labels
        assert "3500" not in labels
        assert "4000" not in labels
    finally:
        plt.close(fig)


def test_plot_result_overview_roi_swe_dense_labels_start_from_bottom() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, ax = plt.subplots()
    try:
        ax.set_ylim(0.0, 200.0)
        ax.yaxis.set_major_locator(MultipleLocator(50.0))

        plot_mod._label_every_second_dense_y_ticks_from_bottom(ax, max_visible_labels=4)
        fig.canvas.draw()

        labels = [label.get_text() for label in ax.get_yticklabels()]
        assert [label for label in labels if label] == ["0", "100"]
        assert "200" not in labels
    finally:
        plt.close(fig)


def test_plot_result_overview_roi_band_excludes_open_loop_and_plots_it_separately(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import matplotlib.axes

    captured: dict[str, object] = {"bands": [], "plots": []}
    original_fill_between = matplotlib.axes.Axes.fill_between
    original_plot = matplotlib.axes.Axes.plot

    def _spy_fill_between(self, x, y1, y2, *args, **kwargs):
        captured["bands"].append((list(y1), list(y2), kwargs.get("color"), kwargs.get("alpha")))
        return original_fill_between(self, x, y1, y2, *args, **kwargs)

    def _spy_plot(self, x, y, *args, **kwargs):
        captured["plots"].append((list(pd.Series(y)), kwargs.get("color")))
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", _spy_fill_between)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

    out_path = tmp_path / "result_overview.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=None,
        wet_obs=None,
        wet_model=None,
        scf_env=None,
        wet_env=None,
        roi_swe_model=_frame("swe", [100.0, 100.0]),
        roi_swe_members=[_series([1.0, 1.0]), _series([3.0, 3.0])],
        output=out_path,
    )

    band_low, band_high, band_color, band_alpha = captured["bands"][0]
    assert band_color == da_variable_style("station_swe")["fill"]
    assert band_alpha == BAND_ALPHA
    assert max(band_high) < 10.0
    open_loop_calls = [vals for vals, color in captured["plots"] if color == "black"]
    assert open_loop_calls == [[100.0, 100.0]]
    assert out_path.is_file()


def test_plot_result_overview_draws_all_assim_events_on_every_panel(monkeypatch, tmp_path: Path) -> None:
    marker_calls: list[list[pd.Timestamp]] = []
    vline_calls: list[tuple[list[pd.Timestamp], str | None, str | None, float | None]] = []
    label_calls: list[tuple[str, list[pd.Timestamp], list[str], object, float | None, list[float] | None, float | None, float | None, str | None]] = []

    def _record_markers(ax, *, dates, **kwargs) -> None:
        marker_calls.append(list(pd.to_datetime(dates)))

    def _record_vlines(ax, dates, **kwargs) -> None:
        vline_calls.append((list(pd.to_datetime(dates)), kwargs.get("color"), kwargs.get("ls"), kwargs.get("lw")))

    def _record_labels(ax, dates, **kwargs) -> None:
        label_calls.append(
            (
                ax.get_label(),
                list(pd.to_datetime(dates)),
                list(kwargs.get("labels") or []),
                kwargs.get("colors"),
                kwargs.get("rotation"),
                list(kwargs.get("row_y_offsets_pts") or []),
                kwargs.get("min_row_spacing_days"),
                kwargs.get("axes_y"),
                kwargs.get("ha"),
            )
        )

    monkeypatch.setattr(plot_mod, "draw_assimilation_markers", _record_markers)
    monkeypatch.setattr(plot_mod, "draw_assimilation_vlines", _record_vlines)
    monkeypatch.setattr(plot_mod, "draw_adaptive_assim_labels", _record_labels)

    scf_date = pd.Timestamp("2023-01-01")
    wet_date = pd.Timestamp("2023-01-02")
    hs_date = pd.Timestamp("2023-01-03")
    swe_date = pd.Timestamp("2023-01-04")
    dates = pd.to_datetime([scf_date, wet_date, hs_date, swe_date])
    scf_frame = pd.DataFrame({"date": dates, "scf": [0.2, 0.4, 0.3, 0.5]})
    wet_frame = pd.DataFrame({"date": dates, "wet_snow_fraction": [0.1, 0.2, 0.3, 0.4]})
    swe_frame = pd.DataFrame({"date": dates, "swe": [10.0, 12.0, 11.0, 13.0]})
    sd_frame = pd.DataFrame({"date": dates, "snow_depth": [0.3, 0.4, 0.5, 0.45]})
    swe_members = [
        pd.Series([8.0, 11.0, 10.0, 12.0], index=dates),
        pd.Series([9.0, 13.0, 12.0, 14.0], index=dates),
    ]
    sd_members = [
        pd.Series([0.2, 0.3, 0.35, 0.4], index=dates),
        pd.Series([0.4, 0.5, 0.55, 0.6], index=dates),
    ]

    out_path = tmp_path / "result_overview.png"
    plot_result_overview(
        scf_obs=scf_frame,
        scf_model=scf_frame,
        wet_obs=wet_frame,
        wet_model=wet_frame,
        scf_env=None,
        wet_env=None,
        roi_swe_model=swe_frame,
        roi_swe_members=swe_members,
        roi_snow_depth_model=sd_frame,
        roi_snow_depth_members=sd_members,
        assim_events=[
            plot_mod.AssimilationEvent(date=scf_date.date(), variable="scf", product="SNOWCOVER"),
            plot_mod.AssimilationEvent(date=wet_date.date(), variable="wet_snow", product="WETSNOW"),
            plot_mod.AssimilationEvent(date=hs_date.date(), variable="station_hs", product="STATION"),
            plot_mod.AssimilationEvent(date=swe_date.date(), variable="station_swe", product="STATION"),
        ],
        output=out_path,
    )

    assert marker_calls == [[scf_date], [wet_date]]
    assert len(vline_calls) == 16
    scf_midday = scf_date + pd.Timedelta(hours=12)
    wet_midday = wet_date + pd.Timedelta(hours=12)
    hs_midday = hs_date + pd.Timedelta(hours=12)
    swe_midday = swe_date + pd.Timedelta(hours=12)
    assert {color for _dates, color, _ls, _lw in vline_calls} == {"#000000", "#777777"}
    assert {ls for _dates, _color, ls, _lw in vline_calls} == {"--"}
    matched_calls = [(dates, lw) for dates, color, _ls, lw in vline_calls if color == "#000000"]
    standard_calls = [(dates, lw) for dates, color, _ls, lw in vline_calls if color == "#777777"]
    assert matched_calls == [([scf_date], 1.8), ([wet_date], 1.8), ([swe_midday], 1.8), ([hs_midday], 1.8)]
    assert len(standard_calls) == 12
    assert all(lw == 1.0 for _dates, lw in standard_calls)
    assert label_calls == [
        (
            "assimilation_label_axis_0",
            [scf_date, wet_date, hs_date, swe_date],
            ["1", "2", "3", "4"],
            None,
            0.0,
            [0.35, 6.5],
            18.0,
            1.0,
            "center",
        ),
    ]
    assert out_path.is_file()


def test_plot_result_overview_lower_assimilation_label_row_stays_above_panel(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        dates = pd.date_range("2023-01-01", periods=4, freq="D")
        frame = pd.DataFrame({"date": dates, "scf": [0.2, 0.3, 0.4, 0.5]})
        out_path = tmp_path / "result_overview.png"
        plot_result_overview(
            scf_obs=frame,
            scf_model=frame,
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            assim_events=[
                plot_mod.AssimilationEvent(date=pd.Timestamp(date).date(), variable="scf", product="SNOWCOVER")
                for date in dates
            ],
            output=out_path,
        )

        fig = plt.gcf()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        main_axes = _panel_axes(fig)
        label_axes = [ax for ax in fig.axes if ax.get_label().startswith("assimilation_label_axis")]

        assert len(label_axes) == 1
        assert label_axes[0].get_label() == "assimilation_label_axis_0"
        assert label_axes[0].texts
        for text in label_axes[0].texts:
            bbox = text.get_window_extent(renderer)
            assert bbox.y0 >= main_axes[0].bbox.y1
        assert out_path.is_file()
    finally:
        original_close(plt.gcf())
        plt.close = original_close


def test_plot_result_overview_adds_assimilation_label_axis_to_top_panel_when_events_exist(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview.png"
        plot_result_overview(
            scf_obs=_frame("scf", [0.2, 0.4]),
            scf_model=_frame("scf", [0.2, 0.4]),
            wet_obs=None,
            wet_model=_frame("wet_snow_fraction", [0.1, 0.2]),
            scf_env=None,
            wet_env=None,
            output=out_path,
            assim_events=[
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-01").date(), variable="wet_snow", product="WETSNOW"),
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-02").date(), variable="scf", product="SNOWCOVER"),
            ],
        )

        label_axes = [ax for ax in plt.gcf().axes if ax.get_label().startswith("assimilation_label_axis")]
        assert len(label_axes) == 1
        assert label_axes[0].get_label() == "assimilation_label_axis_0"
        assert [text.get_text() for text in label_axes[0].texts] == ["DA 1", "DA 2"]
        assert len(_panel_axes(plt.gcf())) == 2
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_dense_assimilation_labels_fall_back_to_numbers(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        frame = pd.DataFrame({"date": dates, "scf": [0.2] * len(dates)})
        out_path = tmp_path / "result_overview.png"
        plot_result_overview(
            scf_obs=frame,
            scf_model=frame,
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            assim_events=[
                plot_mod.AssimilationEvent(date=pd.Timestamp(date).date(), variable="scf", product="SNOWCOVER")
                for date in dates
            ],
        )

        label_axes = [ax for ax in plt.gcf().axes if ax.get_label().startswith("assimilation_label_axis")]
        assert len(label_axes) == 1
        assert [text.get_text() for text in label_axes[0].texts] == [str(i) for i in range(1, 21)]
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_uses_single_figure_legend_labels(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview.png"
        plot_result_overview(
            scf_obs=_frame("scf", [0.2, 0.4]),
            scf_model=_frame("scf", [0.2, 0.4]),
            wet_obs=_frame("wet_snow_fraction", [0.1, 0.2]),
            wet_model=_frame("wet_snow_fraction", [0.1, 0.2]),
            scf_env=None,
            wet_env=None,
            output=out_path,
        )

        assert len(plt.gcf().legends) == 1
        assert _figure_legend_labels(plt.gcf()) == [
            "satellite observation",
            "open loop",
        ]
        legend_handles = plot_mod._build_result_overview_legend_handles(
            legend_state=plot_mod._ResultOverviewLegendState(
                da_observation=True,
                satellite_observation=True,
                open_loop=True,
                ensemble_summary=True,
                da_event=True,
            )
        )
        assert legend_handles[0].get_marker() == "x"
        assert legend_handles[0].get_color() == "#d62728"
        assert legend_handles[1].get_color() == "#d62728"
        assert legend_handles[-1].get_color() == "#777777"
        ensemble_handle = legend_handles[3]
        assert ensemble_handle.get_label() == "ensemble (min - max, mean)"
        assert isinstance(ensemble_handle, tuple)
        assert isinstance(ensemble_handle[0], Patch)
        assert isinstance(ensemble_handle[1], Line2D)
        assert ensemble_handle[0].get_alpha() == BAND_ALPHA
        handler_map = plot_mod._result_overview_legend_handler_map()
        ensemble_handler = handler_map[type(ensemble_handle)]
        assert isinstance(ensemble_handler, plot_mod._EnsembleLegendHandler)
        assert ensemble_handler._line_inset_frac > 0.0
        assert all(ax.get_legend() is None for ax in _panel_axes(plt.gcf()))
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_legend_adds_da_observation_only_for_matching_obs(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview.png"
        plot_result_overview(
            scf_obs=_frame("scf", [0.2, 0.4]),
            scf_model=_frame("scf", [0.2, 0.4]),
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            assim_events=[
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-01").date(), variable="scf", product="SNOWCOVER"),
            ],
        )

        assert _figure_legend_labels(plt.gcf()) == [
            "observation used for data assimilation",
            "satellite observation",
            "open loop",
            "data assimilation event",
        ]
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_legend_omits_da_observation_when_event_has_no_obs(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview.png"
        plot_result_overview(
            scf_obs=_frame("scf", [0.2, 0.4]),
            scf_model=_frame("scf", [0.2, 0.4]),
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            assim_events=[
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-02-01").date(), variable="scf", product="SNOWCOVER"),
            ],
        )

        assert _figure_legend_labels(plt.gcf()) == [
            "satellite observation",
            "open loop",
            "data assimilation event",
        ]
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_uses_shared_band_alpha_on_fraction_and_station_panels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import matplotlib.axes

    captured_alphas: list[float | None] = []
    original_fill_between = matplotlib.axes.Axes.fill_between

    def _spy_fill_between(self, x, y1, y2, *args, **kwargs):
        captured_alphas.append(kwargs.get("alpha"))
        return original_fill_between(self, x, y1, y2, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", _spy_fill_between)

    out_path = tmp_path / "result_overview_custom.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=_frame("scf", [0.2, 0.4]),
        wet_obs=None,
        wet_model=_frame("wet_snow_fraction", [0.1, 0.2]),
        scf_env=_frame("value_mean", [0.3, 0.5]).assign(value_min=[0.1, 0.3], value_max=[0.5, 0.7]),
        wet_env=_frame("value_mean", [0.2, 0.3]).assign(value_min=[0.1, 0.2], value_max=[0.3, 0.4]),
        output=out_path,
        panel_specs=[
            PanelSpec(panel="fSC"),
            PanelSpec(panel="WSF"),
            PanelSpec(panel="roi-sd"),
            PanelSpec(panel="station-sd", station_id="latschbloder"),
        ],
        roi_snow_depth_model=_frame("snow_depth", [0.3, 0.35]),
        roi_snow_depth_members=[_series([0.2, 0.3]), _series([0.4, 0.5])],
        station_panels={
            ("latschbloder", "snow_depth"): StationPanelData(
                station_id="latschbloder",
                display_name="Latschbloder",
                altitude_m=2450.0,
                open_loop=_series([0.25, 0.3]),
                members=[_series([0.2, 0.25]), _series([0.35, 0.4])],
                obs=_series([0.22, 0.32]),
            )
        },
    )

    assert captured_alphas == [BAND_ALPHA, BAND_ALPHA, BAND_ALPHA, BAND_ALPHA]
    assert out_path.is_file()


def test_plot_result_overview_custom_legend_includes_station_observation_when_drawn(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=_frame("scf", [0.2, 0.4]),
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            panel_specs=[
                PanelSpec(panel="fSC"),
                PanelSpec(panel="station-sd", station_id="latschbloder"),
            ],
            station_panels={
                ("latschbloder", "snow_depth"): StationPanelData(
                    station_id="latschbloder",
                    display_name="Latschbloder",
                    altitude_m=2919.0,
                    open_loop=_series([0.4, 0.5]),
                    members=[_series([0.3, 0.45]), _series([0.35, 0.55])],
                    obs=_series([0.32, 0.53]),
                )
            },
            strict_panels=True,
        )

        assert _figure_legend_labels(plt.gcf()) == [
            "open loop",
            "ensemble (min - max, mean)",
            "station observation",
        ]
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_custom_legend_omits_station_observation_when_hidden(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=_frame("scf", [0.2, 0.4]),
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            panel_specs=[
                PanelSpec(panel="fSC"),
                PanelSpec(panel="station-sd", station_id="latschbloder", show_obs=False),
            ],
            station_panels={
                ("latschbloder", "snow_depth"): StationPanelData(
                    station_id="latschbloder",
                    display_name="Latschbloder",
                    altitude_m=2919.0,
                    open_loop=_series([0.4, 0.5]),
                    members=[_series([0.3, 0.45]), _series([0.35, 0.55])],
                    obs=_series([0.32, 0.53]),
                )
            },
            strict_panels=True,
        )

        assert _figure_legend_labels(plt.gcf()) == [
            "open loop",
            "ensemble (min - max, mean)",
        ]
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_bottom_axis_shows_year_on_first_tick_and_year_change(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        dates = pd.to_datetime(["2022-12-01", "2023-01-01", "2023-02-01"])
        scf_model = pd.DataFrame({"date": dates, "scf": [0.2, 0.5, 0.3]})
        wet_model = pd.DataFrame({"date": dates, "wet_snow_fraction": [0.0, 0.1, 0.2]})

        out_path = tmp_path / "result_overview.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=scf_model,
            wet_obs=None,
            wet_model=wet_model,
            scf_env=None,
            wet_env=None,
            output=out_path,
        )

        tick_labels = [tick.get_text() for tick in plt.gcf().axes[-1].get_xticklabels()]
        assert "Dec\n2022" in tick_labels
        assert "Jan\n2023" in tick_labels
        assert "Feb" in tick_labels
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_honors_explicit_x_bounds(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        dates = pd.to_datetime(["2022-10-01", "2023-06-30", "2023-09-01"])
        station_obs = pd.Series([0.2, 0.5, 0.1], index=dates)

        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=_frame("scf", [0.2, 0.4]),
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            panel_specs=[
                PanelSpec(panel="fSC"),
                PanelSpec(panel="station-sd", station_id="proviantdepot"),
            ],
            station_panels={
                ("proviantdepot", "snow_depth"): StationPanelData(
                    station_id="proviantdepot",
                    display_name="Proviantdepot",
                    altitude_m=2659.0,
                    open_loop=_series([0.1, 0.2]),
                    members=[_series([0.15, 0.25])],
                    obs=station_obs,
                )
            },
            strict_panels=True,
            x_bounds=(pd.Timestamp("2022-10-01"), pd.Timestamp("2023-06-30")),
        )

        axes = _panel_axes(plt.gcf())
        left, right = axes[-1].get_xlim()
        assert pd.Timestamp(mdates.num2date(left)).tz_localize(None).date() == pd.Timestamp("2022-10-01").date()
        assert pd.Timestamp(mdates.num2date(right)).tz_localize(None).date() == pd.Timestamp("2023-06-30").date()
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_supports_custom_station_panel(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=_frame("scf", [0.2, 0.4]),
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            panel_specs=[
                PanelSpec(panel="fSC"),
                PanelSpec(panel="station-sd", station_id="latschbloder"),
            ],
            station_panels={
                ("latschbloder", "snow_depth"): StationPanelData(
                    station_id="latschbloder",
                    display_name="Latschbloder",
                    altitude_m=2919.0,
                    open_loop=_series([0.4, 0.5]),
                    members=[_series([0.3, 0.45]), _series([0.35, 0.55])],
                    obs=_series([0.32, 0.53]),
                )
            },
            strict_panels=True,
        )

        axes = _panel_axes(plt.gcf())
        assert [ax.get_ylabel() for ax in axes] == ["", ""]
        assert axes[1].get_title(loc="left").startswith("(b) Snow depth Latschbloder 2919 m")
        line_colors = [line.get_color() for line in axes[1].lines]
        assert da_variable_style("station_hs")["line"] in line_colors
        assert "black" in line_colors
        assert plot_mod.COLOR_DA_OBS == "#d62728"
        assert plot_mod.COLOR_DA_OBS in line_colors
        for line in axes[1].lines[:3]:
            assert line.get_linewidth() == pytest.approx(plot_mod._RESULT_OVERVIEW_DATA_LW)
        assert axes[1].collections
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_custom_ess_panel_uses_threshold_and_top_tick_only(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=None,
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            panel_specs=[PanelSpec(panel="ess")],
            ess_panel=plot_mod.EssPanelData(
                series=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
                        "ess": [22.0, 31.0],
                    }
                ),
                ensemble_size=47,
                threshold=23.5,
            ),
            strict_panels=True,
        )

        axes = _panel_axes(plt.gcf())
        assert axes[0].get_title(loc="left") == "(a) Effective sample size"
        assert list(axes[0].get_yticks()) == [0.0, 10.0, 20.0, 30.0, 40.0, 47.0]
        assert 23.5 not in axes[0].get_yticks()
        assert axes[0].get_ylim()[1] == 47.0
        assert axes[0].get_legend() is not None
        assert axes[0].get_legend()._loc == 1
        assert not axes[0].get_legend().get_frame().get_visible()
        assert [text.get_text() for text in axes[0].get_legend().get_texts()] == ["ESS threshold"]
        assert axes[0].get_legend().legend_handles[0].get_color() == "black"
        assert any(line.get_color() == "black" and line.get_linestyle() == "--" for line in axes[0].lines)
        assert len(plt.gcf().legends) == 0
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_supports_custom_crpss_score_panel(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=None,
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            panel_specs=[PanelSpec(panel="scores-crpss")],
            score_points=_score_points(),
            assim_events=[
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-02").date(), variable="scf", product="SNOWCOVER"),
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-03").date(), variable="station_hs", product="STATION"),
            ],
            strict_panels=True,
        )

        axes = _panel_axes(plt.gcf())
        assert len(axes) == 1
        assert axes[0].get_title(loc="left") == "(a) Continuous ranked probability skill score (CRPSS)"
        assert axes[0].get_ylabel() == ""
        assert 0.5 in list(axes[0].get_yticks())
        assert set(label.get_text() for label in axes[0].get_yticklabels() if label.get_text()) <= {"0", "0.5", "1"}
        assert axes[0].collections
        assert axes[0].get_legend() is None
        event_lines = []
        for line in axes[0].lines:
            xdata = pd.to_datetime(line.get_xdata())
            if len(xdata) >= 2 and all(ts == xdata[0] for ts in xdata):
                event_lines.append((xdata[0].normalize(), line.get_color(), line.get_linestyle(), line.get_linewidth()))
        assert (pd.Timestamp("2023-01-02"), "#777777", "--", 1.0) in event_lines
        assert (pd.Timestamp("2023-01-03"), "#777777", "--", 1.0) in event_lines
        assert len(plt.gcf().legends) == 2
        overview_labels = _figure_legend_labels(plt.gcf(), 0)
        score_labels = _figure_legend_labels(plt.gcf(), 1)
        assert overview_labels == ["data assimilation event"]
        assert "prior" in score_labels
        assert "posterior" in score_labels
        assert "assimilation fit" in score_labels
        assert getattr(plt.gcf().legends[0], "_ncols", None) == 1
        assert getattr(plt.gcf().legends[1], "_ncols", None) == 5
        plt.gcf().canvas.draw()
        renderer = plt.gcf().canvas.get_renderer()
        score_texts = {text.get_text(): text for text in plt.gcf().legends[1].get_texts() if text.get_text()}
        assert score_texts["posterior"].get_window_extent(renderer).y0 > score_texts["prior"].get_window_extent(renderer).y0
        _assert_figure_legends_clear_axes(plt.gcf())
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_load_score_points_for_custom_overview_applies_benchmark_exclusions(tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    (project_dir / "results" / "benchmark" / "scores").mkdir(parents=True)
    (project_dir / "project_2022_2023.yml").write_text(
        "\n".join(
            [
                "start_date: '2022-10-01'",
                "end_date: '2023-06-30'",
                "data_assimilation:",
                "  assimilation_events:",
                "    - date: '2023-01-02'",
                "      variable: scf",
                "      product: SNOWCOVER",
                "    - date: '2023-01-03'",
                "      variable: station_hs",
                "  benchmark:",
                "    performance_scores_exclude_variables: [station_swe]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "score_set": "analysis",
                "variable": "scf",
                "stream": "assimilation_fit",
                "representation": "prior",
                "timestamp": "2023-01-02 09:00:00",
                "date": "2023-01-02",
                "crpss": 0.10,
                "ner": 0.05,
                "zskill": float("nan"),
            },
            {
                "score_set": "analysis",
                "variable": "scf",
                "stream": "assimilation_fit",
                "representation": "posterior",
                "timestamp": "2023-01-02 09:00:00",
                "date": "2023-01-02",
                "crpss": 0.35,
                "ner": 0.22,
                "zskill": float("nan"),
            },
            {
                "score_set": "analysis",
                "variable": "station_swe",
                "stream": "semi_independent",
                "representation": "prior",
                "timestamp": "2023-01-03 09:00:00",
                "date": "2023-01-03",
                "crpss": 0.18,
                "ner": 0.11,
                "zskill": -0.08,
            },
            {
                "score_set": "analysis",
                "variable": "station_swe",
                "stream": "semi_independent",
                "representation": "posterior",
                "timestamp": "2023-01-03 09:00:00",
                "date": "2023-01-03",
                "crpss": 0.26,
                "ner": 0.16,
                "zskill": 0.24,
            },
        ]
    ).to_csv(project_dir / "results" / "benchmark" / "scores" / "event_scores.csv", index=False)

    points = plot_mod._load_score_points_for_custom_overview(project_dir)

    assert set(points["variable"]) == {"scf"}
    assert pd.Timestamp("2023-01-03") not in set(pd.to_datetime(points["assimilation_date"]))


def test_plot_result_overview_supports_custom_ner_score_panel(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=None,
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            panel_specs=[PanelSpec(panel="scores-ner")],
            score_points=_score_points(),
            assim_events=[
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-02").date(), variable="scf", product="SNOWCOVER"),
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-03").date(), variable="station_hs", product="STATION"),
            ],
            strict_panels=True,
        )

        axes = _panel_axes(plt.gcf())
        assert len(axes) == 1
        assert axes[0].get_title(loc="left").endswith("NER")
        assert axes[0].get_ylabel() == ""
        assert 0.5 in list(axes[0].get_yticks())
        assert axes[0].collections
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_supports_custom_zskill_score_panel(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=None,
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            panel_specs=[PanelSpec(panel="scores-zskill")],
            score_points=_score_points(),
            assim_events=[
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-02").date(), variable="scf", product="SNOWCOVER"),
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-03").date(), variable="station_hs", product="STATION"),
            ],
            strict_panels=True,
        )

        axes = _panel_axes(plt.gcf())
        assert len(axes) == 1
        assert axes[0].get_title(loc="left").endswith("zSkill")
        assert axes[0].get_ylabel() == ""
        assert axes[0].collections
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_supports_both_score_panels_and_single_local_legend(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=None,
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            panel_specs=[PanelSpec(panel="scores-crpss"), PanelSpec(panel="scores-ner")],
            score_points=_score_points(),
            assim_events=[
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-02").date(), variable="scf", product="SNOWCOVER"),
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-03").date(), variable="station_hs", product="STATION"),
            ],
            strict_panels=True,
        )

        axes = _panel_axes(plt.gcf())
        assert axes[0].get_title(loc="left") == "(a) Continuous ranked probability skill score (CRPSS)"
        assert axes[1].get_title(loc="left").endswith("NER")
        assert [ax.get_ylabel() for ax in axes] == ["", ""]
        assert 0.5 in list(axes[0].get_yticks())
        assert 0.5 in list(axes[1].get_yticks())
        assert axes[0].get_legend() is None
        assert axes[1].get_legend() is None
        assert len(plt.gcf().legends) == 2
        assert _figure_legend_labels(plt.gcf(), 0) == ["data assimilation event"]
        assert getattr(plt.gcf().legends[0], "_ncols", None) == 1
        assert getattr(plt.gcf().legends[1], "_ncols", None) == 5
        _assert_figure_legends_clear_axes(plt.gcf())
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_uses_uniform_panel_height_ratios(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    recorded: dict[str, object] = {}
    original_subplots = plt.subplots
    original_close = plt.close

    def _spy_subplots(nrows, *args, **kwargs):
        recorded["nrows"] = nrows
        recorded["figsize"] = kwargs.get("figsize")
        recorded["height_ratios"] = kwargs.get("gridspec_kw", {}).get("height_ratios")
        return original_subplots(nrows, *args, **kwargs)

    monkeypatch.setattr(plt, "subplots", _spy_subplots)
    monkeypatch.setattr(plt, "close", lambda fig=None: None)

    out_path = tmp_path / "result_overview_custom.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=_frame("scf", [0.2, 0.4]),
        wet_obs=None,
        wet_model=None,
        scf_env=None,
        wet_env=None,
        output=out_path,
        panel_specs=[
            PanelSpec(panel="fSC"),
            PanelSpec(panel="ess"),
            PanelSpec(panel="scores-crpss"),
            PanelSpec(panel="scores-ner"),
        ],
        score_points=_score_points(),
        assim_events=[
            plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-02").date(), variable="scf", product="SNOWCOVER"),
            plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-03").date(), variable="station_hs", product="STATION"),
        ],
        ess_panel=plot_mod.EssPanelData(
            series=pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
                    "ess": [22.0, 31.0],
                }
            ),
            ensemble_size=47,
            threshold=23.5,
        ),
        strict_panels=True,
    )

    assert recorded["nrows"] == 4
    assert recorded["height_ratios"] == [
        plot_mod.OVERVIEW_STANDARD_PANEL_HEIGHT_FACTOR,
        plot_mod.OVERVIEW_STANDARD_PANEL_HEIGHT_FACTOR,
        plot_mod.OVERVIEW_STANDARD_PANEL_HEIGHT_FACTOR,
        plot_mod.OVERVIEW_STANDARD_PANEL_HEIGHT_FACTOR,
    ]
    assert recorded["figsize"][0] == plot_mod.FIGWIDTH_OVERVIEW_PAPER
    expected_height_units = plot_mod.OVERVIEW_STANDARD_PANEL_HEIGHT_FACTOR * 4.0
    assert recorded["figsize"][1] == pytest.approx(plot_mod.FIGHEIGHT_OVERVIEW_ROW * expected_height_units)
    assert out_path.is_file()
    original_close(plt.gcf())


def test_plot_result_overview_score_panel_keeps_ess_threshold_local(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=None,
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            panel_specs=[PanelSpec(panel="ess"), PanelSpec(panel="scores-crpss")],
            score_points=_score_points(),
            assim_events=[
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-02").date(), variable="scf", product="SNOWCOVER"),
                plot_mod.AssimilationEvent(date=pd.Timestamp("2023-01-03").date(), variable="station_hs", product="STATION"),
            ],
            ess_panel=plot_mod.EssPanelData(
                series=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
                        "ess": [22.0, 31.0],
                    }
                ),
                ensemble_size=47,
                threshold=23.5,
            ),
            strict_panels=True,
        )

        axes = _panel_axes(plt.gcf())
        assert axes[0].get_legend() is not None
        assert axes[0].get_legend()._loc == 1
        assert not axes[0].get_legend().get_frame().get_visible()
        assert [text.get_text() for text in axes[0].get_legend().get_texts()] == ["ESS threshold"]
        assert axes[0].get_title(loc="left") == "(a) Effective sample size"
        assert 23.5 not in axes[0].get_yticks()
        assert axes[0].get_ylim()[1] == 47.0
        assert axes[0].get_legend().legend_handles[0].get_color() == "black"
        assert any(line.get_color() == "black" and line.get_linestyle() == "--" for line in axes[0].lines)
        assert axes[1].get_legend() is None
        assert len(plt.gcf().legends) == 2
        overview_labels = _figure_legend_labels(plt.gcf(), 0)
        assert overview_labels == ["data assimilation event"]
        assert "ESS threshold" not in overview_labels
        _assert_figure_legends_clear_axes(plt.gcf())
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_score_panel_requires_score_points() -> None:
    with pytest.raises(ValueError, match="No data available for requested panel scores-crpss"):
        plot_result_overview(
            scf_obs=None,
            scf_model=None,
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=Path("/tmp/unused.png"),
            panel_specs=[PanelSpec(panel="scores-crpss")],
            strict_panels=True,
        )


def test_plot_result_overview_shares_absolute_y_scale_between_roi_and_station_panels(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        dates = pd.to_datetime(["2023-01-01", "2023-01-02"])
        out_path = tmp_path / "result_overview_custom.png"
        plot_result_overview(
            scf_obs=None,
            scf_model=None,
            wet_obs=None,
            wet_model=None,
            scf_env=None,
            wet_env=None,
            output=out_path,
            panel_specs=[
                PanelSpec(panel="roi-swe"),
                PanelSpec(panel="station-swe", station_id="latschbloder"),
                PanelSpec(panel="roi-sd"),
                PanelSpec(panel="station-sd", station_id="latschbloder"),
            ],
            roi_swe_model=pd.DataFrame({"date": dates, "swe": [120.0, 180.0]}),
            roi_swe_members=[
                pd.Series([100.0, 160.0], index=dates),
                pd.Series([110.0, 170.0], index=dates),
            ],
            roi_snow_depth_model=pd.DataFrame({"date": dates, "snow_depth": [0.6, 1.0]}),
            roi_snow_depth_members=[
                pd.Series([0.5, 1.49], index=dates),
                pd.Series([0.55, 1.51], index=dates),
            ],
            station_panels={
                ("latschbloder", "swe"): StationPanelData(
                    station_id="latschbloder",
                    display_name="Latschbloder",
                    altitude_m=2919.0,
                    open_loop=pd.Series([40.0, 70.0], index=dates),
                    members=[pd.Series([45.0, 80.0], index=dates)],
                    obs=pd.Series([42.0, 72.0], index=dates),
                ),
                ("latschbloder", "snow_depth"): StationPanelData(
                    station_id="latschbloder",
                    display_name="Latschbloder",
                    altitude_m=2919.0,
                    open_loop=pd.Series([0.2, 0.3], index=dates),
                    members=[pd.Series([0.25, 0.35], index=dates)],
                    obs=pd.Series([0.22, 0.32], index=dates),
                ),
            },
            strict_panels=True,
        )

        axes = _panel_axes(plt.gcf())
        assert axes[0].get_ylim() == axes[1].get_ylim()
        assert axes[2].get_ylim() == axes[3].get_ylim()
        assert axes[2].get_ylim() == (0.0, 1.75)

        def _visible_in_range_y_labels(ax) -> list[str]:
            ymin, ymax = ax.get_ylim()
            return [
                tick.label1.get_text()
                for tick in ax.yaxis.get_major_ticks()
                if ymin - 1e-9 <= tick.get_loc() <= ymax + 1e-9 and tick.label1.get_visible() and tick.label1.get_text()
            ]

        assert _visible_in_range_y_labels(axes[0]) == [
            "0",
            "100",
        ]
        assert _visible_in_range_y_labels(axes[1]) == [
            "0",
            "50",
            "100",
            "150",
            "200",
        ]
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_parse_custom_panel_specs_supports_titles_and_obs_toggle(tmp_path: Path) -> None:
    cfg = tmp_path / "custom_plots.yml"
    cfg.write_text(
        "\n".join(
            [
                "panels:",
                "  - panel: fSC",
                "    show_obs: false",
                "  - panel: station-swe",
                "    station_id: proviantdepot",
                "    subtitle: custom station subtitle",
                "  - panel: scores-crpss",
                "  - panel: WSLA",
                "  - panel: scores-zskill",
            ]
        ),
        encoding="utf-8",
    )

    specs = _parse_panel_specs(cfg)

    assert specs == [
        PanelSpec(panel="fSC", title=None, show_obs=False, station_id=None),
        PanelSpec(panel="station-swe", title="custom station subtitle", show_obs=True, station_id="proviantdepot"),
        PanelSpec(panel="scores-crpss", title=None, show_obs=True, station_id=None),
        PanelSpec(panel="WSLA", title=None, show_obs=True, station_id=None),
        PanelSpec(panel="scores-zskill", title=None, show_obs=True, station_id=None),
    ]


@pytest.mark.parametrize(("old_panel", "new_panel"), [("fWS", "WSF"), ("WSL", "WSLA")])
def test_parse_custom_panel_specs_rejects_old_wet_snow_aliases(
    tmp_path: Path,
    old_panel: str,
    new_panel: str,
) -> None:
    cfg = tmp_path / "custom_plots.yml"
    cfg.write_text(f"panels:\n  - panel: {old_panel}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"Use '{new_panel}' instead"):
        _parse_panel_specs(cfg)


def test_project_custom_config_path_uses_project_root_file(tmp_path: Path) -> None:
    cfg = tmp_path / "plots.yml"
    cfg.write_text("panels: []\n", encoding="utf-8")

    assert _project_custom_config_path(tmp_path) == cfg.resolve()


def test_project_custom_config_path_returns_none_when_root_file_missing(tmp_path: Path) -> None:
    assert _project_custom_config_path(tmp_path) is None


def test_shipped_subdomain_custom_overview_uses_roi_satellite_ess_and_scores_without_station() -> None:
    cfg = Path(__file__).resolve().parents[2] / "examples/subdomains/projects/project_2022_2023/plots.yml"

    specs = _parse_panel_specs(cfg)

    assert [spec.panel for spec in specs] == ["fSC", "roi-sd", "ess", "scores-crpss"]
    assert all(spec.station_id is None for spec in specs)


def test_load_station_panel_data_falls_back_to_setup_root_obs_dir(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "setup_root"
    project_dir = root_dir / "projects" / "project_2022_2023"
    obs_dir = root_dir / "obs" / "stations"
    obs_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)

    (obs_dir / "proviantdepot.csv").write_text(
        "\n".join(
            [
                "time,snow_depth",
                "2023-01-01 00:00:00,0.40",
                "2023-01-02 00:00:00,0.55",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(plot_mod, "list_steps_sorted", lambda _project_dir: [])

    station_data = _load_station_panel_data(
        project_dir,
        "proviantdepot",
        value_col="snow_depth",
        stations_df=None,
    )

    assert station_data.obs is not None
    assert list(station_data.obs.values) == [0.40, 0.55]
