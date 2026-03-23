from __future__ import annotations

from pathlib import Path

import pandas as pd

import openamundsen_da.methods.viz.plot_result_overview as plot_mod
from openamundsen_da.methods.viz.plot_result_overview import (
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


def _panel_axes(fig) -> list:
    return [ax for ax in fig.axes if not ax.get_label().startswith("assimilation_label_axis")]


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
    assert [ax.get_ylabel() for ax in axes] == [
        "snow cover fraction",
        "wet snow fraction",
        "swe [mm]",
        "snow depth [m]",
    ]
    assert axes[0].get_title(loc="left").startswith("(a) ")
    assert axes[1].get_title(loc="left").startswith("(b) ")
    assert axes[2].lines[0].get_color() == plot_mod._VARIABLE_STYLES["SWE"]["line"]
    assert axes[3].lines[0].get_color() == plot_mod._VARIABLE_STYLES["SD"]["line"]
    assert isinstance(axes[2].yaxis.get_major_locator(), mticker.MultipleLocator)
    assert isinstance(axes[3].yaxis.get_major_locator(), mticker.MultipleLocator)
    swe_ticks = axes[2].yaxis.get_major_locator().tick_values(0.0, 200.0)
    sd_ticks = axes[3].yaxis.get_major_locator().tick_values(0.0, 2.0)
    assert swe_ticks[1] - swe_ticks[0] == 50.0
    assert sd_ticks[1] - sd_ticks[0] == 0.25
    assert plt.gcf()._suptitle is None
    assert plt.gcf().get_size_inches()[0] == 7.2876875
    assert plt.gcf().get_size_inches()[1] == 6.8494734
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
        assert scf_ticks == [0.25, 0.5, 0.75, 1.0]
        assert wet_ticks == [0.25, 0.5, 0.75, 1.0]
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_plot_result_overview_roi_band_excludes_open_loop_and_plots_it_separately(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import matplotlib.axes

    captured: dict[str, object] = {"bands": [], "plots": []}
    original_fill_between = matplotlib.axes.Axes.fill_between
    original_plot = matplotlib.axes.Axes.plot

    def _spy_fill_between(self, x, y1, y2, *args, **kwargs):
        captured["bands"].append((list(y1), list(y2), kwargs.get("color")))
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

    band_low, band_high, band_color = captured["bands"][0]
    assert band_color == plot_mod._VARIABLE_STYLES["SWE"]["fill"]
    assert max(band_high) < 10.0
    open_loop_calls = [vals for vals, color in captured["plots"] if color == "black"]
    assert open_loop_calls == [[100.0, 100.0]]
    assert out_path.is_file()


def test_plot_result_overview_draws_all_assim_events_on_every_panel(monkeypatch, tmp_path: Path) -> None:
    marker_calls: list[list[pd.Timestamp]] = []
    vline_calls: list[tuple[list[pd.Timestamp], str | None]] = []
    label_calls: list[tuple[str, list[pd.Timestamp], list[str], object, float | None, list[float] | None, float | None, float | None, str | None]] = []

    def _record_markers(ax, *, dates, **kwargs) -> None:
        marker_calls.append(list(pd.to_datetime(dates)))

    def _record_vlines(ax, dates, **kwargs) -> None:
        vline_calls.append((list(pd.to_datetime(dates)), kwargs.get("color")))

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
    monkeypatch.setattr(plot_mod, "draw_assim_labels", _record_labels)

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
    assert sum(1 for dates, color in vline_calls if dates == [scf_date] and color == plot_mod._VARIABLE_STYLES["fSC"]["line"]) == 4
    assert sum(1 for dates, color in vline_calls if dates == [wet_date] and color == plot_mod._VARIABLE_STYLES["fWS"]["line"]) == 4
    assert sum(1 for dates, color in vline_calls if dates == [hs_date] and color == plot_mod._VARIABLE_STYLES["SD"]["line"]) == 4
    assert sum(1 for dates, color in vline_calls if dates == [swe_date] and color == plot_mod._VARIABLE_STYLES["SWE"]["line"]) == 4
    assert label_calls == [
        (
            "assimilation_label_axis_0",
            [scf_date, wet_date, hs_date, swe_date],
            ["1", "2", "3", "4"],
            None,
            0.0,
            [2.0, 8.0],
            18.0,
            1.0,
            "center",
        ),
        (
            "assimilation_label_axis_1",
            [scf_date, wet_date, hs_date, swe_date],
            ["1", "2", "3", "4"],
            None,
            0.0,
            [2.0, 8.0],
            18.0,
            1.0,
            "center",
        ),
        (
            "assimilation_label_axis_2",
            [scf_date, wet_date, hs_date, swe_date],
            ["1", "2", "3", "4"],
            None,
            0.0,
            [2.0, 8.0],
            18.0,
            1.0,
            "center",
        ),
        (
            "assimilation_label_axis_3",
            [scf_date, wet_date, hs_date, swe_date],
            ["1", "2", "3", "4"],
            None,
            0.0,
            [2.0, 8.0],
            18.0,
            1.0,
            "center",
        ),
    ]
    assert out_path.is_file()


def test_plot_result_overview_adds_assimilation_label_axis_to_each_panel_when_events_exist(tmp_path: Path) -> None:
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
        assert len(label_axes) == 2
        assert len(_panel_axes(plt.gcf())) == 2
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
        legend_labels = [text.get_text() for text in plt.gcf().legends[0].get_texts()]
        assert legend_labels == [
            "satellite observation used in DA",
            "satellite observation",
            "open loop",
            "ensemble mean",
            "station observation",
            "DA event",
        ]
        assert all(ax.get_legend() is None for ax in _panel_axes(plt.gcf()))
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
        assert [ax.get_ylabel() for ax in axes] == [
            "snow cover fraction",
            "snow depth [m]",
        ]
        assert axes[1].get_title(loc="left").startswith("(b) snow depth Latschbloder 2919 m")
        line_colors = [line.get_color() for line in axes[1].lines]
        assert plot_mod._VARIABLE_STYLES["SD"]["line"] in line_colors
        assert "black" in line_colors
        assert plot_mod.COLOR_DA_OBS in line_colors
        assert axes[1].collections
        assert out_path.is_file()
    finally:
        plt.close = original_close


def test_parse_custom_panel_specs_supports_titles_and_obs_toggle(tmp_path: Path) -> None:
    cfg = tmp_path / "custom_result_overview.yml"
    cfg.write_text(
        "\n".join(
            [
                "panels:",
                "  - panel: fSC",
                "    show_obs: false",
                "  - panel: station-swe",
                "    station_id: proviantdepot",
                "    subtitle: custom station subtitle",
            ]
        ),
        encoding="utf-8",
    )

    specs = _parse_panel_specs(cfg)

    assert specs == [
        PanelSpec(panel="fSC", title=None, show_obs=False, station_id=None),
        PanelSpec(panel="station-swe", title="custom station subtitle", show_obs=True, station_id="proviantdepot"),
    ]


def test_project_custom_config_path_uses_project_root_file(tmp_path: Path) -> None:
    cfg = tmp_path / "result_overview_custom.yml"
    cfg.write_text("panels: []\n", encoding="utf-8")

    assert _project_custom_config_path(tmp_path) == cfg.resolve()


def test_project_custom_config_path_returns_none_when_root_file_missing(tmp_path: Path) -> None:
    assert _project_custom_config_path(tmp_path) is None


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
