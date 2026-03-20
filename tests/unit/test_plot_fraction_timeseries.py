from __future__ import annotations

from pathlib import Path

import pandas as pd

from openamundsen_da.methods.viz import plot_fraction_timeseries as plot_mod
from openamundsen_da.methods.viz.plot_fraction_timeseries import plot_fraction_timeseries
from openamundsen_da.methods.viz._style import COLOR_DA_STATION_HS, COLOR_DA_STATION_SWE


def _series(values: list[float]) -> pd.Series:
    dates = pd.to_datetime(["2023-01-01", "2023-01-02"])
    return pd.Series(values, index=dates)


def _frame(col: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2023-01-02"]), col: values})


def test_plot_fraction_timeseries_uses_four_panels_when_roi_series_exist(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    recorded: dict[str, int] = {}
    original_subplots = plt.subplots
    original_close = plt.close

    def _spy_subplots(nrows, *args, **kwargs):
        recorded["nrows"] = nrows
        return original_subplots(nrows, *args, **kwargs)

    monkeypatch.setattr(plt, "subplots", _spy_subplots)
    monkeypatch.setattr(plt, "close", lambda fig=None: None)

    out_path = tmp_path / "fraction_timeseries.png"
    plot_fraction_timeseries(
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
    axes = plt.gcf().axes
    assert [ax.get_ylabel() for ax in axes] == [
        "Snow cover fraction",
        "Wet snow fraction",
        "ROI mean SWE [mm]",
        "ROI mean snow depth [m]",
    ]
    assert out_path.is_file()
    original_close(plt.gcf())


def test_plot_fraction_timeseries_keeps_two_panels_without_roi_series(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    recorded: dict[str, int] = {}
    original_subplots = plt.subplots

    def _spy_subplots(nrows, *args, **kwargs):
        recorded["nrows"] = nrows
        return original_subplots(nrows, *args, **kwargs)

    monkeypatch.setattr(plt, "subplots", _spy_subplots)

    out_path = tmp_path / "fraction_timeseries.png"
    plot_fraction_timeseries(
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


def test_plot_fraction_timeseries_roi_band_excludes_open_loop_and_plots_it_separately(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import matplotlib.axes

    captured: dict[str, object] = {"bands": [], "plots": []}
    original_fill_between = matplotlib.axes.Axes.fill_between
    original_plot = matplotlib.axes.Axes.plot

    def _spy_fill_between(self, x, y1, y2, *args, **kwargs):
        captured["bands"].append((list(y1), list(y2), kwargs.get("label")))
        return original_fill_between(self, x, y1, y2, *args, **kwargs)

    def _spy_plot(self, x, y, *args, **kwargs):
        captured["plots"].append((list(pd.Series(y)), kwargs.get("label")))
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", _spy_fill_between)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

    out_path = tmp_path / "fraction_timeseries.png"
    plot_fraction_timeseries(
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

    band_low, band_high, band_label = captured["bands"][0]
    assert band_label == "5-95% band"
    assert max(band_high) < 10.0
    open_loop_calls = [vals for vals, label in captured["plots"] if label == "open loop"]
    assert open_loop_calls == [[100.0, 100.0]]
    assert out_path.is_file()


def test_plot_fraction_timeseries_marks_only_variable_specific_da_dates(monkeypatch, tmp_path: Path) -> None:
    marker_calls: list[list[pd.Timestamp]] = []
    vline_calls: list[tuple[list[pd.Timestamp], str | None]] = []
    label_calls: list[tuple[list[pd.Timestamp], list[str], str | None]] = []

    def _record_markers(ax, *, dates, **kwargs) -> None:
        marker_calls.append(list(pd.to_datetime(dates)))

    def _record_vlines(ax, dates, **kwargs) -> None:
        vline_calls.append((list(pd.to_datetime(dates)), kwargs.get("color")))

    def _record_labels(ax, dates, **kwargs) -> None:
        label_calls.append(
            (
                list(pd.to_datetime(dates)),
                list(kwargs.get("labels") or []),
                kwargs.get("color"),
            )
        )

    monkeypatch.setattr(plot_mod, "draw_assimilation_markers", _record_markers)
    monkeypatch.setattr(plot_mod, "draw_assimilation_vlines", _record_vlines)
    monkeypatch.setattr(plot_mod, "draw_assim_labels", _record_labels)

    scf_date = pd.Timestamp("2023-01-01")
    wet_date = pd.Timestamp("2023-01-02")
    hs_date = pd.Timestamp("2023-01-03")
    swe_date = pd.Timestamp("2023-01-04")

    out_path = tmp_path / "fraction_timeseries.png"
    plot_fraction_timeseries(
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
        assim_scf=[scf_date],
        assim_wet=[wet_date],
        assim_station_hs=[hs_date],
        assim_station_swe=[swe_date],
        assim_labels={
            scf_date: "1",
            wet_date: "2",
            hs_date: "3",
            swe_date: "4",
        },
        output=out_path,
    )

    assert marker_calls == [[scf_date], [wet_date]]
    assert vline_calls == [
        ([swe_date], COLOR_DA_STATION_SWE),
        ([hs_date], COLOR_DA_STATION_HS),
    ]
    assert label_calls == [
        ([scf_date], ["1"], "black"),
        ([wet_date], ["2"], "black"),
        ([swe_date], ["4"], "black"),
        ([hs_date], ["3"], "black"),
    ]
    assert out_path.is_file()
