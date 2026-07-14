from __future__ import annotations

from pathlib import Path

import pandas as pd

import openamundsen_da.methods.viz.plots.assimilation.ess_timeline as ess_mod
import openamundsen_da.methods.viz.plots.observer.scf_summary as scf_summary_mod
from openamundsen_da.methods.viz.plots.common import (
    add_assim_label_axis,
    apply_month_interval_axis_labels,
    draw_adaptive_assim_labels,
    result_axis_scale,
)
from openamundsen_da.methods.viz.plots.result_overview import plot_result_overview


def _frame(col: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2023-01-02"]), col: values})


def test_plot_result_overview_accepts_backend_argument(tmp_path: Path) -> None:
    out_path = tmp_path / "result_overview.png"
    plot_result_overview(
        scf_obs=None,
        scf_model=_frame("scf", [0.2, 0.4]),
        wet_obs=None,
        wet_model=_frame("wet_snow_fraction", [0.1, 0.2]),
        scf_env=None,
        wet_env=None,
        output=out_path,
        backend="Agg",
    )
    assert out_path.is_file()


def test_result_axis_scale_reuses_shared_swe_and_snow_depth_rules() -> None:
    assert result_axis_scale("swe", 12.0) == (50.0, 200.0)
    assert result_axis_scale("roi-sd", 0.44) == (0.25, 1.0)
    assert result_axis_scale("station-swe", 90.0, shared=True) == (50.0, 100.0)


def test_month_interval_axis_labels_center_labels_between_month_boundaries() -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    bounds = (pd.Timestamp("2022-12-15"), pd.Timestamp("2023-02-20"))
    apply_month_interval_axis_labels(ax, bounds)

    major_tick_dates = [pd.Timestamp(mdates.num2date(value)).tz_localize(None).date() for value in ax.get_xticks(minor=False)]
    minor_tick_dates = [pd.Timestamp(mdates.num2date(value)).tz_localize(None).date() for value in ax.get_xticks(minor=True)]
    minor_labels = [label.get_text() for label in ax.get_xticklabels(minor=True)]

    assert major_tick_dates == [
        pd.Timestamp("2022-12-01").date(),
        pd.Timestamp("2023-01-01").date(),
        pd.Timestamp("2023-02-01").date(),
        pd.Timestamp("2023-03-01").date(),
    ]
    assert minor_tick_dates == [
        pd.Timestamp("2022-12-16").date(),
        pd.Timestamp("2023-01-16").date(),
        pd.Timestamp("2023-02-15").date(),
    ]
    assert minor_labels == ["Dec\n2022", "Jan\n2023", "Feb"]
    assert all(label.get_text() == "" for label in ax.get_xticklabels(minor=False))
    plt.close(fig)


def test_month_interval_axis_labels_survive_top_assimilation_label_axis() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    bounds = (pd.Timestamp("2022-10-01"), pd.Timestamp("2023-07-01"))
    ax.set_xlim(*bounds)
    apply_month_interval_axis_labels(ax, bounds)
    add_assim_label_axis(ax, pd.to_datetime(["2022-11-17", "2022-12-07"]))
    fig.canvas.draw()

    visible_minor_labels = [
        label.get_text() for label in ax.get_xticklabels(minor=True) if label.get_visible() and label.get_text()
    ]

    assert visible_minor_labels[:4] == ["Oct\n2022", "Nov", "Dec", "Jan\n2023"]
    assert all(label.get_text() == "" for label in ax.get_xticklabels(minor=False))
    plt.close(fig)


def test_plot_ess_timeline_renders_subtitle() -> None:
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "ess": [5.0, 6.0],
            "ess_norm": [0.5, 0.6],
            "n": [10, 10],
        }
    )
    fig = ess_mod._plot(
        df,
        normalized=False,
        threshold=None,
        title="ESS title",
        subtitle="Subtitle text",
        ensemble_size=10,
        backend="Agg",
    )
    texts = [text.get_text() for text in fig.texts]
    assert "Subtitle text" in texts
    plt.close(fig)


def test_observer_scf_summary_uses_centered_month_labels() -> None:
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "scf": [0.4, 0.6],
        }
    )
    fig = scf_summary_mod._plot(df, backend="Agg")
    ax = fig.axes[0]

    assert {line.get_color() for line in ax.lines} == {"#3c4f8a"}
    assert any(label.get_text() == "Jan\n2023" for label in ax.get_xticklabels(minor=True))
    assert all(label.get_text() == "" for label in ax.get_xticklabels(minor=False))
    plt.close(fig)


def test_ess_title_omits_ensemble_size_and_threshold_details() -> None:
    assert ess_mod.ess_title(ensemble_size=30, normalized=False) == "Effective sample size"
    assert ess_mod.ess_title(ensemble_size=30, normalized=True) == "Effective sample size ratio"


def test_ess_axis_ticks_use_regular_scale_with_ensemble_size_top_bound() -> None:
    assert ess_mod.ess_axis_ticks(30, threshold=21.0) == [0.0, 10.0, 20.0, 30.0]
    assert ess_mod.ess_axis_ticks(47, threshold=23.5) == [0.0, 10.0, 20.0, 30.0, 40.0, 47.0]


def test_adaptive_da_labels_use_prefix_when_sparse() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 2.0))
    dates = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    ax.set_xlim(pd.Timestamp("2022-12-15"), pd.Timestamp("2023-03-15"))
    label_axis = add_assim_label_axis(ax, dates, row_y_offsets_pts=(2.0,), min_row_spacing_days=0.0)
    fig.canvas.draw()

    assert label_axis is not None
    assert [text.get_text() for text in label_axis.texts] == ["DA 1", "DA 2", "DA 3"]
    plt.close(fig)


def test_adaptive_da_labels_fall_back_when_dense() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.8, 2.0))
    dates = pd.date_range("2023-01-01", periods=12, freq="D")
    ax.set_xlim(dates.min() - pd.Timedelta(days=1), dates.max() + pd.Timedelta(days=1))
    label_axis = add_assim_label_axis(ax, dates, row_y_offsets_pts=(2.0,), min_row_spacing_days=0.0)
    fig.canvas.draw()

    assert label_axis is not None
    assert [text.get_text() for text in label_axis.texts] == [str(i) for i in range(1, 13)]
    plt.close(fig)


def test_adaptive_da_labels_fall_back_when_prefixed_label_hits_title() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    date = pd.Timestamp("2023-01-01")
    ax.set_xlim(date - pd.Timedelta(days=1), date + pd.Timedelta(days=1))
    blocker = ax.annotate(
        "DA 1",
        xy=(date, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0.0, 3.0),
        textcoords="offset points",
        ha="center",
        va="bottom",
    )
    artists = draw_adaptive_assim_labels(
        ax,
        [date],
        labels=["1"],
        avoid_artists=[blocker],
        y_offset_pts=3.0,
        row_y_offsets_pts=(3.0,),
        min_row_spacing_days=0.0,
    )
    fig.canvas.draw()

    assert [artist.get_text() for artist in artists] == ["1"]
    plt.close(fig)


def test_plot_ess_timeline_uses_regular_y_ticks_without_threshold_label() -> None:
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "ess": [22.0, 31.0],
            "ess_norm": [22.0 / 47.0, 31.0 / 47.0],
            "n": [47, 47],
        }
    )
    fig = ess_mod._plot(
        df,
        normalized=False,
        threshold=23.5,
        title="ESS title",
        subtitle=None,
        ensemble_size=47,
        backend="Agg",
    )

    ax = fig.axes[0]
    assert ax.get_title(loc="left") == "ESS title"
    assert list(ax.get_yticks()) == [0.0, 10.0, 20.0, 30.0, 40.0, 47.0]
    assert 23.5 not in ax.get_yticks()
    assert ax.get_ylim()[1] == 47.0
    assert ax.get_legend() is not None
    assert ax.get_legend()._loc == 1
    assert not ax.get_legend().get_frame().get_visible()
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["ESS threshold"]
    assert any(line.get_color() == "black" and line.get_linestyle() == "--" for line in ax.lines)
    plt.close(fig)


def test_plot_setup_ess_timeline_uses_canonical_project_results_dir(tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    step_dir = project_dir / "steps" / "step_00_init"
    (step_dir / "assim").mkdir(parents=True, exist_ok=True)
    (step_dir / "step_00_init.yml").write_text(
        "start_date: '2023-01-01'\nend_date: '2023-01-31'\n",
        encoding="utf-8",
    )
    (project_dir / "project_2022_2023.yml").write_text(
        "\n".join(
            [
                "start_date: '2023-01-01'",
                "end_date: '2023-01-31'",
                "data_assimilation:",
                "  assimilation_events:",
                "    - date: '2023-01-15'",
                "      variable: station_hs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame({"weight": [0.6, 0.4]}).to_csv(
        step_dir / "assim" / "weights_station_hs_20230115.csv",
        index=False,
    )

    out_path = ess_mod.plot_setup_ess_timeline(project_dir)

    assert out_path == project_dir / "results" / "plots" / "assim" / "ess" / "setup_ess_timeline_2022_2023.png"
    assert out_path.is_file()
    assert not (project_dir / "plots").exists()
