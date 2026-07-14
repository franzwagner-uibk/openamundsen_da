from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openamundsen_da.methods.viz.plots import multi_project_snow as plot_mod
from openamundsen_da.methods.viz.plots.theme import COLOR_DA_OBS, LS_STATION_OBS


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_result_files(
    root: Path,
    *,
    filename: str,
    variable: str,
    dates: list[str],
    open_loop_values: list[float],
    member_1_values: list[float],
    member_2_values: list[float],
) -> None:
    _write_csv(
        root / "open_loop" / "results" / filename,
        [{"time": date, variable: value} for date, value in zip(dates, open_loop_values, strict=True)],
    )
    _write_csv(
        root / "member_001" / "results" / filename,
        [{"time": date, variable: value} for date, value in zip(dates, member_1_values, strict=True)],
    )
    _write_csv(
        root / "member_002" / "results" / filename,
        [{"time": date, variable: value} for date, value in zip(dates, member_2_values, strict=True)],
    )


def _write_station_result_files(root: Path, *, station: str, dates: list[str]) -> None:
    rows_open = [
        {"time": dates[0], "snow_depth": 1.0, "swe": 100.0},
        {"time": dates[1], "snow_depth": 3.0, "swe": 140.0},
        {"time": dates[2], "snow_depth": 4.0, "swe": 160.0},
    ]
    rows_member_1 = [
        {"time": dates[0], "snow_depth": 0.0, "swe": 80.0},
        {"time": dates[1], "snow_depth": 2.0, "swe": 100.0},
        {"time": dates[2], "snow_depth": 3.0, "swe": 110.0},
    ]
    rows_member_2 = [
        {"time": dates[0], "snow_depth": 2.0, "swe": 120.0},
        {"time": dates[1], "snow_depth": 4.0, "swe": 160.0},
        {"time": dates[2], "snow_depth": 5.0, "swe": 180.0},
    ]
    filename = f"point_{station}.csv"
    _write_csv(root / "open_loop" / "results" / filename, rows_open)
    _write_csv(root / "member_001" / "results" / filename, rows_member_1)
    _write_csv(root / "member_002" / "results" / filename, rows_member_2)


def _build_project(setup: Path, name: str, *, start: str, end: str, day_1: str, day_2: str) -> Path:
    project_dir = setup / "projects" / name
    _write_text(
        project_dir / f"{name}.yml",
        f"start_date: '{start}'\nend_date: '{end}'\ndata_assimilation:\n  assimilation_events: []\n",
    )
    step_dir = project_dir / "steps" / f"step_00_{day_1.replace('-', '')}-{day_2.replace('-', '')}"
    _write_text(step_dir / "step.yml", f"start_date: '{start}'\nend_date: '{end}'\n")
    result_root = step_dir / "ensembles" / "prior"

    for station in ("latschbloder", "proviantdepot"):
        _write_station_result_files(
            result_root,
            station=station,
            dates=[f"{day_1} 00:00:00", f"{day_1} 12:00:00", f"{day_2} 00:00:00"],
        )

    _write_result_files(
        result_root,
        filename="point_snow_depth_roi.csv",
        variable="snow_depth",
        dates=[day_1, day_2],
        open_loop_values=[1.5, 1.6],
        member_1_values=[1.0, 1.2],
        member_2_values=[2.0, 2.2],
    )
    _write_result_files(
        result_root,
        filename="point_swe_roi.csv",
        variable="swe",
        dates=[day_1, day_2],
        open_loop_values=[300.0, 310.0],
        member_1_values=[250.0, 260.0],
        member_2_values=[350.0, 360.0],
    )
    (project_dir / "results" / "maps").mkdir(parents=True, exist_ok=True)
    (project_dir / "results" / "maps" / "setup_overview.png").write_bytes(b"fake-png")
    return project_dir


def _build_setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    setup = tmp_path / "rofental"
    _write_text(setup / "rofental.yml", "start_date: '2020-10-01'\nend_date: '2022-09-30'\n")
    project_1 = _build_project(
        setup,
        "project_2020_2021",
        start="2020-10-01",
        end="2021-09-30",
        day_1="2020-10-01",
        day_2="2020-10-02",
    )
    project_2 = _build_project(
        setup,
        "project_2021_2022",
        start="2021-10-01",
        end="2022-09-30",
        day_1="2021-10-01",
        day_2="2021-10-02",
    )
    _write_csv(
        setup / "obs" / "stations" / "latschbloder.csv",
        [
            {"time": "2020-10-01 00:00:00", "snow_depth": -0.2, "swe": ""},
            {"time": "2020-10-01 12:00:00", "snow_depth": 0.4, "swe": ""},
            {"time": "2021-10-01 00:00:00", "snow_depth": 0.8, "swe": ""},
        ],
    )
    _write_csv(
        setup / "obs" / "stations" / "proviantdepot.csv",
        [
            {"time": "2020-10-01 00:00:00", "snow_depth": 0.1, "swe": 50.0},
            {"time": "2020-10-01 12:00:00", "snow_depth": 0.3, "swe": 70.0},
            {"time": "2021-10-01 00:00:00", "snow_depth": 0.9, "swe": 90.0},
        ],
    )
    return setup, project_1, project_2


def test_cli_resolves_setup_projects_and_writes_six_plots_plus_context_map(tmp_path: Path) -> None:
    setup, _, _ = _build_setup(tmp_path)
    out = setup / "results" / "plots" / "multi_year_snow"

    rc = plot_mod.cli_main(
        [
            "--setup",
            str(setup),
            "--project",
            "project_2020_2021",
            "--project",
            "project_2021_2022",
            "--output-dir",
            str(out),
            "--log-level",
            "ERROR",
        ]
    )

    assert rc == 0
    expected = {
        "station_latschbloder_snow_depth_2020_2022.png",
        "station_latschbloder_swe_2020_2022.png",
        "station_proviantdepot_snow_depth_2020_2022.png",
        "station_proviantdepot_swe_2020_2022.png",
        "roi_mean_snow_depth_2020_2022.png",
        "roi_mean_swe_2020_2022.png",
        "context_map.png",
    }
    assert {path.name for path in out.iterdir()} == expected


def test_cli_accepts_direct_project_dirs(tmp_path: Path) -> None:
    _, project_1, project_2 = _build_setup(tmp_path)
    out = tmp_path / "plots"

    rc = plot_mod.cli_main(
        [
            "--project-dir",
            str(project_1),
            "--project-dir",
            str(project_2),
            "--output-dir",
            str(out),
            "--log-level",
            "ERROR",
        ]
    )

    assert rc == 0
    assert (out / "station_latschbloder_snow_depth_2020_2022.png").is_file()
    assert (out / "context_map.png").is_file()


def test_station_model_series_are_daily_means_and_members_are_range_ready(tmp_path: Path) -> None:
    setup, project_1, project_2 = _build_setup(tmp_path)
    start, end = plot_mod._time_bounds([project_1, project_2])

    series = plot_mod._load_snow_plot_series(
        [project_1, project_2],
        setup_root=setup,
        variable="snow_depth",
        station_id="latschbloder",
        start=start,
        end=end,
    )
    mean, lo, hi = plot_mod.envelope(series.members, q_low=0.0, q_high=1.0)

    assert series.open_loop.loc[pd.Timestamp("2020-10-01")] == pytest.approx(2.0)
    assert len(series.members) == 2
    assert mean.loc[pd.Timestamp("2020-10-01")] == pytest.approx(2.0)
    assert lo.loc[pd.Timestamp("2020-10-01")] == pytest.approx(1.0)
    assert hi.loc[pd.Timestamp("2020-10-01")] == pytest.approx(3.0)


def test_negative_observed_snow_depth_values_are_masked(tmp_path: Path) -> None:
    setup, project_1, project_2 = _build_setup(tmp_path)
    start, end = plot_mod._time_bounds([project_1, project_2])

    obs = plot_mod._load_station_observation(setup, "latschbloder", "snow_depth", start=start, end=end)

    assert obs is not None
    assert obs.loc[pd.Timestamp("2020-10-01")] == pytest.approx(0.4)
    assert (obs >= 0).all()


def test_missing_station_observations_do_not_fail_model_only_plot(tmp_path: Path) -> None:
    setup, project_1, project_2 = _build_setup(tmp_path)
    start, end = plot_mod._time_bounds([project_1, project_2])

    series = plot_mod._load_snow_plot_series(
        [project_1, project_2],
        setup_root=setup,
        variable="swe",
        station_id="latschbloder",
        start=start,
        end=end,
    )

    assert series.obs is None
    assert series.open_loop.notna().any()
    assert len(series.members) == 2


def test_station_observations_use_shared_accessible_style(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    dates = pd.to_datetime(["2022-01-01", "2022-01-02"])
    series = plot_mod.SnowPlotSeries(
        open_loop=pd.Series([0.1, 0.2], index=dates),
        members=[
            pd.Series([0.15, 0.25], index=dates),
            pd.Series([0.2, 0.3], index=dates),
        ],
        obs=pd.Series([0.12, 0.22], index=dates),
        start=dates[0],
        end=dates[-1],
    )
    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        plot_mod._plot_snow_series(
            series=series,
            variable="snow_depth",
            title="Station snow depth",
            ylabel="Snow depth [m]",
            output=tmp_path / "station.png",
            backend="Agg",
        )
        ax = plt.gcf().axes[0]
        observation_line = next(line for line in ax.lines if line.get_color() == COLOR_DA_OBS)
        assert observation_line.get_linestyle() == LS_STATION_OBS
        legend = ax.get_legend()
        legend_handles = getattr(legend, "legend_handles", None) or legend.legendHandles
        station_index = [text.get_text() for text in legend.get_texts()].index("station observation")
        station_legend_line = legend_handles[station_index]
        assert station_legend_line.get_color() == COLOR_DA_OBS
        assert station_legend_line.get_linestyle() == LS_STATION_OBS
    finally:
        plt.close = original_close
        original_close("all")


def test_overwrite_protection_fails_when_outputs_exist(tmp_path: Path) -> None:
    setup, _, _ = _build_setup(tmp_path)
    out = setup / "results" / "plots" / "multi_year_snow"
    out.mkdir(parents=True)
    (out / "station_latschbloder_snow_depth_2020_2022.png").write_text("existing", encoding="utf-8")

    rc = plot_mod.cli_main(
        [
            "--setup",
            str(setup),
            "--project",
            "project_2020_2021",
            "--project",
            "project_2021_2022",
            "--output-dir",
            str(out),
            "--log-level",
            "ERROR",
        ]
    )

    assert rc == 1
