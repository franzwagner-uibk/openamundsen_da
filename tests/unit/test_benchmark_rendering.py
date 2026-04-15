from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pytest

from openamundsen_da.benchmark.aggregate import aggregate_scores, build_case_scores, enrich_case_scores, reliability_rows
from openamundsen_da.benchmark.cases import RawBenchmarkCase
from openamundsen_da.benchmark.pipeline import load_benchmark_config
from openamundsen_da.benchmark.render.plots import core as plots_core
from openamundsen_da.benchmark.render.plots.core import build_event_skill_plot_data, compute_event_skill_plot_positions
from openamundsen_da.benchmark.render.plots import write_plots
from openamundsen_da.io.paths import project_plot_assim_scores_dir
from openamundsen_da.methods.viz._style import da_variable_style
from openamundsen_da.methods.viz._utils import CRPSS_AXIS_POLICY, bounded_metric_range
from openamundsen_da.benchmark.render.tables import write_summary_tables


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def _setup_render_project(tmp_path: Path) -> tuple[Path, Path]:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    _write_yaml(
        setup_dir / "setup.yml",
        """
        resolution: 100
        """,
    )
    _write_yaml(
        project_dir / "project_2022_2023.yml",
        """
        start_date: '2023-01-01'
        end_date: '2023-01-05'
        data_assimilation:
          assimilation_events:
            - date: '2023-01-02'
              variable: scf
              product: SNOWCOVER
            - date: '2023-01-03'
              variable: station_hs
              product: station_hs
          benchmark:
            independent_variables:
              - station_swe
            figure_tier: extended
        """,
    )
    return setup_dir, project_dir


def test_load_benchmark_config_ignores_obsolete_figure_tier(tmp_path: Path) -> None:
    _, project_dir = _setup_render_project(tmp_path)
    cfg = load_benchmark_config(project_dir)

    assert cfg.plots is True
    assert cfg.independent_variables == ("station_swe",)
    assert cfg.output_dir == project_dir / "results" / "benchmark"


def test_build_event_skill_plot_data_reduces_station_same_day_rows(tmp_path: Path) -> None:
    _, project_dir = _setup_render_project(tmp_path)
    event_scores = pd.DataFrame(
        [
            {
                "score_set": "analysis",
                "variable": "scf",
                "stream": "assimilation_fit",
                "representation": "prior",
                "timestamp": "2023-01-02 00:00:00",
                "date": "2023-01-02",
                "crpss": 0.25,
                "ner": 0.10,
                "zskill": np.nan,
            },
            {
                "score_set": "analysis",
                "variable": "scf",
                "stream": "assimilation_fit",
                "representation": "posterior",
                "timestamp": "2023-01-02 00:00:00",
                "date": "2023-01-02",
                "crpss": 0.55,
                "ner": 0.40,
                "zskill": np.nan,
            },
            {
                "score_set": "analysis",
                "variable": "station_hs",
                "stream": "semi_independent",
                "representation": "prior",
                "timestamp": "2023-01-02 00:00:00",
                "date": "2023-01-02",
                "crpss": 0.20,
                "ner": 0.10,
                "zskill": -0.20,
            },
            {
                "score_set": "analysis",
                "variable": "station_hs",
                "stream": "semi_independent",
                "representation": "prior",
                "timestamp": "2023-01-02 12:00:00",
                "date": "2023-01-02",
                "crpss": 0.40,
                "ner": 0.30,
                "zskill": 0.10,
            },
            {
                "score_set": "analysis",
                "variable": "station_hs",
                "stream": "semi_independent",
                "representation": "posterior",
                "timestamp": "2023-01-02 00:00:00",
                "date": "2023-01-02",
                "crpss": 0.35,
                "ner": 0.28,
                "zskill": -0.05,
            },
            {
                "score_set": "analysis",
                "variable": "station_hs",
                "stream": "semi_independent",
                "representation": "posterior",
                "timestamp": "2023-01-02 12:00:00",
                "date": "2023-01-02",
                "crpss": 0.45,
                "ner": 0.36,
                "zskill": 0.20,
            },
            {
                "score_set": "analysis",
                "variable": "wet_snow",
                "stream": "independent",
                "representation": "prior",
                "timestamp": "2023-01-04 12:00:00",
                "date": "2023-01-04",
                "crpss": 0.90,
                "ner": 0.90,
                "zskill": np.nan,
            },
        ]
    )

    points = build_event_skill_plot_data(event_scores, project_dir=project_dir)

    assert set(points["point_type"]) == {"prior", "posterior"}
    assert pd.Timestamp("2023-01-04") not in set(pd.to_datetime(points["assimilation_date"]))

    station_prior = points[
        (points["variable"] == "station_hs")
        & (points["stream"] == "semi_independent")
        & (points["point_type"] == "prior")
        & (pd.to_datetime(points["assimilation_date"]) == pd.Timestamp("2023-01-02"))
    ]
    station_posterior = points[
        (points["variable"] == "station_hs")
        & (points["stream"] == "semi_independent")
        & (points["point_type"] == "posterior")
        & (pd.to_datetime(points["assimilation_date"]) == pd.Timestamp("2023-01-02"))
    ]
    assert len(station_prior) == 1
    assert len(station_posterior) == 1
    assert float(station_prior["crpss"].iloc[0]) == pytest.approx(0.30)
    assert float(station_prior["ner"].iloc[0]) == pytest.approx(0.20)
    assert float(station_prior["zskill"].iloc[0]) == pytest.approx(-0.05)
    assert float(station_posterior["crpss"].iloc[0]) == pytest.approx(0.40)
    assert float(station_posterior["ner"].iloc[0]) == pytest.approx(0.32)
    assert float(station_posterior["zskill"].iloc[0]) == pytest.approx(0.075)


def test_benchmark_render_outputs_write_single_plot_and_curated_tables(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_render_project(tmp_path)
    raw_cases = [
        RawBenchmarkCase(
            score_set="continuous",
            variable="scf",
            stream="assimilation_fit",
            timestamp="2023-01-02",
            obs_id="roi",
            step_name="step_00_init",
            obs_value=0.4,
            open_loop_value=0.2,
            da_informed_values=(0.3, 0.45, 0.5),
            prior_values=None,
            posterior_values=None,
            posterior_weights=None,
        ),
        RawBenchmarkCase(
            score_set="analysis",
            variable="scf",
            stream="assimilation_fit",
            timestamp="2023-01-02",
            obs_id="roi",
            step_name="step_00_init",
            obs_value=0.7,
            open_loop_value=0.35,
            da_informed_values=None,
            prior_values=(0.45, 0.55, 0.65),
            posterior_values=(0.55, 0.68, 0.8),
            posterior_weights=(0.2, 0.5, 0.3),
        ),
        RawBenchmarkCase(
            score_set="continuous",
            variable="wet_snow",
            stream="independent",
            timestamp="2023-01-02",
            obs_id="roi",
            step_name="step_00_init",
            obs_value=0.4,
            open_loop_value=0.2,
            da_informed_values=(0.3, 0.45, 0.5),
            prior_values=None,
            posterior_values=None,
            posterior_weights=None,
        ),
        RawBenchmarkCase(
            score_set="analysis",
            variable="wet_snow",
            stream="independent",
            timestamp="2023-01-02",
            obs_id="roi",
            step_name="step_00_init",
            obs_value=0.4,
            open_loop_value=0.2,
            da_informed_values=None,
            prior_values=(0.15, 0.25, 0.35),
            posterior_values=(0.15, 0.25, 0.35),
            posterior_weights=(0.3, 0.4, 0.3),
        ),
        RawBenchmarkCase(
            score_set="continuous",
            variable="station_hs",
            stream="assimilation_fit",
            timestamp="2023-01-03 00:00:00",
            obs_id="station_a",
            step_name="step_00_init",
            obs_value=1.05,
            open_loop_value=1.2,
            da_informed_values=(1.0, 1.08, 1.12),
            prior_values=None,
            posterior_values=None,
            posterior_weights=None,
            sigma_base=0.20,
        ),
        RawBenchmarkCase(
            score_set="continuous",
            variable="station_hs",
            stream="assimilation_fit",
            timestamp="2023-01-03 12:00:00",
            obs_id="station_a",
            step_name="step_00_init",
            obs_value=1.15,
            open_loop_value=1.3,
            da_informed_values=(1.02, 1.10, 1.18),
            prior_values=None,
            posterior_values=None,
            posterior_weights=None,
            sigma_base=0.20,
        ),
        RawBenchmarkCase(
            score_set="analysis",
            variable="station_hs",
            stream="assimilation_fit",
            timestamp="2023-01-03 06:00:00",
            obs_id="station_a",
            step_name="step_00_init",
            obs_value=1.1,
            open_loop_value=1.35,
            da_informed_values=None,
            prior_values=(1.22, 1.18, 1.14),
            posterior_values=(1.14, 1.11, 1.08),
            posterior_weights=(0.2, 0.4, 0.4),
            sigma_base=0.20,
        ),
        RawBenchmarkCase(
            score_set="continuous",
            variable="station_swe",
            stream="semi_independent",
            timestamp="2023-01-03 09:00:00",
            obs_id="station_a",
            step_name="step_00_init",
            obs_value=250.0,
            open_loop_value=290.0,
            da_informed_values=(230.0, 240.0, 255.0),
            prior_values=None,
            posterior_values=None,
            posterior_weights=None,
            sigma_base=15.0,
        ),
        RawBenchmarkCase(
            score_set="analysis",
            variable="station_swe",
            stream="semi_independent",
            timestamp="2023-01-03 09:00:00",
            obs_id="station_a",
            step_name="step_00_init",
            obs_value=250.0,
            open_loop_value=290.0,
            da_informed_values=None,
            prior_values=(235.0, 245.0, 260.0),
            posterior_values=(235.0, 245.0, 260.0),
            posterior_weights=(0.2, 0.5, 0.3),
            sigma_base=15.0,
        ),
    ]
    case_scores = build_case_scores(raw_cases)
    case_scores = enrich_case_scores(case_scores, project_dir=project_dir, setup_dir=setup_dir)
    event_scores = aggregate_scores(
        case_scores,
        group_cols=("score_set", "variable", "stream", "step_name", "timestamp", "date"),
    )
    project_scores = aggregate_scores(
        case_scores,
        group_cols=("score_set", "variable", "stream"),
    )
    reliability = reliability_rows(case_scores, group_cols=("score_set", "variable", "stream"))

    table_outputs, tables = write_summary_tables(
        project_dir / "results" / "benchmark",
        event_scores=event_scores,
        project_scores=project_scores,
        reliability=reliability,
    )
    legacy_results_plot_dir = project_dir / "results" / "benchmark" / "plots"
    legacy_results_plot_dir.mkdir(parents=True, exist_ok=True)
    (legacy_results_plot_dir / "performance_scores.png").write_text("stale", encoding="utf-8")
    legacy_project_plot_dir = project_dir / "plots" / "benchmark"
    legacy_project_plot_dir.mkdir(parents=True, exist_ok=True)

    plot_outputs = write_plots(
        project_plot_assim_scores_dir(project_dir),
        case_scores=case_scores,
        event_scores=event_scores,
        reliability=reliability,
        project_dir=project_dir,
    )

    tables_dir = project_dir / "results" / "benchmark" / "tables"
    plot_path = project_plot_assim_scores_dir(project_dir) / "performance_scores.png"

    assert (tables_dir / "project_summary.csv").is_file()
    assert (tables_dir / "update_summary.csv").is_file()
    assert not (tables_dir / "project_summary_wide.csv").exists()
    assert not (tables_dir / "event_summary_wide.csv").exists()
    assert not (tables_dir / "reliability_summary_wide.csv").exists()
    assert not (tables_dir / "improvement_summary.csv").exists()
    assert not any(tables_dir.glob("*.md"))

    assert plot_path.is_file()
    assert not legacy_results_plot_dir.exists()
    assert not legacy_project_plot_dir.exists()
    assert not (project_dir / "plots").exists()

    project_summary = tables["project_summary"]
    update_summary = tables["update_summary"]
    assert list(project_summary.columns) == [
        "variable",
        "stream",
        "n_project_points",
        "whole_project_crpss",
        "whole_project_ner",
        "whole_project_zskill",
        "whole_project_bias",
        "n_update_dates",
        "update_prior_crpss",
        "update_posterior_crpss",
        "update_prior_ner",
        "update_posterior_ner",
        "update_prior_zskill",
        "update_posterior_zskill",
        "update_prior_bias",
        "update_posterior_bias",
    ]
    assert list(update_summary.columns) == [
        "assimilation_date",
        "variable",
        "stream",
        "prior_crpss",
        "posterior_crpss",
        "prior_ner",
        "posterior_ner",
        "prior_zskill",
        "posterior_zskill",
        "prior_bias",
        "posterior_bias",
        "delta_crpss",
        "delta_ner",
        "delta_zskill",
        "delta_abs_bias",
    ]

    station_swe_row = project_summary[
        (project_summary["variable"] == "station_swe") & (project_summary["stream"] == "semi_independent")
    ].iloc[0]
    assert not pd.isna(station_swe_row["update_prior_crpss"])
    assert not pd.isna(station_swe_row["update_posterior_crpss"])
    assert not pd.isna(station_swe_row["whole_project_zskill"])
    assert not pd.isna(station_swe_row["update_posterior_zskill"])

    scf_update = update_summary[
        (update_summary["variable"] == "scf") & (update_summary["stream"] == "assimilation_fit")
    ].iloc[0]
    assert float(scf_update["posterior_crpss"]) > float(scf_update["prior_crpss"])
    wet_snow_update = update_summary[
        (update_summary["variable"] == "wet_snow") & (update_summary["stream"] == "independent")
    ].iloc[0]
    assert wet_snow_update["stream"] == "independent"

    assert "project_summary" in table_outputs
    assert "update_summary" in table_outputs
    assert "performance_scores" in plot_outputs


def test_write_plots_adds_zskill_third_panel_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.pyplot as plt

    _, project_dir = _setup_render_project(tmp_path)
    event_scores = pd.DataFrame(
        [
            {
                "score_set": "analysis",
                "variable": "scf",
                "stream": "assimilation_fit",
                "representation": "prior",
                "timestamp": "2023-01-02 00:00:00",
                "date": "2023-01-02",
                "crpss": 0.25,
                "ner": 0.10,
                "zskill": np.nan,
            },
            {
                "score_set": "analysis",
                "variable": "scf",
                "stream": "assimilation_fit",
                "representation": "posterior",
                "timestamp": "2023-01-02 00:00:00",
                "date": "2023-01-02",
                "crpss": 0.55,
                "ner": 0.40,
                "zskill": np.nan,
            },
            {
                "score_set": "analysis",
                "variable": "station_hs",
                "stream": "assimilation_fit",
                "representation": "prior",
                "timestamp": "2023-01-03 00:00:00",
                "date": "2023-01-03",
                "crpss": 0.20,
                "ner": 0.10,
                "zskill": -0.15,
            },
            {
                "score_set": "analysis",
                "variable": "station_hs",
                "stream": "assimilation_fit",
                "representation": "posterior",
                "timestamp": "2023-01-03 00:00:00",
                "date": "2023-01-03",
                "crpss": 0.35,
                "ner": 0.28,
                "zskill": 0.22,
            },
        ]
    )
    recorded: dict[str, object] = {}
    original_subplots = plt.subplots

    def _spy_subplots(nrows, *args, **kwargs):
        recorded["nrows"] = nrows
        recorded["figsize"] = kwargs.get("figsize")
        return original_subplots(nrows, *args, **kwargs)

    monkeypatch.setattr(plt, "subplots", _spy_subplots)

    outputs = write_plots(
        project_dir / "results" / "benchmark" / "plots",
        event_scores=event_scores,
        project_dir=project_dir,
    )

    assert outputs["performance_scores"].is_file()
    assert recorded["nrows"] == 3
    assert recorded["figsize"][1] == pytest.approx(
        plots_core.FIGHEIGHT_OVERVIEW_ROW * plots_core.STANDALONE_SCORE_FIGURE_ROW_UNITS * 1.5
    )


def test_write_plots_can_hide_station_swe_only_from_performance_scores_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, project_dir = _setup_render_project(tmp_path)
    event_scores = pd.DataFrame(
        [
            {
                "score_set": "analysis",
                "variable": "scf",
                "stream": "assimilation_fit",
                "representation": "prior",
                "timestamp": "2023-01-02 00:00:00",
                "date": "2023-01-02",
                "crpss": 0.25,
                "ner": 0.10,
            },
            {
                "score_set": "analysis",
                "variable": "scf",
                "stream": "assimilation_fit",
                "representation": "posterior",
                "timestamp": "2023-01-02 00:00:00",
                "date": "2023-01-02",
                "crpss": 0.55,
                "ner": 0.40,
            },
            {
                "score_set": "analysis",
                "variable": "station_swe",
                "stream": "semi_independent",
                "representation": "prior",
                "timestamp": "2023-01-03 09:00:00",
                "date": "2023-01-03",
                "crpss": 0.20,
                "ner": 0.12,
            },
            {
                "score_set": "analysis",
                "variable": "station_swe",
                "stream": "semi_independent",
                "representation": "posterior",
                "timestamp": "2023-01-03 09:00:00",
                "date": "2023-01-03",
                "crpss": 0.28,
                "ner": 0.18,
            },
        ]
    )
    captured: list[list[str]] = []
    original_score_legend_handles = plots_core.score_legend_handles

    def _capture_score_legend_handles(variables: list[str], *, include_da_event: bool = True):
        captured.append(list(variables))
        return original_score_legend_handles(variables, include_da_event=include_da_event)

    monkeypatch.setattr(plots_core, "score_legend_handles", _capture_score_legend_handles)

    outputs = write_plots(
        project_dir / "results" / "benchmark" / "plots",
        event_scores=event_scores,
        project_dir=project_dir,
        exclude_variables=("station_swe",),
    )

    assert outputs["performance_scores"].is_file()
    assert "station_swe" in set(event_scores["variable"])
    assert captured == [["scf"]]


def test_event_skill_plot_positions_distinguish_same_date_markers(tmp_path: Path) -> None:
    _, project_dir = _setup_render_project(tmp_path)
    points = pd.DataFrame(
        [
            {
                "variable": "scf",
                "stream": "assimilation_fit",
                "assimilation_date": "2023-01-02",
                "point_type": "prior",
                "crpss": 0.10,
                "ner": 0.05,
            },
            {
                "variable": "scf",
                "stream": "assimilation_fit",
                "assimilation_date": "2023-01-02",
                "point_type": "posterior",
                "crpss": 0.30,
                "ner": 0.25,
            },
            {
                "variable": "station_hs",
                "stream": "semi_independent",
                "assimilation_date": "2023-01-02",
                "point_type": "prior",
                "crpss": 0.20,
                "ner": 0.15,
            },
            {
                "variable": "wet_snow",
                "stream": "independent",
                "assimilation_date": "2023-01-02",
                "point_type": "posterior",
                "crpss": -0.10,
                "ner": -0.05,
            },
        ]
    )

    positioned = compute_event_skill_plot_positions(
        points,
        assimilation_dates=[pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")],
    )

    assert positioned["plot_x"].nunique() == 4
    assert all(pd.Timestamp("2023-01-02") != ts for ts in pd.to_datetime(positioned["plot_x"]))


def test_write_plots_trims_to_da_window_and_drops_subtitle(tmp_path: Path, monkeypatch) -> None:
    setup_dir, project_dir = _setup_render_project(tmp_path)
    raw_cases = [
        RawBenchmarkCase(
            score_set="analysis",
            variable="scf",
            stream="assimilation_fit",
            timestamp="2023-01-02",
            obs_id="roi",
            step_name="step_00_init",
            obs_value=0.7,
            open_loop_value=0.35,
            da_informed_values=None,
            prior_values=(0.45, 0.55, 0.65),
            posterior_values=(0.55, 0.68, 0.8),
            posterior_weights=(0.2, 0.5, 0.3),
        ),
        RawBenchmarkCase(
            score_set="analysis",
            variable="station_swe",
            stream="semi_independent",
            timestamp="2023-01-03 09:00:00",
            obs_id="station_a",
            step_name="step_00_init",
            obs_value=250.0,
            open_loop_value=290.0,
            da_informed_values=None,
            prior_values=(230.0, 240.0, 255.0),
            posterior_values=(230.0, 240.0, 255.0),
            posterior_weights=(0.2, 0.5, 0.3),
        ),
    ]
    case_scores = build_case_scores(raw_cases)
    case_scores = enrich_case_scores(case_scores, project_dir=project_dir, setup_dir=setup_dir)
    event_scores = aggregate_scores(
        case_scores,
        group_cols=("score_set", "variable", "stream", "step_name", "timestamp", "date"),
    )
    reliability = reliability_rows(case_scores, group_cols=("score_set", "variable", "stream"))

    captured: dict[str, object] = {}

    def _capture(fig, out_path, **kwargs):
        captured["fig"] = fig
        captured["path"] = out_path
        captured["kwargs"] = kwargs
        fig.savefig(out_path, **kwargs)

    monkeypatch.setattr(plots_core, "save_figure_png", _capture)

    outputs = write_plots(
        project_dir / "results" / "benchmark" / "plots",
        case_scores=case_scores,
        event_scores=event_scores,
        reliability=reliability,
        project_dir=project_dir,
    )

    fig = captured["fig"]
    assert outputs["performance_scores"].is_file()
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "Data assimilation performance scores"
    assert fig._suptitle.get_ha() == "left"
    assert fig.get_size_inches()[0] == plots_core.FIGWIDTH_OVERVIEW_PAPER
    assert fig.get_size_inches()[1] == pytest.approx(
        plots_core.FIGHEIGHT_OVERVIEW_ROW * plots_core.STANDALONE_SCORE_FIGURE_ROW_UNITS
    )
    assert captured["kwargs"] == {}
    label_axes = [ax for ax in fig.axes if ax.get_label().startswith("assimilation_label_axis_")]
    main_axes = [ax for ax in fig.axes if not ax.get_label().startswith("assimilation_label_axis_")]
    assert len(label_axes) == 2
    assert len(main_axes) == 2
    ax_crpss, ax_ner = main_axes
    x_title, y_title = fig._suptitle.get_position()
    assert x_title < ax_crpss.get_position().x0
    assert y_title < 1.0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bbox = fig._suptitle.get_window_extent(renderer)
    assert fig.bbox.y1 - title_bbox.y1 < 20.0
    for ax in label_axes:
        for text in ax.texts:
            assert not title_bbox.overlaps(text.get_window_extent(renderer))
    assert ax_crpss.get_title() == ""
    assert ax_ner.get_title() == ""
    assert ax_ner.get_xlabel() == ""
    assert ax_crpss.get_ylabel() == "CRPSS"
    assert ax_ner.get_ylabel() == "NER"
    crpss_ticks = [tick for tick in ax_crpss.get_yticks() if ax_crpss.get_ylim()[0] - 1e-9 <= tick <= ax_crpss.get_ylim()[1] + 1e-9]
    assert len(crpss_ticks) >= 3
    crpss_step = float(np.diff(crpss_ticks)[0])
    assert any(crpss_step == pytest.approx(candidate) for candidate in CRPSS_AXIS_POLICY.preferred_steps)
    crpss_data = np.concatenate([collection.get_offsets()[:, 1] for collection in ax_crpss.collections])
    assert float(np.min(crpss_data)) > 0.0
    expected_lower = np.floor(float(np.min(crpss_data)) / crpss_step) * crpss_step
    expected_upper = min(
        CRPSS_AXIS_POLICY.upper_cap if CRPSS_AXIS_POLICY.upper_cap is not None else np.inf,
        np.ceil(float(np.max(crpss_data)) / crpss_step) * crpss_step,
    )
    assert ax_crpss.get_ylim()[0] == pytest.approx(expected_lower)
    assert ax_crpss.get_ylim()[1] == pytest.approx(expected_upper)
    assert ax_crpss.get_ylim()[0] >= 0.0
    assert 0.5 in list(ax_ner.get_yticks())
    x0, x1 = ax_crpss.get_xlim()
    first = plots_core.mdates.date2num(pd.Timestamp("2023-01-02"))
    last = plots_core.mdates.date2num(pd.Timestamp("2023-01-03"))
    assert x0 < first < last < x1
    assert x0 <= plots_core.mdates.date2num(pd.Timestamp("2022-12-31"))
    assert x1 >= plots_core.mdates.date2num(pd.Timestamp("2023-01-05"))
    assert any("\n2023" in label.get_text() for label in ax_ner.get_xticklabels())
    assert len(ax_crpss.collections) > 0
    assert len(ax_ner.collections) > 0
    assert ax_crpss.lines
    assert max(line.get_zorder() for line in ax_crpss.lines) < min(c.get_zorder() for c in ax_crpss.collections)
    vline_colors = []
    for line in ax_crpss.lines:
        xdata = pd.to_datetime(line.get_xdata())
        if len(xdata) >= 2 and all(ts == xdata[0] for ts in xdata):
            vline_colors.append(mcolors.to_hex(line.get_color()).lower())
    assert plots_core.variable_style("scf")["line"].lower() in vline_colors
    assert plots_core.variable_style("station_hs")["line"].lower() in vline_colors
    assert {text.get_text() for ax in label_axes for text in ax.texts} >= {"1", "2"}
    assert fig.subplotpars.bottom < 0.25
    saw_prior = False
    saw_posterior = False
    for collection in (*ax_crpss.collections, *ax_ner.collections):
        facecolors = collection.get_facecolors()
        edgecolors = collection.get_edgecolors()
        assert facecolors.size > 0
        assert edgecolors.size > 0
        if np.allclose(facecolors[:, :3], np.array([[1.0, 1.0, 1.0]])):
            saw_prior = True
            assert not np.allclose(edgecolors[:, :3], np.array([[0.0, 0.0, 0.0]]))
        else:
            saw_posterior = True
            assert not np.allclose(edgecolors[:, :3], np.array([[0.0, 0.0, 0.0]]))
            assert np.allclose(edgecolors[:, :3], facecolors[:, :3])
    assert saw_prior
    assert saw_posterior
    legend_handles = plots_core.score_legend_handles(["scf", "wet_snow"])
    assert legend_handles[0].get_markeredgecolor() == plots_core.score_variable_color("scf")
    assert legend_handles[0].get_markerfacecolor() == plots_core.score_variable_color("scf")
    assert plots_core.score_variable_color("station_hs") == da_variable_style("station_hs")["line"]
    assert plots_core.score_variable_color("station_swe") == da_variable_style("station_swe")["line"]
    assert plots_core.variable_style("station_hs")["line"] == da_variable_style("station_hs")["line"]
    assert plots_core.variable_style("station_swe")["line"] == da_variable_style("station_swe")["line"]
    handle_by_label = {handle.get_label(): handle for handle in legend_handles if handle.get_label()}
    prior_handle = handle_by_label["prior"]
    posterior_handle = handle_by_label["posterior"]
    assert prior_handle.get_label() == "prior"
    assert posterior_handle.get_label() == "posterior"
    assert isinstance(prior_handle, tuple)
    assert isinstance(posterior_handle, tuple)
    assert [artist.get_marker() for artist in prior_handle] == ["o", "s", "^"]
    assert [artist.get_markerfacecolor() for artist in prior_handle] == ["white", "white", "white"]
    assert [artist.get_markeredgecolor() for artist in prior_handle] == ["#000000", "#000000", "#000000"]
    assert [artist.get_marker() for artist in posterior_handle] == ["o", "s", "^"]
    assert [artist.get_markerfacecolor() for artist in posterior_handle] == ["#000000", "#000000", "#000000"]
    assert [artist.get_markeredgecolor() for artist in posterior_handle] == ["#000000", "#000000", "#000000"]
    handler_map = plots_core.score_legend_handler_map()
    stage_handler = handler_map[type(prior_handle)]
    assert isinstance(stage_handler, plots_core._StageLegendHandler)
    assert tuple(stage_handler._x_fracs) == (0.04, 0.5, 0.96)
    grid_lines = [line for line in ax_crpss.get_xgridlines() if line.get_visible()]
    assert grid_lines
    assert grid_lines[0].get_linestyle() == "--"
    assert grid_lines[0].get_linewidth() == pytest.approx(0.8)
    assert fig.legends
    legend = fig.legends[0]
    assert getattr(legend, "_loc", None) == 3
    assert getattr(legend, "_mode", None) == "expand"
    legend_texts = {text.get_text(): text for text in legend.get_texts() if text.get_text()}
    assert legend_texts["posterior"].get_window_extent(renderer).y0 > legend_texts["prior"].get_window_extent(renderer).y0


@pytest.mark.parametrize(
    ("variables", "include_da_event"),
    [
        (["scf", "wet_snow"], True),
        (["scf", "wet_snow", "station_hs"], True),
        (["scf", "wet_snow", "station_hs", "station_swe"], True),
        (["scf", "station_swe"], False),
    ],
)
def test_score_legend_handles_keep_posterior_above_prior_for_supported_variable_counts(
    variables: list[str],
    include_da_event: bool,
) -> None:
    rows = plots_core._score_legend_display_rows(variables, include_da_event=include_da_event)
    stage_col = (len(variables) + 1) // 2
    assert rows[0][stage_col].get_label() == "posterior"
    assert rows[1][stage_col].get_label() == "prior"

    labels = [handle.get_label() for handle in plots_core.score_legend_handles(variables, include_da_event=include_da_event)]
    posterior_idx = labels.index("posterior")
    prior_idx = labels.index("prior")
    assert posterior_idx % 2 == 0
    assert prior_idx == posterior_idx + 1


def test_bounded_metric_range_rounds_rofental_like_crpss_to_quarter_steps() -> None:
    lower, upper, step = bounded_metric_range([-0.056, 0.903], policy=CRPSS_AXIS_POLICY)

    assert (lower, upper, step) == pytest.approx((-0.25, 1.0, 0.25))


@pytest.mark.parametrize("values", ([0.83, 0.97], [0.31, 0.42, 0.57], [0.61, 0.62, 0.63]))
def test_bounded_metric_range_keeps_positive_only_crpss_nonnegative(values) -> None:
    lower, upper, step = bounded_metric_range(values, policy=CRPSS_AXIS_POLICY)

    clipped_min = min(values)
    clipped_max = max(values)
    if CRPSS_AXIS_POLICY.lower_cap is not None:
        clipped_min = max(clipped_min, CRPSS_AXIS_POLICY.lower_cap)
    if CRPSS_AXIS_POLICY.upper_cap is not None:
        clipped_max = min(clipped_max, CRPSS_AXIS_POLICY.upper_cap)
    assert lower <= clipped_min
    assert upper >= clipped_max
    assert lower >= 0.0
    assert upper <= CRPSS_AXIS_POLICY.upper_cap
    assert any(step == pytest.approx(candidate) for candidate in CRPSS_AXIS_POLICY.preferred_steps)
    assert (upper - lower) / step >= CRPSS_AXIS_POLICY.min_intervals - 1e-9
    assert (upper - lower) / step <= CRPSS_AXIS_POLICY.max_intervals + 1e-9


def test_bounded_metric_range_keeps_negative_only_crpss_nonpositive() -> None:
    lower, upper, step = bounded_metric_range([-0.72, -0.31, -0.15], policy=CRPSS_AXIS_POLICY)

    assert lower <= -0.72
    assert upper >= -0.15
    assert upper <= 0.0
    assert step == pytest.approx(0.25)


def test_bounded_metric_range_expands_narrow_crpss_ranges_to_readable_span() -> None:
    lower, upper, step = bounded_metric_range([0.61, 0.62, 0.63], policy=CRPSS_AXIS_POLICY)

    assert lower <= 0.61
    assert upper >= 0.63
    assert (upper - lower) / step >= CRPSS_AXIS_POLICY.min_intervals - 1e-9
    assert upper <= CRPSS_AXIS_POLICY.upper_cap


def test_bounded_metric_range_uses_larger_steps_for_wide_future_ranges() -> None:
    lower, upper, step = bounded_metric_range([-3.2, 0.9], policy=CRPSS_AXIS_POLICY)

    assert step == pytest.approx(1.0)
    assert lower == pytest.approx(-4.0)
    assert upper == pytest.approx(1.0)
