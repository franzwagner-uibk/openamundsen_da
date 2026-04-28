from __future__ import annotations

from pathlib import Path

from openamundsen_da.io.paths import (
    project_benchmark_plots_dir,
    project_benchmark_root,
    project_da_output_grids_path,
    project_fraction_envelope_path,
    project_grids_root,
    project_maps_output_dir,
    project_maps_root,
    project_misc_root,
    project_obs_selection_plot_path,
    project_plot_assim_dir,
    project_plot_assim_ess_dir,
    project_plot_assim_scores_dir,
    project_plot_assim_weights_dir,
    project_plot_perf_dir,
    project_plot_points_dir,
    project_plot_results_dir,
    project_plots_maps_collection_pdf_path,
    project_plots_root,
    project_reports_root,
    project_result_overview_custom_output_path,
    project_result_overview_output_path,
    project_results_root,
)


def test_project_level_results_paths_use_canonical_results_tree(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"

    assert project_results_root(project_dir) == project_dir / "results"
    assert project_plots_root(project_dir) == project_dir / "results" / "plots"
    assert project_plot_results_dir(project_dir) == project_dir / "results" / "plots" / "results"
    assert project_plot_assim_dir(project_dir) == project_dir / "results" / "plots" / "assim"
    assert project_plot_assim_weights_dir(project_dir) == project_dir / "results" / "plots" / "assim" / "weights"
    assert project_plot_assim_ess_dir(project_dir) == project_dir / "results" / "plots" / "assim" / "ess"
    assert project_plot_assim_scores_dir(project_dir) == project_dir / "results" / "plots" / "assim" / "scores"
    assert project_plot_perf_dir(project_dir) == project_dir / "results" / "plots" / "perf"
    assert project_plot_points_dir(project_dir) == project_dir / "results" / "plots" / "points"
    assert project_result_overview_output_path(project_dir) == project_dir / "results" / "plots" / "results" / "result_overview.png"
    assert project_result_overview_custom_output_path(project_dir) == project_dir / "results" / "plots" / "results" / "result_overview_custom.png"
    assert project_obs_selection_plot_path(project_dir) == project_dir / "results" / "plots" / "results" / "obs_selection.png"
    assert project_misc_root(project_dir) == project_dir / "results" / "misc"
    assert project_fraction_envelope_path(project_dir, "scf") == project_dir / "results" / "misc" / "point_scf_roi_envelope.csv"
    assert project_fraction_envelope_path(project_dir, "wet_snow") == project_dir / "results" / "misc" / "point_wet_snow_roi_envelope.csv"
    assert project_grids_root(project_dir) == project_dir / "results" / "grids"
    assert project_da_output_grids_path(project_dir) == project_dir / "results" / "grids" / "da_output_grids.nc"
    assert project_maps_root(project_dir) == project_dir / "results" / "maps"
    assert project_maps_output_dir(project_dir) == project_dir / "results" / "maps"
    assert project_reports_root(project_dir) == project_dir / "results" / "reports"
    assert (
        project_plots_maps_collection_pdf_path(project_dir)
        == project_dir / "results" / "reports" / "project_plots_maps_collection.pdf"
    )
    assert project_benchmark_root(project_dir) == project_dir / "results" / "benchmark"
    assert project_benchmark_plots_dir(project_dir) == project_plot_assim_scores_dir(project_dir)
