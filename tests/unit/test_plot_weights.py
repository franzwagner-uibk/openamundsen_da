from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openamundsen_da.methods.pf import plot_weights as plot_mod
from openamundsen_da.methods.viz.plots.theme import da_variable_style
from openamundsen_da.util.da_observables import weight_plot_title_from_csv_path


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_project_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    setup_dir = tmp_path / "setup_case"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    step_dir = project_dir / "steps" / "step_00_init"
    _write_text(setup_dir / "setup_case.yml", "name: setup_case\n")
    _write_text(project_dir / "project_2022_2023.yml", "name: project_2022_2023\n")
    (step_dir / "assim").mkdir(parents=True, exist_ok=True)
    _write_text(
        step_dir / "00.yml",
        "start_date: '2022-11-01 00:00:00'\nend_date: '2022-11-01 21:00:00'\nresults_dir: results\n",
    )
    return setup_dir, project_dir, step_dir


def _add_weights_event(
    project_dir: Path,
    *,
    step_idx: int,
    observable: str,
    date_str: str,
    weights_rows: list[dict[str, object]],
    diag_rows: list[dict[str, object]] | None = None,
) -> Path:
    weight_prefix = {
        "station_hs": "weights_station_hs",
        "station_swe": "weights_station_swe",
        "scf": "weights_scf",
        "wet_snow": "weights_wet_snow",
        "wet_snow_line": "weights_wet_snow_line",
    }[observable]
    diag_prefix = {
        "station_hs": "station_diagnostics_station_hs",
        "station_swe": "station_diagnostics_station_swe",
    }
    step_dir = project_dir / "steps" / f"step_{step_idx:02d}_event"
    assim_dir = step_dir / "assim"
    dt = pd.to_datetime(date_str, format="%Y%m%d")
    _write_text(
        step_dir / f"{step_idx:02d}.yml",
        (
            f"start_date: '{dt.strftime('%Y-%m-%d 00:00:00')}'\n"
            f"end_date: '{dt.strftime('%Y-%m-%d 21:00:00')}'\n"
            "results_dir: results\n"
        ),
    )
    csv_path = assim_dir / f"{weight_prefix}_{date_str}.csv"
    _write_csv(csv_path, weights_rows)
    if diag_rows is not None:
        _write_csv(assim_dir / f"{diag_prefix[observable]}_{date_str}.csv", diag_rows)
    return csv_path


def _render_setup_weights_overview_figure(project_dir: Path, monkeypatch) -> object:
    import matplotlib.pyplot as plt

    saved: dict[str, object] = {}

    def _fake_save(fig, out, **kwargs) -> None:
        saved["fig"] = fig
        saved["out"] = out

    monkeypatch.setattr(plot_mod, "save_figure_png", _fake_save)
    plot_mod.plot_setup_weights_overview(project_dir, backend="Agg")
    fig = saved["fig"]
    fig.canvas.draw()
    return fig


def _axes_with_xlabel(fig, label: str) -> list[object]:
    return [ax for ax in fig.axes if ax.get_xlabel() == label]


def test_axis_labels_use_residual_terminology() -> None:
    assert plot_mod._fraction_axis_label("scf") == "snow cover fraction residual"
    assert plot_mod._fraction_axis_label("wet_snow") == "wet-snow fraction residual"
    assert plot_mod._fraction_axis_label("wet_snow_line") == "wet-snow line residual [m]"
    assert plot_mod._station_axis_label("station_hs") == "snow depth residual [m]"
    assert plot_mod._station_axis_label("station_swe") == "SWE residual [mm]"


def test_wet_snow_line_weights_csv_is_not_misclassified_as_wet_snow(tmp_path: Path) -> None:
    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_wet_snow_line_20230523.csv"
    _write_csv(csv_path, [{"member_id": "member_001", "residual": 12.0, "sigma": 150.0, "log_weight": -1.0, "weight": 1.0}])

    assert plot_mod._observable_from_csv_path(csv_path) == "wet_snow_line"
    assert weight_plot_title_from_csv_path(csv_path) == "wet snow line data assimilation weights"


def test_nice_axis_extent_uses_quarter_steps_just_above_one() -> None:
    assert plot_mod._nice_axis_extent(1.0894838) == pytest.approx(1.25)


def test_overview_member_ticks_use_sparse_readable_labels_for_high_ensemble_sizes() -> None:
    assert plot_mod._member_ticks(47) == [1, 10, 20, 30, 40]
    assert plot_mod._member_ticks(8) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_setup_weights_overview_default_output_path_uses_project_weights_dir(tmp_path: Path) -> None:
    _, project_dir, _ = _build_project_tree(tmp_path)

    out = plot_mod._default_setup_weights_overview_output(project_dir)

    assert out == project_dir / "results" / "plots" / "assim" / "weights" / "setup_weights_overview_2022_2023.png"


def test_collect_marker_legend_entries_combines_station_and_fraction_labels(tmp_path: Path) -> None:
    setup_dir, project_dir, step_dir = _build_project_tree(tmp_path)
    _write_csv(
        setup_dir / "meteo" / "stations.csv",
        [
            {"id": "latschbloder", "name": "Latschbloder"},
            {"id": "proviantdepot", "name": "Proviantdepot"},
        ],
    )
    _write_csv(
        project_dir / "obs" / "stations" / "stations_da_metadata.csv",
        [
            {"station_id": "latschbloder", "station_uncertainty_pct": 500},
            {"station_id": "proviantdepot", "station_uncertainty_pct": 10},
        ],
    )
    station_csv = step_dir / "assim" / "weights_station_hs_20221122.csv"
    _write_csv(
        station_csv,
        [
            {"member_id": "member_001", "residual": -0.1, "sigma": 0.29, "log_weight": -1.0, "weight": 0.7},
            {"member_id": "member_002", "residual": 0.2, "sigma": 0.29, "log_weight": -2.0, "weight": 0.3},
        ],
    )
    _write_csv(
        step_dir / "assim" / "station_diagnostics_station_hs_20221122.csv",
        [
            {"station_id": "latschbloder", "member_id": "member_001", "residual": -0.1, "sigma": 0.29},
            {"station_id": "latschbloder", "member_id": "member_002", "residual": 0.2, "sigma": 0.29},
            {"station_id": "proviantdepot", "member_id": "member_001", "residual": -0.05, "sigma": 0.05},
            {"station_id": "proviantdepot", "member_id": "member_002", "residual": 0.1, "sigma": 0.05},
        ],
    )
    scf_csv = step_dir / "assim" / "weights_scf_20230518.csv"
    wet_csv = step_dir / "assim" / "weights_wet_snow_20230523.csv"
    _write_csv(
        scf_csv,
        [{"member_id": "member_001", "residual": 0.1, "sigma": 0.1, "log_weight": -1.0, "weight": 1.0}],
    )
    _write_csv(
        wet_csv,
        [{"member_id": "member_001", "residual": 0.1, "sigma": 0.08, "log_weight": -1.0, "weight": 1.0}],
    )

    entries = plot_mod._collect_marker_legend_entries([station_csv, wet_csv, scf_csv])

    assert entries == [
        ("Latschbloder (σ=500%)", "#ff7f0e"),
        ("Proviantdepot (σ=10%)", "#9467bd"),
        ("wet snow", "#2c8a64"),
        ("SCF", "#2f6fb5"),
    ]


def test_weights_color_sources_follow_shared_da_palette() -> None:
    assert plot_mod._FRACTION_MISMATCH_COLORS["scf"] == da_variable_style("scf")["line"]
    assert plot_mod._FRACTION_MISMATCH_COLORS["wet_snow"] == da_variable_style("wet_snow")["line"]
    assert plot_mod._station_color_map(["station_a"], observable="station_hs") == {
        "station_a": da_variable_style("station_hs")["line"],
    }
    assert plot_mod._station_color_map(["station_b"], observable="station_swe") == {
        "station_b": da_variable_style("station_swe")["line"],
    }


def test_station_plot_uses_right_aligned_sigma_strip_and_shared_bottom_legend(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    _write_csv(
        setup_dir / "meteo" / "stations.csv",
        [
            {"id": "latschbloder", "name": "Latschbloder"},
            {"id": "proviantdepot", "name": "Proviantdepot"},
        ],
    )
    _write_csv(
        _project_dir / "obs" / "stations" / "stations_da_metadata.csv",
        [
            {"station_id": "latschbloder", "station_uncertainty_pct": 500},
            {"station_id": "proviantdepot", "station_uncertainty_pct": 10},
        ],
    )
    csv_path = step_dir / "assim" / "weights_station_hs_20221122.csv"
    _write_csv(
        csv_path,
        [
            {"member_id": "member_001", "residual": -0.1, "sigma": 0.17, "log_weight": -1.0, "weight": 0.7},
            {"member_id": "member_002", "residual": 0.2, "sigma": 0.17, "log_weight": -2.0, "weight": 0.3},
        ],
    )
    _write_csv(
        step_dir / "assim" / "station_diagnostics_station_hs_20221122.csv",
        [
            {"station_id": "latschbloder", "member_id": "member_001", "residual": -0.1, "sigma": 0.29},
            {"station_id": "latschbloder", "member_id": "member_002", "residual": 0.2, "sigma": 0.29},
            {"station_id": "proviantdepot", "member_id": "member_001", "residual": -0.05, "sigma": 0.05},
            {"station_id": "proviantdepot", "member_id": "member_002", "residual": 0.1, "sigma": 0.05},
        ],
    )

    fig = plot_mod._plot(
        csv_path,
        plot_mod._load_weights(csv_path),
        title="station snow depth data assimilation weights",
        subtitle="DA 1 - 2022-11-22",
        observable="station_hs",
        backend="Agg",
    )

    ax1 = fig.axes[1]
    sigma_legend = ax1.get_legend()
    sigma_labels = [text.get_text() for text in sigma_legend.get_texts()]
    bottom_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sigma_bbox = sigma_legend.get_window_extent(renderer=renderer)
    ax_bbox = ax1.get_window_extent(renderer=renderer)

    assert ax1.get_xlabel() == "snow depth residual [m]"
    assert sigma_labels == ["σ=0.29", "σ=0.05"]
    assert sigma_legend._legend_box.align == "right"
    assert abs(sigma_bbox.x1 - ax_bbox.x1) <= 2.0
    assert 0.5 < (sigma_bbox.y0 - ax_bbox.y1) < 18.0
    assert bottom_labels == [
        "Latschbloder (σ=500%)",
        "Proviantdepot (σ=10%)",
        "redrawn source member (extra rings = repeated draws)",
    ]
    plt.close(fig)


def test_fraction_plot_uses_observable_legend_entry_and_sigma_strip(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_scf_20230518.csv"
    _write_csv(
        csv_path,
        [
            {"member_id": "member_001", "residual": 0.1, "sigma": 0.10, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": -0.2, "sigma": 0.10, "log_weight": -1.5, "weight": 0.4},
        ],
    )

    fig = plot_mod._plot(
        csv_path,
        plot_mod._load_weights(csv_path),
        title="snow cover data assimilation weights",
        subtitle="DA 8 - 2023-05-18",
        observable="scf",
        backend="Agg",
    )

    ax1 = fig.axes[1]
    sigma_legend = ax1.get_legend()
    sigma_labels = [text.get_text() for text in sigma_legend.get_texts()]
    bottom_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sigma_bbox = sigma_legend.get_window_extent(renderer=renderer)
    ax_bbox = ax1.get_window_extent(renderer=renderer)

    assert ax1.get_xlabel() == "snow cover fraction residual"
    assert sigma_labels == ["σ=0.10"]
    assert sigma_legend._legend_box.align == "right"
    assert abs(sigma_bbox.x1 - ax_bbox.x1) <= 2.0
    assert 0.5 < (sigma_bbox.y0 - ax_bbox.y1) < 18.0
    assert bottom_labels == [
        "SCF",
        "redrawn source member (extra rings = repeated draws)",
    ]
    plt.close(fig)


def test_wet_snow_line_plot_labels_zero_line_and_skipped_event(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_wet_snow_line_20230523.csv"
    _write_csv(
        csv_path,
        [
            {
                "member_id": "member_001",
                "residual": "",
                "sigma": "",
                "weight": 0.5,
                "log_weight": 0.0,
                "value_obs": 3066.7,
                "value_model": 2890.0,
                "wet_information_gate_triggered": True,
                "wet_information_gate_reason": "no_crossing_fraction",
                "model_gate_triggered": True,
                "model_gate_reason": "no_crossing_fraction",
            },
            {
                "member_id": "member_002",
                "residual": "",
                "sigma": "",
                "weight": 0.5,
                "log_weight": 0.0,
                "value_obs": 3066.7,
                "value_model": 3010.0,
                "wet_information_gate_triggered": True,
                "wet_information_gate_reason": "no_crossing_fraction",
                "model_gate_triggered": True,
                "model_gate_reason": "no_crossing_fraction",
            },
        ],
    )

    fig = plot_mod._plot(
        csv_path,
        plot_mod._load_weights(csv_path),
        title="wet snow line data assimilation weights",
        subtitle="DA 13 - 2023-05-23",
        observable="wet_snow_line",
        backend="Agg",
    )

    ax1 = fig.axes[1]
    note_texts = [text.get_text() for text in ax1.texts]

    assert ax1.get_xlabel() == "wet-snow line residual [m]"
    assert any("obs WSL 3067 m" in text for text in note_texts)
    assert any("WSL update skipped" in text for text in note_texts)
    assert not any("model range" in text for text in note_texts)
    plt.close(fig)


def test_station_legend_falls_back_to_name_when_metadata_missing(tmp_path: Path) -> None:
    setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    _write_csv(
        setup_dir / "meteo" / "stations.csv",
        [
            {"id": "latschbloder", "name": "Latschbloder"},
            {"id": "proviantdepot", "name": "Proviantdepot"},
        ],
    )
    station_csv = step_dir / "assim" / "weights_station_hs_20221122.csv"
    _write_csv(
        station_csv,
        [
            {"member_id": "member_001", "residual": -0.1, "sigma": 0.29, "log_weight": -1.0, "weight": 0.7},
            {"member_id": "member_002", "residual": 0.2, "sigma": 0.29, "log_weight": -2.0, "weight": 0.3},
        ],
    )
    _write_csv(
        step_dir / "assim" / "station_diagnostics_station_hs_20221122.csv",
        [
            {"station_id": "latschbloder", "member_id": "member_001", "residual": -0.1, "sigma": 0.29},
            {"station_id": "proviantdepot", "member_id": "member_001", "residual": -0.05, "sigma": 0.05},
        ],
    )

    entries = plot_mod._collect_marker_legend_entries([station_csv])

    assert entries == [
        ("Latschbloder", "#ff7f0e"),
        ("Proviantdepot", "#9467bd"),
    ]


def test_axes_subplot_titles_bold_da_prefix_and_use_lower_anchor(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_scf_20230518.csv"
    _write_csv(
        csv_path,
        [
            {"member_id": "member_001", "residual": 0.1, "sigma": 0.10, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": -0.2, "sigma": 0.10, "log_weight": -1.5, "weight": 0.4},
        ],
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    plot_mod._draw_weights_event(
        fig,
        ax0,
        ax1,
        csv_path=csv_path,
        df=plot_mod._load_weights(csv_path),
        title="DA 3 - 2023-05-18 - snow cover",
        subtitle=None,
        observable="scf",
        title_mode="axes",
        show_metrics_label=False,
    )

    assert len(ax0.texts) == 1
    title_text = ax0.texts[0]
    assert title_text.get_text() == r"$\mathbf{DA\ 3}$ - 2023-05-18 - snow cover"
    assert title_text.get_position() == (0.0, 1.18)
    plt.close(fig)


def test_metrics_label_is_left_aligned_and_overview_enables_it(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_scf_20230518.csv"
    _write_csv(
        csv_path,
        [
            {"member_id": "member_001", "residual": 0.1, "sigma": 0.10, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": -0.2, "sigma": 0.10, "log_weight": -1.5, "weight": 0.4},
        ],
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    plot_mod._draw_weights_event(
        fig,
        ax0,
        ax1,
        csv_path=csv_path,
        df=plot_mod._load_weights(csv_path),
        title="DA 8 - 2023-05-18 - snow cover",
        subtitle=None,
        observable="scf",
        title_mode="axes",
        show_metrics_label=True,
        show_metrics_threshold=False,
    )

    metrics_text = next(text for text in ax0.texts if text.get_text().startswith("ESS ="))
    assert metrics_text.get_position() == (0.0, 1.02)
    assert metrics_text.get_ha() == "left"
    assert metrics_text.get_text() == "ESS = 1.9"
    plt.close(fig)


def test_standalone_metrics_label_includes_threshold(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_scf_20230518.csv"
    _write_csv(
        csv_path,
        [
            {"member_id": "member_001", "residual": 0.1, "sigma": 0.10, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": -0.2, "sigma": 0.10, "log_weight": -1.5, "weight": 0.4},
        ],
    )
    _write_text(
        step_dir / "assim" / "resample_manifest_20230518.json",
        '{"ess_threshold": 7.5}',
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    plot_mod._draw_weights_event(
        fig,
        ax0,
        ax1,
        csv_path=csv_path,
        df=plot_mod._load_weights(csv_path),
        title="snow cover",
        subtitle=None,
        observable="scf",
        title_mode="figure",
        show_metrics_label=True,
        show_metrics_threshold=True,
    )

    metrics_text = next(text for text in ax0.texts if text.get_text().startswith("ESS ="))
    assert metrics_text.get_text() == "ESS = 1.9 (threshold=7.5)"
    plt.close(fig)


def test_scale_axes_group_shrinks_standalone_axes_and_keeps_them_high() -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.2876875, 3.013))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.575, 3.425])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.095, right=0.965, top=0.78, bottom=0.30, wspace=0.30)

    original_positions = [ax.get_position().frozen() for ax in (ax0, ax1)]
    original_left = min(pos.x0 for pos in original_positions)
    original_right = max(pos.x1 for pos in original_positions)
    original_bottom = min(pos.y0 for pos in original_positions)
    original_top = max(pos.y1 for pos in original_positions)

    plot_mod._scale_axes_group(
        [ax0, ax1],
        width_scale=plot_mod._STANDALONE_PLOT_WIDTH_SCALE,
        height_scale=plot_mod._STANDALONE_PLOT_HEIGHT_SCALE,
        top_anchor=plot_mod._STANDALONE_PLOT_TOP,
    )

    scaled_positions = [ax.get_position().frozen() for ax in (ax0, ax1)]
    scaled_left = min(pos.x0 for pos in scaled_positions)
    scaled_right = max(pos.x1 for pos in scaled_positions)
    scaled_bottom = min(pos.y0 for pos in scaled_positions)
    scaled_top = max(pos.y1 for pos in scaled_positions)

    assert scaled_right - scaled_left == pytest.approx(
        (original_right - original_left) * plot_mod._STANDALONE_PLOT_WIDTH_SCALE
    )
    assert scaled_top - scaled_bottom == pytest.approx(
        (original_top - original_bottom) * plot_mod._STANDALONE_PLOT_HEIGHT_SCALE
    )
    assert scaled_top == pytest.approx(plot_mod._STANDALONE_PLOT_TOP)
    plt.close(fig)


def test_standalone_plot_uses_lower_title_anchor(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_scf_20230518.csv"
    _write_csv(
        csv_path,
        [
            {"member_id": "member_001", "residual": 0.1, "sigma": 0.10, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": -0.2, "sigma": 0.10, "log_weight": -1.5, "weight": 0.4},
        ],
    )

    fig = plot_mod._plot(
        csv_path,
        plot_mod._load_weights(csv_path),
        title="snow cover data assimilation weights",
        subtitle="DA 8 - 2023-05-18",
        observable="scf",
        backend="Agg",
    )

    title_text = next(text for text in fig.texts if text.get_text().startswith("snow cover data assimilation weights"))
    legend = fig.legends[0]
    legend_anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    ax0 = fig.axes[0]
    assert title_text.get_position() == (0.11, plot_mod._STANDALONE_TITLE_Y)
    assert legend_anchor.y0 == pytest.approx(plot_mod._STANDALONE_LEGEND_Y)
    assert list(ax0.get_xticks()) == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert [tick.get_text() for tick in ax0.get_xticklabels()] == ["0.2", "0.4", "0.6", "0.8", "1"]
    xlim = ax0.get_xlim()
    assert xlim[0] < 0.0
    assert xlim[1] > 1.0
    plt.close(fig)


def test_standalone_plot_uses_sparse_member_ticks_for_high_ensemble_sizes(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_scf_20230518.csv"
    _write_csv(
        csv_path,
        [
            {
                "member_id": f"member_{idx:03d}",
                "residual": (idx - 24) / 100.0,
                "sigma": 0.1,
                "log_weight": -1.0 - idx * 0.01,
                "weight": 1.0 / 47.0,
            }
            for idx in range(1, 48)
        ],
    )

    fig = plot_mod._plot(
        csv_path,
        plot_mod._load_weights(csv_path),
        title="snow cover data assimilation weights",
        subtitle="DA 8 - 2023-05-18",
        observable="scf",
        backend="Agg",
    )

    ax0 = fig.axes[0]
    ax1 = fig.axes[1]
    expected_ticks = [1, 10, 20, 30, 40]

    assert [int(tick) for tick in ax0.get_yticks()] == expected_ticks
    assert [int(tick) for tick in ax1.get_yticks()] == expected_ticks
    assert len(ax0.yaxis.get_minorticklocs()) == 0
    assert len(ax1.yaxis.get_minorticklocs()) == 0
    plt.close(fig)


def test_explicit_residual_xlim_gets_small_buffer(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_scf_20230518.csv"
    _write_csv(
        csv_path,
        [
            {"member_id": "member_001", "residual": -0.2, "sigma": 0.10, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 0.2, "sigma": 0.10, "log_weight": -1.5, "weight": 0.4},
        ],
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    plot_mod._draw_weights_event(
        fig,
        ax0,
        ax1,
        csv_path=csv_path,
        df=plot_mod._load_weights(csv_path),
        title="snow cover",
        subtitle=None,
        observable="scf",
        residual_xlim=(-0.2, 0.2),
    )

    xlim = ax1.get_xlim()
    assert xlim[0] < -0.2
    assert xlim[1] > 0.2
    plt.close(fig)


def test_autoscaled_residual_xlim_gets_small_buffer(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_scf_20230518.csv"
    _write_csv(
        csv_path,
        [
            {"member_id": "member_001", "residual": -0.2, "sigma": 0.10, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 0.15, "sigma": 0.10, "log_weight": -1.5, "weight": 0.4},
        ],
    )

    fig = plot_mod._plot(
        csv_path,
        plot_mod._load_weights(csv_path),
        title="snow cover data assimilation weights",
        subtitle="DA 8 - 2023-05-18",
        observable="scf",
        backend="Agg",
    )

    ax1 = fig.axes[1]
    xlim = ax1.get_xlim()
    assert xlim[0] < -0.2
    assert xlim[1] > 0.15
    plt.close(fig)


def test_setup_weights_overview_legend_prefers_single_row_until_wrap_is_needed() -> None:
    import matplotlib.pyplot as plt

    handles = [
        plot_mod._marker_handle("#ff7f0e"),
        plot_mod._marker_handle("#9467bd"),
        plot_mod._marker_handle("#2c8a64"),
        plot_mod._marker_handle("#2f6fb5"),
    ]
    labels = [
        "Latschbloder (σ=500%)",
        "Proviantdepot (σ=10%)",
        "wet snow",
        "redrawn source member (extra rings = repeated draws)",
    ]
    legend_kwargs = dict(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        frameon=False,
        fontsize=6.2,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    wide_fig = plt.figure(figsize=(7.2876875, 3.0))
    wide_ncol = plot_mod._best_figure_legend_ncol(
        wide_fig,
        handles,
        labels,
        handler_map={},
        **legend_kwargs,
    )

    narrow_fig = plt.figure(figsize=(3.2, 3.0))
    narrow_ncol = plot_mod._best_figure_legend_ncol(
        narrow_fig,
        handles,
        labels,
        handler_map={},
        **legend_kwargs,
    )

    assert wide_ncol == len(labels)
    assert 1 <= narrow_ncol < len(labels)
    plt.close(wide_fig)
    plt.close(narrow_fig)


def test_setup_overview_shares_residual_xlim_within_same_observable(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    _write_csv(setup_dir / "meteo" / "stations.csv", [{"id": "station_a", "name": "Station A"}])
    _add_weights_event(
        project_dir,
        step_idx=0,
        observable="station_hs",
        date_str="20221122",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.1, "sigma": 0.2, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 0.2, "sigma": 0.2, "log_weight": -1.2, "weight": 0.4},
        ],
        diag_rows=[
            {"station_id": "station_a", "member_id": "member_001", "residual": -0.10, "sigma": 0.20},
            {"station_id": "station_a", "member_id": "member_002", "residual": 0.15, "sigma": 0.20},
        ],
    )
    _add_weights_event(
        project_dir,
        step_idx=1,
        observable="station_hs",
        date_str="20221222",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.2, "sigma": 0.25, "log_weight": -1.0, "weight": 0.7},
            {"member_id": "member_002", "residual": 0.5, "sigma": 0.25, "log_weight": -1.2, "weight": 0.3},
        ],
        diag_rows=[
            {"station_id": "station_a", "member_id": "member_001", "residual": -0.45, "sigma": 0.25},
            {"station_id": "station_a", "member_id": "member_002", "residual": 0.50, "sigma": 0.25},
        ],
    )

    fig = _render_setup_weights_overview_figure(project_dir, monkeypatch)
    residual_axes = _axes_with_xlabel(fig, "snow depth residual [m]")
    expected_xlim = plot_mod._expand_xlim((-0.5, 0.5))

    assert residual_axes[0].get_xlim() == pytest.approx(expected_xlim)
    assert residual_axes[1].get_xlim() == pytest.approx(expected_xlim)
    plt.close(fig)


def test_setup_overview_uses_separate_residual_xlims_per_observable(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    _write_csv(
        setup_dir / "meteo" / "stations.csv",
        [
            {"id": "station_a", "name": "Station A"},
            {"id": "station_b", "name": "Station B"},
        ],
    )
    _add_weights_event(
        project_dir,
        step_idx=0,
        observable="station_hs",
        date_str="20221122",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.4, "sigma": 0.3, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 0.8, "sigma": 0.3, "log_weight": -1.2, "weight": 0.4},
        ],
        diag_rows=[
            {"station_id": "station_a", "member_id": "member_001", "residual": -0.80, "sigma": 0.30},
            {"station_id": "station_a", "member_id": "member_002", "residual": 0.60, "sigma": 0.30},
        ],
    )
    _add_weights_event(
        project_dir,
        step_idx=1,
        observable="station_swe",
        date_str="20221222",
        weights_rows=[
            {"member_id": "member_001", "residual": -10.0, "sigma": 8.0, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 12.0, "sigma": 8.0, "log_weight": -1.2, "weight": 0.4},
        ],
        diag_rows=[
            {"station_id": "station_b", "member_id": "member_001", "residual": -12.0, "sigma": 8.0},
            {"station_id": "station_b", "member_id": "member_002", "residual": 15.0, "sigma": 8.0},
        ],
    )
    _add_weights_event(
        project_dir,
        step_idx=2,
        observable="scf",
        date_str="20230122",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.35, "sigma": 0.10, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 0.20, "sigma": 0.10, "log_weight": -1.2, "weight": 0.4},
        ],
    )
    _add_weights_event(
        project_dir,
        step_idx=3,
        observable="wet_snow",
        date_str="20230221",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.10, "sigma": 0.55, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 0.25, "sigma": 0.55, "log_weight": -1.2, "weight": 0.4},
        ],
    )

    fig = _render_setup_weights_overview_figure(project_dir, monkeypatch)
    hs_axes = _axes_with_xlabel(fig, "snow depth residual [m]")
    swe_axes = _axes_with_xlabel(fig, "SWE residual [mm]")
    scf_axes = _axes_with_xlabel(fig, "snow cover fraction residual")
    wet_axes = _axes_with_xlabel(fig, "wet-snow fraction residual")

    assert len(hs_axes) == 1
    assert len(swe_axes) == 1
    assert len(scf_axes) == 1
    assert len(wet_axes) == 1
    assert hs_axes[0].get_xlim() == pytest.approx(plot_mod._expand_xlim((-0.8, 0.8)))
    assert swe_axes[0].get_xlim() == pytest.approx(plot_mod._expand_xlim((-15.0, 15.0)))
    assert scf_axes[0].get_xlim() == pytest.approx(plot_mod._expand_xlim((-0.35, 0.35)))
    assert wet_axes[0].get_xlim() == pytest.approx(plot_mod._expand_xlim((-0.6, 0.6)))
    plt.close(fig)


def test_setup_overview_residual_xlim_respects_sigma_when_residuals_are_narrow(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    _add_weights_event(
        project_dir,
        step_idx=0,
        observable="scf",
        date_str="20230518",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.08, "sigma": 0.40, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 0.10, "sigma": 0.40, "log_weight": -1.2, "weight": 0.4},
        ],
    )
    _add_weights_event(
        project_dir,
        step_idx=1,
        observable="scf",
        date_str="20230526",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.12, "sigma": 0.30, "log_weight": -1.0, "weight": 0.7},
            {"member_id": "member_002", "residual": 0.09, "sigma": 0.30, "log_weight": -1.2, "weight": 0.3},
        ],
    )

    fig = _render_setup_weights_overview_figure(project_dir, monkeypatch)
    residual_axes = _axes_with_xlabel(fig, "snow cover fraction residual")
    expected_xlim = plot_mod._expand_xlim((-0.4, 0.4))

    assert residual_axes[0].get_xlim() == pytest.approx(expected_xlim)
    assert residual_axes[1].get_xlim() == pytest.approx(expected_xlim)
    plt.close(fig)


def test_setup_overview_robust_shared_xlim_keeps_extreme_residual_visible(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    _write_csv(setup_dir / "meteo" / "stations.csv", [{"id": "station_a", "name": "Station A"}])

    weights_rows_0 = []
    diag_rows_0 = []
    for idx in range(1, 11):
        member_id = f"member_{idx:03d}"
        residual = 0.18 + 0.004 * idx
        weights_rows_0.append(
            {"member_id": member_id, "residual": residual, "sigma": 0.1, "log_weight": -1.0 - idx * 0.01, "weight": 0.1}
        )
        diag_rows_0.append({"station_id": "station_a", "member_id": member_id, "residual": residual, "sigma": 0.1})
    _add_weights_event(
        project_dir,
        step_idx=0,
        observable="station_hs",
        date_str="20221122",
        weights_rows=weights_rows_0,
        diag_rows=diag_rows_0,
    )

    weights_rows_1 = []
    diag_rows_1 = []
    for idx in range(1, 11):
        member_id = f"member_{idx:03d}"
        residual = 0.19 + 0.003 * idx
        if idx == 10:
            residual = 2.0
        weights_rows_1.append(
            {"member_id": member_id, "residual": residual, "sigma": 0.1, "log_weight": -1.1 - idx * 0.01, "weight": 0.1}
        )
        diag_rows_1.append({"station_id": "station_a", "member_id": member_id, "residual": residual, "sigma": 0.1})
    _add_weights_event(
        project_dir,
        step_idx=1,
        observable="station_hs",
        date_str="20221222",
        weights_rows=weights_rows_1,
        diag_rows=diag_rows_1,
    )

    fig = _render_setup_weights_overview_figure(project_dir, monkeypatch)
    residual_axes = _axes_with_xlabel(fig, "snow depth residual [m]")
    expected_xlim = plot_mod._expand_xlim((-2.0, 2.0))

    assert len(residual_axes) == 2
    assert residual_axes[0].get_xlim() == pytest.approx(expected_xlim)
    assert residual_axes[1].get_xlim() == pytest.approx(expected_xlim)
    plt.close(fig)


def test_setup_overview_robust_shared_xlim_ignores_single_sigma_outlier(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    _write_csv(setup_dir / "meteo" / "stations.csv", [{"id": "station_a", "name": "Station A"}])

    weights_rows_0 = []
    diag_rows_0 = []
    for idx in range(1, 11):
        member_id = f"member_{idx:03d}"
        residual = 0.18 + 0.004 * idx
        sigma = 0.1
        weights_rows_0.append(
            {"member_id": member_id, "residual": residual, "sigma": sigma, "log_weight": -1.0 - idx * 0.01, "weight": 0.1}
        )
        diag_rows_0.append({"station_id": "station_a", "member_id": member_id, "residual": residual, "sigma": sigma})
    _add_weights_event(
        project_dir,
        step_idx=0,
        observable="station_hs",
        date_str="20221122",
        weights_rows=weights_rows_0,
        diag_rows=diag_rows_0,
    )

    weights_rows_1 = []
    diag_rows_1 = []
    for idx in range(1, 11):
        member_id = f"member_{idx:03d}"
        residual = 0.19 + 0.003 * idx
        sigma = 0.1
        if idx == 10:
            sigma = 2.0
        weights_rows_1.append(
            {"member_id": member_id, "residual": residual, "sigma": sigma, "log_weight": -1.1 - idx * 0.01, "weight": 0.1}
        )
        diag_rows_1.append({"station_id": "station_a", "member_id": member_id, "residual": residual, "sigma": sigma})
    _add_weights_event(
        project_dir,
        step_idx=1,
        observable="station_hs",
        date_str="20221222",
        weights_rows=weights_rows_1,
        diag_rows=diag_rows_1,
    )

    fig = _render_setup_weights_overview_figure(project_dir, monkeypatch)
    residual_axes = _axes_with_xlabel(fig, "snow depth residual [m]")
    expected_xlim = plot_mod._expand_xlim((-0.25, 0.25))

    assert len(residual_axes) == 2
    assert residual_axes[0].get_xlim() == pytest.approx(expected_xlim)
    assert residual_axes[1].get_xlim() == pytest.approx(expected_xlim)
    plt.close(fig)


def test_setup_overview_uses_sparse_member_ticks_for_high_ensemble_sizes(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    weights_rows = [
        {
            "member_id": f"member_{idx:03d}",
            "residual": (idx - 24) / 100.0,
            "sigma": 0.1,
            "log_weight": -1.0 - idx * 0.01,
            "weight": 1.0 / 47.0,
        }
        for idx in range(1, 48)
    ]
    _add_weights_event(
        project_dir,
        step_idx=0,
        observable="scf",
        date_str="20230518",
        weights_rows=weights_rows,
    )

    fig = _render_setup_weights_overview_figure(project_dir, monkeypatch)
    weight_axes = _axes_with_xlabel(fig, "weight")
    residual_axes = _axes_with_xlabel(fig, "snow cover fraction residual")

    assert len(weight_axes) == 1
    assert [int(tick) for tick in weight_axes[0].get_yticks()] == [1, 10, 20, 30, 40]
    assert len(weight_axes[0].yaxis.get_minorticklocs()) == 0
    assert len(residual_axes) == 1
    assert len(residual_axes[0].yaxis.get_minorticklocs()) == 0
    plt.close(fig)
