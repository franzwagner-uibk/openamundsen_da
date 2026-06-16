from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openamundsen_da.methods.viz.plots.assimilation import weights as plot_mod
from openamundsen_da.methods.viz.plots.theme import GRID_ALPHA, GRID_LS, GRID_LW, da_variable_style
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


def test_da_variable_styles_use_viridis_option_c_colors() -> None:
    expected = {
        "wet_snow_line": "#482475",
        "scf": "#23898e",
        "station_swe": "#2c738e",
        "wet_snow": "#1f9a8a",
        "station_hs": "#4ec36b",
    }

    for variable, color in expected.items():
        assert da_variable_style(variable) == {"fill": color, "line": color}


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

    saved: list[dict[str, object]] = []

    def _fake_save(fig, out, **kwargs) -> None:
        saved.append({"fig": fig, "out": out, "kwargs": kwargs})

    monkeypatch.setattr(plot_mod, "save_figure_png", _fake_save)
    plot_mod.plot_setup_weights_overview(project_dir, backend="Agg")
    fig = saved[0]["fig"]
    fig.canvas.draw()
    for item in saved[1:]:
        plt.close(item["fig"])
    return fig


def _render_setup_weights_overview_pages(project_dir: Path, monkeypatch) -> list[dict[str, object]]:
    saved: list[dict[str, object]] = []

    def _fake_save(fig, out, **kwargs) -> None:
        saved.append({"fig": fig, "out": out, "kwargs": kwargs})

    monkeypatch.setattr(plot_mod, "save_figure_png", _fake_save)
    plot_mod.plot_setup_weights_overview(project_dir, backend="Agg")
    for item in saved:
        item["fig"].canvas.draw()
    return saved


def _axes_with_xlabel(fig, label: str) -> list[object]:
    return [ax for ax in fig.axes if ax.get_xlabel() == label]


def _overview_axis_pairs(fig) -> list[tuple[object, object]]:
    axes = [
        ax
        for ax in fig.axes
        if ax.get_xlabel()
        in {
            "Weight",
            "Residual [-]",
            "Residual [m]",
            "Residual [mm]",
            "weight",
            "residual [-]",
            "residual [m]",
            "residual [mm]",
        }
    ]
    return list(zip(axes[0::2], axes[1::2]))


def _residual_axes_for_title(fig, title_fragment: str) -> list[object]:
    axes: list[object] = []
    for weight_ax, residual_ax in _overview_axis_pairs(fig):
        if any(title_fragment in text.get_text() for text in weight_ax.texts):
            axes.append(residual_ax)
    return axes


def _axis_legends(ax) -> list[object]:
    from matplotlib.legend import Legend

    return [child for child in ax.get_children() if isinstance(child, Legend)]


def test_axis_labels_use_residual_terminology() -> None:
    assert plot_mod._fraction_axis_label("scf") == "residual [-]"
    assert plot_mod._fraction_axis_label("wet_snow") == "residual [-]"
    assert plot_mod._fraction_axis_label("wet_snow_line") == "residual [m]"
    assert plot_mod._station_axis_label("station_hs") == "residual [m]"
    assert plot_mod._station_axis_label("station_swe") == "residual [mm]"


def test_wet_snow_line_weights_csv_is_not_misclassified_as_wet_snow(tmp_path: Path) -> None:
    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_wet_snow_line_20230523.csv"
    _write_csv(csv_path, [{"member_id": "member_001", "residual": 12.0, "sigma": 150.0, "log_weight": -1.0, "weight": 1.0}])

    assert plot_mod._observable_from_csv_path(csv_path) == "wet_snow_line"
    assert weight_plot_title_from_csv_path(csv_path) == "Wet snow line data assimilation weights"


def test_nice_axis_extent_uses_quarter_steps_just_above_one() -> None:
    assert plot_mod._nice_axis_extent(1.0894838) == pytest.approx(1.25)


def test_overview_member_ticks_use_sparse_readable_labels_for_high_ensemble_sizes() -> None:
    assert plot_mod._member_ticks(47) == [1, 10, 20, 30, 40]
    assert plot_mod._member_ticks(8) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_setup_weights_overview_default_output_path_uses_project_weights_dir(tmp_path: Path) -> None:
    _, project_dir, _ = _build_project_tree(tmp_path)

    out = plot_mod._default_setup_weights_overview_output(project_dir)

    assert out == project_dir / "results" / "plots" / "assim" / "weights" / "setup_weights_overview_2022_2023.png"


def test_setup_weights_overview_page_output_uses_stable_base_name_for_page_one(tmp_path: Path) -> None:
    _, project_dir, _ = _build_project_tree(tmp_path)
    out = plot_mod._default_setup_weights_overview_output(project_dir)

    assert plot_mod._setup_weights_overview_page_output(out, 0) == out
    assert plot_mod._setup_weights_overview_page_output(out, 1) == out.with_name("setup_weights_overview_2022_2023_page_02.png")


def test_remove_stale_setup_weights_overview_pages_keeps_requested_outputs(tmp_path: Path) -> None:
    _, project_dir, _ = _build_project_tree(tmp_path)
    out = plot_mod._default_setup_weights_overview_output(project_dir)
    page_02 = out.with_name("setup_weights_overview_2022_2023_page_02.png")
    page_03 = out.with_name("setup_weights_overview_2022_2023_page_03.png")
    page_99 = out.with_name("setup_weights_overview_2022_2023_page_99.png")
    page_02.write_text("page 2", encoding="utf-8")
    page_03.write_text("page 3", encoding="utf-8")
    page_99.write_text("page 99", encoding="utf-8")

    plot_mod._remove_stale_setup_weights_overview_pages(out, [out, page_02])

    assert page_02.exists()
    assert not page_03.exists()
    assert not page_99.exists()


def test_setup_weights_overview_writes_paper_copy_without_figure_title(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    _add_weights_event(
        project_dir,
        step_idx=0,
        observable="scf",
        date_str="20230501",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.1, "sigma": 0.2, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 0.1, "sigma": 0.2, "log_weight": -1.2, "weight": 0.4},
        ],
    )

    saved = _render_setup_weights_overview_pages(project_dir, monkeypatch)

    assert len(saved) == 2
    normal, paper = saved
    assert normal["out"] == project_dir / "results" / "plots" / "assim" / "weights" / "setup_weights_overview_2022_2023.png"
    assert paper["out"] == plot_mod.project_paper_output_path(project_dir, normal["out"])
    normal_title = next(text for text in normal["fig"].texts if text.get_text().startswith("Data assimilation weights"))
    assert normal_title.get_fontsize() == pytest.approx(8.0)
    assert normal["fig"].legends == []
    assert not any(text.get_text().startswith("Data assimilation weights") for text in paper["fig"].texts)
    assert paper["fig"].legends == []

    for item in saved:
        plt.close(item["fig"])


def test_setup_weights_overview_places_event_title_close_to_panel(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    _add_weights_event(
        project_dir,
        step_idx=0,
        observable="scf",
        date_str="20230501",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.1, "sigma": 0.2, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 0.1, "sigma": 0.2, "log_weight": -1.2, "weight": 0.4},
        ],
    )

    fig = _render_setup_weights_overview_figure(project_dir, monkeypatch)
    weight_ax = _overview_axis_pairs(fig)[0][0]
    title_text = next(text for text in weight_ax.texts if "2023-05-01" in text.get_text())

    assert title_text.get_text() == "(a) DA 1 - 2023-05-01 - Snow cover"
    assert title_text.get_position()[1] == pytest.approx(1.035)
    assert title_text.get_ha() == "left"
    assert title_text.get_va() == "bottom"
    assert title_text.get_fontsize() > plot_mod._OVERVIEW_WEIGHTS_AXIS_LABEL_SIZE
    plt.close(fig)


def test_setup_weights_overview_capitalizes_labels_and_rotates_member_ticks(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    _add_weights_event(
        project_dir,
        step_idx=0,
        observable="scf",
        date_str="20230501",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.1, "sigma": 0.2, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 0.1, "sigma": 0.2, "log_weight": -1.2, "weight": 0.4},
        ],
    )

    fig = _render_setup_weights_overview_figure(project_dir, monkeypatch)
    weight_ax, residual_ax = _overview_axis_pairs(fig)[0]

    assert weight_ax.get_xlabel() == "Weight"
    assert weight_ax.get_ylabel() == "Sorted member"
    assert residual_ax.get_xlabel() == "Residual [-]"
    assert weight_ax.xaxis.labelpad == pytest.approx(plot_mod._OVERVIEW_XLABEL_PAD)
    assert residual_ax.xaxis.labelpad == pytest.approx(plot_mod._OVERVIEW_XLABEL_PAD)
    assert weight_ax.xaxis.majorTicks[0].get_pad() == pytest.approx(plot_mod._OVERVIEW_XTICK_PAD)
    assert residual_ax.xaxis.majorTicks[0].get_pad() == pytest.approx(plot_mod._OVERVIEW_XTICK_PAD)
    assert weight_ax.xaxis.label.get_fontsize() == pytest.approx(plot_mod._OVERVIEW_WEIGHTS_AXIS_LABEL_SIZE)
    assert residual_ax.xaxis.label.get_fontsize() == pytest.approx(plot_mod._OVERVIEW_WEIGHTS_AXIS_LABEL_SIZE)
    assert weight_ax.yaxis.label.get_fontsize() == pytest.approx(plot_mod._OVERVIEW_WEIGHTS_AXIS_LABEL_SIZE)
    assert weight_ax.yaxis.labelpad == pytest.approx(plot_mod._OVERVIEW_YLABEL_PAD)
    assert weight_ax.yaxis.majorTicks[0].get_pad() == pytest.approx(plot_mod._OVERVIEW_YTICK_PAD)
    assert {label.get_fontsize() for label in weight_ax.get_xticklabels() if label.get_text()} == {
        plot_mod.OVERVIEW_XTICK_SIZE
    }
    assert {label.get_fontsize() for label in residual_ax.get_xticklabels() if label.get_text()} == {
        plot_mod.OVERVIEW_XTICK_SIZE
    }
    visible_y_labels = [label for label in weight_ax.get_yticklabels() if label.get_visible()]
    assert visible_y_labels
    assert {label.get_rotation() for label in visible_y_labels} == {90.0}
    assert {label.get_fontsize() for label in visible_y_labels if label.get_text()} == {plot_mod.OVERVIEW_YTICK_SIZE}
    assert all(label.get_fontsize() < residual_ax.get_xticklabels()[0].get_fontsize() for label in visible_y_labels)
    assert weight_ax.collections[0].get_sizes()[0] == pytest.approx(13.0 * 0.72)
    assert residual_ax.collections[0].get_sizes()[0] == pytest.approx(20.0 * 0.72)
    plt.close(fig)


def test_setup_weights_overview_uses_moderately_narrower_weight_axis(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    _add_weights_event(
        project_dir,
        step_idx=0,
        observable="scf",
        date_str="20230501",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.2, "sigma": 0.1, "log_weight": -1.0, "weight": 0.6},
            {"member_id": "member_002", "residual": 0.3, "sigma": 0.1, "log_weight": -1.2, "weight": 0.4},
        ],
    )

    fig = _render_setup_weights_overview_figure(project_dir, monkeypatch)
    weight_ax, residual_ax = _overview_axis_pairs(fig)[0]
    expected_ratio = plot_mod._OVERVIEW_WEIGHTS_PANEL_WIDTH_RATIOS[0] / plot_mod._OVERVIEW_WEIGHTS_PANEL_WIDTH_RATIOS[1]

    assert weight_ax.get_position().width / residual_ax.get_position().width == pytest.approx(expected_ratio)
    assert plot_mod._OVERVIEW_WEIGHTS_PANEL_WIDTH_RATIOS[0] == pytest.approx(
        plot_mod._WEIGHTS_PANEL_WIDTH_RATIOS[0] * plot_mod._OVERVIEW_WEIGHT_AXIS_WIDTH_SCALE
    )
    assert plot_mod._OVERVIEW_WEIGHT_AXIS_WIDTH_SCALE == pytest.approx(0.85)
    plt.close(fig)


def test_setup_weights_overview_uses_panel_local_station_legend_and_sigma_strip(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
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
    _add_weights_event(
        project_dir,
        step_idx=0,
        observable="station_hs",
        date_str="20221122",
        weights_rows=[
            {"member_id": "member_001", "residual": -0.1, "sigma": 0.17, "log_weight": -1.0, "weight": 0.7},
            {"member_id": "member_002", "residual": 0.2, "sigma": 0.17, "log_weight": -2.0, "weight": 0.3},
        ],
        diag_rows=[
            {"station_id": "latschbloder", "member_id": "member_001", "residual": -0.1, "sigma": 0.29},
            {"station_id": "latschbloder", "member_id": "member_002", "residual": 0.2, "sigma": 0.29},
            {"station_id": "proviantdepot", "member_id": "member_001", "residual": -0.05, "sigma": 0.05},
            {"station_id": "proviantdepot", "member_id": "member_002", "residual": 0.1, "sigma": 0.05},
        ],
    )

    fig = _render_setup_weights_overview_figure(project_dir, monkeypatch)
    _weight_ax, residual_ax = _overview_axis_pairs(fig)[0]
    legends = _axis_legends(residual_ax)
    legend_label_sets = [[text.get_text() for text in legend.get_texts()] for legend in legends]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    station_legend = next(legend for legend in legends if [text.get_text() for text in legend.get_texts()] == ["Latschbloder", "Proviantdepot"])
    sigma_legend = next(legend for legend in legends if [text.get_text() for text in legend.get_texts()] == ["σ=0.29", "σ=0.05"])

    assert ["Latschbloder", "Proviantdepot"] in legend_label_sets
    assert ["σ=0.29", "σ=0.05"] in legend_label_sets
    station_first_text_y = station_legend.get_texts()[0].get_window_extent(renderer=renderer).y0
    sigma_first_text_y = sigma_legend.get_texts()[0].get_window_extent(renderer=renderer).y0
    assert sigma_first_text_y == pytest.approx(
        station_first_text_y,
        abs=3.0,
    )
    station_marker_size = station_legend.legend_handles[0].get_markersize()
    assert station_marker_size == pytest.approx(plot_mod._OVERVIEW_STATION_LEGEND_MARKER_SIZE)
    assert all(handle.get_markersize() == pytest.approx(station_marker_size) for handle in sigma_legend.legend_handles)
    assert fig.legends == []
    plt.close(fig)


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
        ("Latschbloder (σ=500%)", da_variable_style("station_hs")["line"]),
        ("Proviantdepot (σ=10%)", da_variable_style("station_swe")["line"]),
        ("WSF", da_variable_style("wet_snow")["line"]),
        ("SCF", da_variable_style("scf")["line"]),
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


def test_weights_grid_uses_shared_plot_style() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        plot_mod._apply_grid(ax)
        gridline = ax.xaxis.get_gridlines()[0]
        assert gridline.get_linestyle() == GRID_LS
        assert gridline.get_linewidth() == pytest.approx(GRID_LW)
        assert gridline.get_alpha() == pytest.approx(GRID_ALPHA)
    finally:
        plt.close(fig)


def test_station_color_config_resolves_aliases_and_skips_reserved_colors(tmp_path: Path) -> None:
    _setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    _write_text(
        project_dir / "plots.yml",
        "\n".join(
            [
                "weights:",
                "  station_colors:",
                "    station_hs:",
                "      proviantdepot: station_hs",
                "    station_swe:",
                "      station_b: '#123456'",
            ]
        )
        + "\n",
    )

    config = plot_mod._load_weights_station_color_config(project_dir)
    color_map = plot_mod._station_color_map(
        ["latschbloder", "proviantdepot"],
        observable="station_hs",
        station_color_config=config,
    )

    assert config == {
        "station_hs": {"proviantdepot": da_variable_style("station_hs")["line"]},
        "station_swe": {"station_b": "#123456"},
    }
    assert color_map == {
        "latschbloder": da_variable_style("station_swe")["line"],
        "proviantdepot": da_variable_style("station_hs")["line"],
    }


def test_marker_legend_entries_use_configured_station_colors(tmp_path: Path) -> None:
    setup_dir, project_dir, step_dir = _build_project_tree(tmp_path)
    _write_csv(
        setup_dir / "meteo" / "stations.csv",
        [
            {"id": "latschbloder", "name": "Latschbloder"},
            {"id": "proviantdepot", "name": "Proviantdepot"},
        ],
    )
    _write_text(
        project_dir / "plots.yml",
        "weights:\n  station_colors:\n    station_hs:\n      proviantdepot: station_hs\n",
    )
    station_csv = step_dir / "assim" / "weights_station_hs_20221122.csv"
    _write_csv(
        station_csv,
        [{"member_id": "member_001", "residual": -0.1, "sigma": 0.29, "log_weight": -1.0, "weight": 0.7}],
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
        ("Latschbloder", da_variable_style("station_swe")["line"]),
        ("Proviantdepot", da_variable_style("station_hs")["line"]),
    ]


def test_station_plot_uses_inside_sigma_strip_and_shared_bottom_legend(tmp_path: Path) -> None:
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

    assert ax1.get_xlabel() == "residual [m]"
    assert sigma_labels == ["σ=0.29", "σ=0.05"]
    assert sigma_legend._legend_box.align == "right"
    assert sigma_legend._loc == 1
    assert getattr(sigma_legend, "_ncols", None) == 1
    assert ax_bbox.x0 <= sigma_bbox.x0
    assert ax_bbox.x1 - 8.0 <= sigma_bbox.x1 <= ax_bbox.x1
    assert sigma_bbox.y1 <= ax_bbox.y1 - 0.03 * ax_bbox.height
    assert sigma_bbox.y0 >= ax_bbox.y0 - 2.0
    sigma_text_bboxes = [text.get_window_extent(renderer=renderer) for text in sigma_legend.get_texts()]
    assert sigma_text_bboxes[0].y0 > sigma_text_bboxes[1].y0
    assert bottom_labels == [
        "Latschbloder (σ=500%)",
        "Proviantdepot (σ=10%)",
        "redrawn source member (extra rings = repeated draws)",
    ]
    plt.close(fig)


def test_station_plot_with_five_sigma_entries_stacks_inside_panel(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    _write_csv(
        setup_dir / "meteo" / "stations.csv",
        [
            {"id": "station_a", "name": "Station A"},
            {"id": "station_b", "name": "Station B"},
            {"id": "station_c", "name": "Station C"},
            {"id": "station_d", "name": "Station D"},
            {"id": "station_e", "name": "Station E"},
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
            {"station_id": "station_a", "member_id": "member_001", "residual": -0.1, "sigma": 0.29},
            {"station_id": "station_b", "member_id": "member_001", "residual": -0.05, "sigma": 0.05},
            {"station_id": "station_c", "member_id": "member_001", "residual": 0.08, "sigma": 0.11},
            {"station_id": "station_d", "member_id": "member_001", "residual": 0.03, "sigma": 0.13},
            {"station_id": "station_e", "member_id": "member_001", "residual": -0.02, "sigma": 0.19},
            {"station_id": "station_a", "member_id": "member_002", "residual": 0.2, "sigma": 0.29},
            {"station_id": "station_b", "member_id": "member_002", "residual": 0.1, "sigma": 0.05},
            {"station_id": "station_c", "member_id": "member_002", "residual": -0.04, "sigma": 0.11},
            {"station_id": "station_d", "member_id": "member_002", "residual": 0.06, "sigma": 0.13},
            {"station_id": "station_e", "member_id": "member_002", "residual": -0.06, "sigma": 0.19},
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
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sigma_bbox = sigma_legend.get_window_extent(renderer=renderer)
    ax_bbox = ax1.get_window_extent(renderer=renderer)

    assert sigma_labels == ["σ=0.29", "σ=0.05", "σ=0.11", "σ=0.13", "σ=0.19"]
    assert sigma_legend._loc == 1
    assert getattr(sigma_legend, "_ncols", None) == 1
    assert ax_bbox.x0 <= sigma_bbox.x0
    assert ax_bbox.x1 - 8.0 <= sigma_bbox.x1 <= ax_bbox.x1
    assert sigma_bbox.y1 <= ax_bbox.y1 - 0.03 * ax_bbox.height
    assert sigma_bbox.y0 >= ax_bbox.y0 - 2.0
    sigma_text_bboxes = [text.get_window_extent(renderer=renderer) for text in sigma_legend.get_texts()]
    assert all(upper.y0 > lower.y0 for upper, lower in zip(sigma_text_bboxes, sigma_text_bboxes[1:]))
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

    assert ax1.get_xlabel() == "residual [-]"
    assert sigma_labels == ["σ=0.10"]
    assert sigma_legend._legend_box.align == "right"
    assert sigma_legend._loc == 1
    assert ax_bbox.x0 <= sigma_bbox.x0
    assert ax_bbox.x1 - 8.0 <= sigma_bbox.x1 <= ax_bbox.x1
    assert sigma_bbox.y1 <= ax_bbox.y1 - 0.03 * ax_bbox.height
    assert sigma_bbox.y0 >= ax_bbox.y0 - 2.0
    assert bottom_labels == [
        "SCF",
        "redrawn source member (extra rings = repeated draws)",
    ]
    plt.close(fig)


def test_wet_snow_line_plot_omits_zero_line_label_and_shows_unavailable_event(tmp_path: Path) -> None:
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
    _write_text(step_dir / "assim" / "resample_manifest_20230523.json", '{"skipped": true}\n')
    _write_csv(
        step_dir / "assim" / "resample_indices_20230523.csv",
        [{"source_member_id": "member_001"}, {"source_member_id": "member_002"}],
    )

    fig = plot_mod._plot(
        csv_path,
        plot_mod._load_weights(csv_path),
        title="Wet snow line data assimilation weights",
        subtitle="DA 13 - 2023-05-23",
        observable="wet_snow_line",
        backend="Agg",
    )

    ax1 = fig.axes[1]
    note_texts = [text.get_text() for text in ax1.texts]

    assert ax1.get_xlabel() == "residual [m]"
    assert not any("obs wet snow line" in text for text in note_texts)
    assert any("Wet snow line unavailable" in text for text in note_texts)
    assert not any("model range" in text for text in note_texts)
    plt.close(fig)


def test_wet_snow_line_gated_event_shows_unavailable_with_residuals(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_wet_snow_line_20230328.csv"
    _write_csv(
        csv_path,
        [
            {
                "member_id": "member_001",
                "residual": -30.0,
                "sigma": 150.0,
                "weight": 1.0,
                "log_weight": 0.0,
                "value_obs": 2280.0,
                "model_gate_triggered": True,
                "model_gate_reason": "model_finite_fraction<0.9000",
            },
            {
                "member_id": "member_002",
                "residual": 45.0,
                "sigma": 150.0,
                "weight": 1.0,
                "log_weight": 0.0,
                "value_obs": 2280.0,
                "model_gate_triggered": True,
                "model_gate_reason": "model_finite_fraction<0.9000",
            },
        ],
    )
    _write_text(step_dir / "assim" / "resample_manifest_20230328.json", '{"skipped": true}\n')
    _write_csv(
        step_dir / "assim" / "resample_indices_20230328.csv",
        [{"source_member_id": "member_001"}, {"source_member_id": "member_002"}],
    )

    fig = plot_mod._plot(
        csv_path,
        plot_mod._load_weights(csv_path),
        title="Wet snow line data assimilation weights",
        subtitle="DA 7 - 2023-03-28",
        observable="wet_snow_line",
        backend="Agg",
    )

    ax1 = fig.axes[1]
    note_texts = [text.get_text() for text in ax1.texts]
    sigma_labels = [text.get_text() for text in ax1.get_legend().get_texts()]

    assert any("Wet snow line unavailable" in text for text in note_texts)
    assert sigma_labels == ["σ=150"]
    plt.close(fig)


def test_wet_snow_line_skipped_resampling_does_not_show_unavailable_when_not_gated(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, _project_dir, step_dir = _build_project_tree(tmp_path)
    csv_path = step_dir / "assim" / "weights_wet_snow_line_20230503.csv"
    _write_csv(
        csv_path,
        [
            {
                "member_id": "member_001",
                "residual": -30.0,
                "sigma": 150.0,
                "weight": 1.0,
                "log_weight": 0.0,
                "value_obs": 2280.0,
                "model_gate_triggered": False,
            },
            {
                "member_id": "member_002",
                "residual": 45.0,
                "sigma": 150.0,
                "weight": 1.0,
                "log_weight": 0.0,
                "value_obs": 2280.0,
                "model_gate_triggered": False,
            },
        ],
    )
    _write_text(step_dir / "assim" / "resample_manifest_20230503.json", '{"skipped": true}\n')
    _write_csv(
        step_dir / "assim" / "resample_indices_20230503.csv",
        [{"source_member_id": "member_001"}, {"source_member_id": "member_002"}],
    )

    fig = plot_mod._plot(
        csv_path,
        plot_mod._load_weights(csv_path),
        title="Wet snow line data assimilation weights",
        subtitle="DA 10 - 2023-05-03",
        observable="wet_snow_line",
        backend="Agg",
    )

    note_texts = [text.get_text() for text in fig.axes[1].texts]

    assert not any("Wet snow line unavailable" in text for text in note_texts)
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
        ("Latschbloder", da_variable_style("station_hs")["line"]),
        ("Proviantdepot", da_variable_style("station_swe")["line"]),
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


def test_metrics_label_is_inside_weight_panel_bottom_right(tmp_path: Path) -> None:
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
    assert metrics_text.get_position() == (0.97, 0.04)
    assert metrics_text.get_ha() == "right"
    assert metrics_text.get_va() == "bottom"
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

    fig = plt.figure(figsize=plot_mod._WEIGHTS_FIGSIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=plot_mod._WEIGHTS_PANEL_WIDTH_RATIOS)
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
    assert ax1.get_position().width / ax0.get_position().width == pytest.approx(
        plot_mod._WEIGHTS_PANEL_WIDTH_RATIOS[1] / plot_mod._WEIGHTS_PANEL_WIDTH_RATIOS[0]
    )
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
    ax1 = fig.axes[1]
    assert title_text.get_position() == (0.11, plot_mod._STANDALONE_TITLE_Y)
    assert title_text.get_fontsize() == pytest.approx(plot_mod._FS_TITLE)
    assert legend_anchor.y0 == pytest.approx(plot_mod._STANDALONE_LEGEND_Y)
    assert ax1.get_position().width / ax0.get_position().width > 2.6
    assert list(ax0.get_xticks()) == plot_mod._WEIGHT_AXIS_TICKS
    assert [tick.get_text() for tick in ax0.get_xticklabels()] == plot_mod._WEIGHT_AXIS_TICK_LABELS
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
        plot_mod._marker_handle(da_variable_style("station_hs")["line"]),
        plot_mod._marker_handle(da_variable_style("station_swe")["line"]),
        plot_mod._marker_handle(da_variable_style("wet_snow")["line"]),
        plot_mod._marker_handle(da_variable_style("scf")["line"]),
    ]
    labels = [
        "Latschbloder (σ=500%)",
        "Proviantdepot (σ=10%)",
        "WSF",
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
    residual_axes = _axes_with_xlabel(fig, "Residual [m]")
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
    hs_axes = _residual_axes_for_title(fig, "Station snow depth")
    swe_axes = _residual_axes_for_title(fig, "Station SWE")
    scf_axes = _residual_axes_for_title(fig, "Snow cover")
    wet_axes = _residual_axes_for_title(fig, "Wet snow fraction")

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
    residual_axes = _axes_with_xlabel(fig, "Residual [-]")
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
    residual_axes = _axes_with_xlabel(fig, "Residual [m]")
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
    residual_axes = _axes_with_xlabel(fig, "Residual [m]")
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
    weight_axes = _axes_with_xlabel(fig, "Weight")
    residual_axes = _axes_with_xlabel(fig, "Residual [-]")

    assert len(weight_axes) == 1
    assert [int(tick) for tick in weight_axes[0].get_yticks()] == [1, 10, 20, 30, 40]
    assert len(weight_axes[0].yaxis.get_minorticklocs()) == 0
    assert len(residual_axes) == 1
    assert len(residual_axes[0].yaxis.get_minorticklocs()) == 0
    plt.close(fig)


def test_setup_overview_hides_right_column_y_tick_labels(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    for step_idx, date_str in enumerate(("20230518", "20230526")):
        _add_weights_event(
            project_dir,
            step_idx=step_idx,
            observable="scf",
            date_str=date_str,
            weights_rows=[
                {"member_id": "member_001", "residual": -0.1, "sigma": 0.2, "log_weight": -1.0, "weight": 0.6},
                {"member_id": "member_002", "residual": 0.1, "sigma": 0.2, "log_weight": -1.2, "weight": 0.4},
            ],
        )

    fig = _render_setup_weights_overview_figure(project_dir, monkeypatch)
    pairs = _overview_axis_pairs(fig)
    left_weight_ax, left_residual_ax = pairs[0]
    right_weight_ax, right_residual_ax = pairs[1]
    left_title = next(text.get_text() for text in left_weight_ax.texts if "2023-05-18" in text.get_text())
    right_title = next(text.get_text() for text in right_weight_ax.texts if "2023-05-26" in text.get_text())

    assert left_title.startswith("(a) DA 1")
    assert right_title.startswith("(b) DA 2")
    assert any(tick.label1.get_visible() for tick in left_weight_ax.yaxis.get_major_ticks())
    assert all(not tick.label1.get_visible() for tick in left_residual_ax.yaxis.get_major_ticks())
    assert all(not tick.label1.get_visible() for tick in right_weight_ax.yaxis.get_major_ticks())
    assert all(not tick.label1.get_visible() for tick in right_residual_ax.yaxis.get_major_ticks())
    plt.close(fig)


def test_setup_overview_splits_many_events_across_multiple_pages(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    _setup_dir, project_dir, _step_dir = _build_project_tree(tmp_path)
    for step_idx in range(15):
        day = step_idx + 1
        _add_weights_event(
            project_dir,
            step_idx=step_idx,
            observable="scf",
            date_str=f"202305{day:02d}",
            weights_rows=[
                {
                    "member_id": "member_001",
                    "residual": -0.1,
                    "sigma": 0.2,
                    "log_weight": -1.0,
                    "weight": 0.6,
                },
                {
                    "member_id": "member_002",
                    "residual": 0.1,
                    "sigma": 0.2,
                    "log_weight": -1.2,
                    "weight": 0.4,
                },
            ],
        )

    saved = _render_setup_weights_overview_pages(project_dir, monkeypatch)
    normal_saved = [saved[0], saved[2]]
    paper_saved = [saved[1], saved[3]]

    assert len(saved) == 4
    assert normal_saved[0]["out"] == project_dir / "results" / "plots" / "assim" / "weights" / "setup_weights_overview_2022_2023.png"
    assert normal_saved[1]["out"] == project_dir / "results" / "plots" / "assim" / "weights" / "setup_weights_overview_2022_2023_page_02.png"
    assert paper_saved[0]["out"] == plot_mod.project_paper_output_path(project_dir, normal_saved[0]["out"])
    assert paper_saved[1]["out"] == plot_mod.project_paper_output_path(project_dir, normal_saved[1]["out"])
    assert normal_saved[0]["fig"].get_figheight() == pytest.approx(
        plot_mod._COMPOSITE_ROW_HEIGHT * plot_mod._OVERVIEW_MAX_ROWS_PER_PAGE
    )
    assert normal_saved[1]["fig"].get_figheight() == pytest.approx(
        plot_mod._COMPOSITE_ROW_HEIGHT * plot_mod._OVERVIEW_MAX_ROWS_PER_PAGE
    )
    assert paper_saved[0]["fig"].get_figheight() == pytest.approx(
        plot_mod._COMPOSITE_ROW_HEIGHT * plot_mod._OVERVIEW_MAX_ROWS_PER_PAGE
    )
    assert paper_saved[1]["fig"].get_figheight() == pytest.approx(
        plot_mod._COMPOSITE_ROW_HEIGHT * plot_mod._OVERVIEW_MAX_ROWS_PER_PAGE
    )
    assert "page 1/2" in normal_saved[0]["fig"].texts[0].get_text()
    assert "page 2/2" in normal_saved[1]["fig"].texts[0].get_text()
    assert not any(text.get_text().startswith("Data assimilation weights") for text in paper_saved[0]["fig"].texts)
    assert not any(text.get_text().startswith("Data assimilation weights") for text in paper_saved[1]["fig"].texts)
    renderer = normal_saved[1]["fig"].canvas.get_renderer()
    summary_box = normal_saved[1]["fig"].texts[0].get_window_extent(renderer)
    event_title_boxes = [
        text.get_window_extent(renderer)
        for ax in normal_saved[1]["fig"].axes
        for text in ax.texts
        if "Snow cover" in text.get_text()
    ]
    assert event_title_boxes
    assert all(not summary_box.overlaps(box) for box in event_title_boxes)

    for item in saved:
        plt.close(item["fig"])
