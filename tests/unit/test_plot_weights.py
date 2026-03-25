from __future__ import annotations

from pathlib import Path

import pandas as pd

from openamundsen_da.methods.pf import plot_weights as plot_mod


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
    return setup_dir, project_dir, step_dir


def test_axis_labels_use_residual_terminology() -> None:
    assert plot_mod._fraction_axis_label("scf") == "snow cover fraction residual"
    assert plot_mod._fraction_axis_label("wet_snow") == "wet-snow fraction residual"
    assert plot_mod._station_axis_label("station_hs") == "snow depth residual [m]"
    assert plot_mod._station_axis_label("station_swe") == "SWE residual [mm]"


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
