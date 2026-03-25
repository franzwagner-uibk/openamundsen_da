from __future__ import annotations

from pathlib import Path

import pandas as pd

from openamundsen_da.methods.viz.plot_project_ensemble import plot_setup_results


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    step_dir = project_dir / "steps" / "step_00_init"
    _write_text(
        project_dir / "project_2022_2023.yml",
        "\n".join(
            [
                "start_date: '2022-11-01'",
                "end_date: '2023-06-30'",
                "data_assimilation:",
                "  assimilation_events:",
                "    - date: '2022-11-22'",
                "      variable: station_hs",
            ]
        )
        + "\n",
    )
    _write_text(
        step_dir / "step_00_init.yml",
        "\n".join(
            [
                "start_date: '2022-11-01'",
                "end_date: '2022-12-01'",
            ]
        )
        + "\n",
    )
    _write_csv(
        step_dir / "ensembles" / "prior" / "open_loop" / "results" / "point_latschbloder.csv",
        [
            {"time": "2022-11-01", "swe": 10.0},
            {"time": "2022-11-02", "swe": 12.0},
        ],
    )
    _write_csv(
        step_dir / "ensembles" / "prior" / "member_001" / "results" / "point_latschbloder.csv",
        [
            {"time": "2022-11-01", "swe": 8.0},
            {"time": "2022-11-02", "swe": 9.0},
        ],
    )
    _write_csv(
        step_dir / "ensembles" / "prior" / "member_002" / "results" / "point_latschbloder.csv",
        [
            {"time": "2022-11-01", "swe": 14.0},
            {"time": "2022-11-02", "swe": 15.0},
        ],
    )
    return project_dir


def test_plot_setup_results_members_mode_draws_members_without_band(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    project_dir = _build_project(tmp_path)
    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        plot_setup_results(
            setup_dir=project_dir,
            var_col="swe",
            mode="members",
            configure_logger=False,
        )
        fig = plt.gcf()
        ax = fig.axes[0]
        assert len(ax.collections) == 0
        assert len(ax.lines) == 4
        legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert legend_labels == ["open loop", "data assimilation event"]
    finally:
        plt.close = original_close
        original_close("all")


def test_plot_setup_results_band_mode_uses_quantile_band(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    project_dir = _build_project(tmp_path)
    original_close = plt.close
    plt.close = lambda fig=None: None
    try:
        plot_setup_results(
            setup_dir=project_dir,
            var_col="swe",
            mode="band",
            band_low=0.25,
            band_high=0.75,
            configure_logger=False,
        )
        fig = plt.gcf()
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        assert len(ax.lines) == 3
        legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert legend_labels == ["open loop", "ensemble mean", "ensemble", "data assimilation event"]
    finally:
        plt.close = original_close
        original_close("all")
