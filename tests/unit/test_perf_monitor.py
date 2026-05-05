from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from openamundsen_da.methods.viz.theme import EXPORT_DPI, FIGHEIGHT_OVERVIEW_ROW, FIGWIDTH_OVERVIEW_PAPER
from openamundsen_da.util import perf_monitor


def test_project_perf_plot_uses_report_overview_page_width(tmp_path: Path) -> None:
    if perf_monitor.plt is None:
        pytest.skip("matplotlib is not available")

    start = datetime(2026, 1, 1, 12, 0)
    timestamps = [start + timedelta(minutes=idx) for idx in range(4)]
    out = tmp_path / "project_perf.png"

    perf_monitor._render_plot(
        out,
        timestamps,
        cpu_pct=[0.0, 55.0, 100.0, 25.0],
        mem_pct=[20.0, 22.0, 23.0, 21.0],
        mem_used_gb=[5.0, 5.5, 5.7, 5.3],
        mem_total_gb=[24.0, 24.0, 24.0, 24.0],
        run_start=start,
    )

    with Image.open(out) as image:
        width, height = image.size

    assert width == pytest.approx(FIGWIDTH_OVERVIEW_PAPER * EXPORT_DPI, abs=2)
    assert height == pytest.approx(FIGHEIGHT_OVERVIEW_ROW * 1.4 * EXPORT_DPI, abs=2)


def test_project_perf_plot_replaces_target_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "project_perf.png"
    out.write_text("old", encoding="utf-8")
    saved_paths: list[Path] = []

    def _fake_save(_fig, path: Path, **_kwargs) -> None:
        saved_paths.append(Path(path))
        Path(path).write_text("new", encoding="utf-8")

    monkeypatch.setattr(perf_monitor, "save_figure_png", _fake_save)

    perf_monitor._save_perf_plot_atomic(object(), out)

    assert out.read_text(encoding="utf-8") == "new"
    assert len(saved_paths) == 1
    assert saved_paths[0].parent == out.parent
    assert saved_paths[0].name != out.name
    assert not saved_paths[0].exists()
