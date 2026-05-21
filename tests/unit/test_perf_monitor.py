from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

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


def test_project_perf_csv_appends_disk_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "project_perf_metrics.csv"
    t = datetime(2026, 1, 1, 12, 0)

    perf_monitor._append_csv_row(
        csv_path,
        t,
        cpu_total_pct=12.3456,
        mem_used_pct=45.6789,
        mem_used_gb=10.1234,
        mem_total_gb=64.0,
        disk_fs_used_pct=37.5,
        disk_fs_used_gb=1500.0,
        disk_fs_free_gb=2500.0,
        disk_fs_total_gb=4000.0,
        disk_project_used_gb=123.4567,
    )

    lines = csv_path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == (
        "timestamp,cpu_total_pct,mem_used_pct,mem_used_gb,mem_total_gb,"
        "disk_fs_used_pct,disk_fs_used_gb,disk_fs_free_gb,"
        "disk_fs_total_gb,disk_project_used_gb"
    )
    assert lines[1] == "2026-01-01T12:00:00,12.346,45.679,10.123,64.000,37.500,1500.000,2500.000,4000.000,123.457"


def test_project_perf_plot_accepts_disk_series(tmp_path: Path) -> None:
    if perf_monitor.plt is None:
        pytest.skip("matplotlib is not available")

    start = datetime(2026, 1, 1, 12, 0)
    timestamps = [start + timedelta(minutes=idx) for idx in range(4)]
    out = tmp_path / "project_perf.png"

    perf_monitor._render_plot(
        out,
        timestamps,
        cpu_pct=[10.0, 50.0, 80.0, 30.0],
        mem_pct=[20.0, 25.0, 30.0, 26.0],
        mem_used_gb=[20.0, 25.0, 30.0, 26.0],
        mem_total_gb=[128.0, 128.0, 128.0, 128.0],
        run_start=start,
        disk_fs_used_pct=[40.0, 42.0, 43.0, 45.0],
        disk_fs_free_gb=[2200.0, 2150.0, 2125.0, 2100.0],
        disk_project_used_gb=[5.0, 20.0, 50.0, 80.0],
    )

    with Image.open(out) as image:
        width, height = image.size

    assert width > 0
    assert height > 0


def test_project_size_scan_is_throttled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "perf"
    out_dir.mkdir()
    scans: list[Path] = []

    class StopAfterThreeSamples:
        def __init__(self) -> None:
            self.waits = 0

        def is_set(self) -> bool:
            return self.waits >= 3

        def wait(self, _interval: float) -> None:
            self.waits += 1

    fake_vm = SimpleNamespace(percent=25.0, used=8 * 1024**3, total=32 * 1024**3)
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: fake_vm,
        cpu_percent=lambda interval=None: 50.0,
    )
    monkeypatch.setattr(perf_monitor, "psutil", fake_psutil)
    monkeypatch.setattr(perf_monitor, "plt", None)
    monkeypatch.setattr(perf_monitor, "_filesystem_disk_usage_gb", lambda _path: (40.0, 400.0, 600.0, 1000.0))

    def _fake_directory_size(path: Path) -> float:
        scans.append(path)
        return 12.0

    monkeypatch.setattr(perf_monitor, "_directory_size_gb", _fake_directory_size)

    perf_monitor._monitor_loop(
        perf_monitor.PerfMonitorConfig(
            project_dir=tmp_path,
            sample_interval_sec=0.0,
            plot_interval_sec=9999.0,
            disk_scan_interval_sec=9999.0,
            run_start=datetime(2026, 1, 1, 12, 0),
        ),
        out_dir,
        StopAfterThreeSamples(),
    )

    rows = (out_dir / "project_perf_metrics.csv").read_text(encoding="utf-8").splitlines()

    assert len(rows) == 4
    assert scans == [tmp_path]
    assert all(row.endswith(",12.000") for row in rows[1:])
