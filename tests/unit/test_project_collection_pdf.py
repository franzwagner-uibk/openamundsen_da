from __future__ import annotations

import re
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from openamundsen_da.methods.viz.reports.project_collection_pdf import (
    MissingProjectPdfArtifactsError,
    _fit_rect,
    build_project_collection_pdf,
    cli_main,
    collect_project_pdf_items,
    collect_project_report_summary,
)


def _write_project_yaml(project_dir: Path, event_count: int = 2) -> None:
    events = "\n".join(
        f"  - date: '2023-01-{idx:02d}'\n    variable: scf\n    product: TEST" for idx in range(1, event_count + 1)
    )
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{project_dir.name}.yml").write_text(
        "start_date: '2023-01-01'\n"
        "end_date: '2023-01-31'\n"
        "data_assimilation:\n"
        "  assimilation_events:\n"
        f"{events}\n",
        encoding="utf-8",
    )


def _write_png(path: Path, *, width: int = 80, height: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=float)
    image[:, :, 0] = np.linspace(0.0, 1.0, width)
    image[:, :, 1] = np.linspace(0.0, 1.0, height)[:, None]
    mpimg.imsave(path, image, dpi=200.0)


def _create_project(tmp_path: Path, *, event_count: int = 2) -> Path:
    project_dir = tmp_path / "setup" / "projects" / "project_2023"
    _write_project_yaml(project_dir, event_count=event_count)
    _write_png(project_dir / "results/plots/results/result_overview.png", width=60, height=90)
    _write_png(project_dir / "results/maps/setup_overview.png", width=90, height=60)
    _write_png(project_dir / "results/plots/assim/weights/setup_weights_overview_2023.png", width=70, height=90)
    for idx in range(1, event_count + 1):
        _write_png(project_dir / f"results/maps/da_events/da_{idx}.png", width=100, height=35)
        _write_png(project_dir / f"results/plots/assim/weights/DA_{idx:02d}_weights.png", width=90, height=35)
    return project_dir


def _pdf_page_count(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))


def test_collect_project_pdf_items_orders_front_da_and_appendix(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=2)
    _write_png(project_dir / "results/plots/results/result_overview_custom.png", width=80, height=80)
    _write_png(project_dir / "results/plots/assim/weights/setup_weights_overview_2023_page_02.png", width=70, height=90)
    _write_png(project_dir / "results/plots/perf/project_perf.png", width=90, height=40)
    _write_png(project_dir / "results/maps/custom_map.png", width=90, height=60)

    plan = collect_project_pdf_items(project_dir)

    assert plan.missing_paths == ()
    assert [item.path.name for item in plan.front_items] == [
        "result_overview.png",
        "result_overview_custom.png",
        "setup_overview.png",
        "setup_weights_overview_2023.png",
        "setup_weights_overview_2023_page_02.png",
    ]
    assert [item.map_path.name for item in plan.da_steps] == ["da_1.png", "da_2.png"]
    assert plan.appendix_items == ()
    assert plan.page_count == 8


def test_collect_project_report_summary_reads_cost_stats(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=1)
    (project_dir / "project_2023.log").write_text(
        "2026-01-01 00:00:00 | INFO | Launching ensemble with max_workers=8\n"
        "2026-01-01 00:02:03 | INFO | Project processing complete: "
        "/data/projects/project_2023 (wall-clock 123.4 s, ~0.03 h)\n",
        encoding="utf-8",
    )
    perf_dir = project_dir / "results/plots/perf"
    perf_dir.mkdir(parents=True, exist_ok=True)
    (perf_dir / "project_perf_metrics.csv").write_text(
        "timestamp,cpu_total_pct,mem_used_pct,mem_used_gb,mem_total_gb\n"
        "2026-01-01T00:00:00,10.0,20.0,2.0,16.0\n"
        "2026-01-01T00:01:00,99.5,50.0,5.8,16.0\n",
        encoding="utf-8",
    )

    summary = collect_project_report_summary(project_dir)
    cost = next(section for section in summary.sections if section.title == "Computing Cost")

    assert "Max workers/cores: 8" in cost.lines
    assert "Runtime: 2m 03s (project log)" in cost.lines
    assert "Peak CPU: 99.5%" in cost.lines
    assert "Peak RAM: 5.8 GB (50.0%)" in cost.lines
    assert "Total RAM: 16.0 GB" in cost.lines
    assert "Perf samples: 2026-01-01 00:00:00 to 2026-01-01 00:01:00" in cost.lines


def test_collect_project_report_summary_handles_missing_cost_stats(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=1)

    summary = collect_project_report_summary(project_dir)
    cost = next(section for section in summary.sections if section.title == "Computing Cost")

    assert "Max workers/cores: n/a" in cost.lines
    assert "Runtime: n/a" in cost.lines


def test_build_project_collection_pdf_fails_with_all_missing_core_paths(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2023"
    _write_project_yaml(project_dir, event_count=1)
    output = tmp_path / "out.pdf"

    with pytest.raises(MissingProjectPdfArtifactsError) as exc_info:
        build_project_collection_pdf(project_dir=project_dir, output=output)

    text = str(exc_info.value)
    assert "result_overview.png" in text
    assert "setup_overview.png" in text
    assert "setup_weights_overview_2023.png" in text
    assert "da_1.png" in text
    assert "oa-da-plot-project-plots" in text
    assert "oa-da-plot-project-maps" in text
    assert not output.exists()


def test_build_project_collection_pdf_writes_expected_page_count(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=1)
    _write_png(project_dir / "results/plots/perf/project_perf.png", width=90, height=40)
    output = tmp_path / "collection.pdf"

    written = build_project_collection_pdf(project_dir=project_dir, output=output)

    assert written == output
    assert output.is_file()
    assert _pdf_page_count(output) == 5


def test_cli_main_writes_project_collection_pdf(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=1)
    output = tmp_path / "cli.pdf"

    rc = cli_main(["--project-dir", str(project_dir), "--output", str(output)])

    assert rc == 0
    assert output.is_file()


def test_fit_rect_preserves_aspect_ratio() -> None:
    left, bottom, width, height = _fit_rect(400, 200, (0.0, 0.0, 8.0, 8.0))

    assert left == pytest.approx(0.0)
    assert bottom == pytest.approx(4.0)
    assert width == pytest.approx(8.0)
    assert height == pytest.approx(4.0)
