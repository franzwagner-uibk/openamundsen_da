from __future__ import annotations

import re
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from openamundsen_da.methods.viz.reports.project_collection_pdf import (
    MissingProjectPdfArtifactsError,
    _image_physical_size,
    build_project_collection_pdf,
    cli_main,
    collect_project_pdf_items,
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
    assert [(item.map_path.name, item.weights_path.name) for item in plan.da_steps] == [
        ("da_1.png", "DA_01_weights.png"),
        ("da_2.png", "DA_02_weights.png"),
    ]
    assert [item.path.relative_to(project_dir).as_posix() for item in plan.appendix_items] == [
        "results/maps/custom_map.png",
        "results/plots/perf/project_perf.png",
    ]


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
    assert "DA_01_weights.png" in text
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
    assert _pdf_page_count(output) == 6


def test_cli_main_writes_project_collection_pdf(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=1)
    output = tmp_path / "cli.pdf"

    rc = cli_main(["--project-dir", str(project_dir), "--output", str(output)])

    assert rc == 0
    assert output.is_file()


def test_image_physical_size_uses_png_dpi(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _write_png(path, width=400, height=200)

    width, height = _image_physical_size(path)

    assert width == pytest.approx(2.0, abs=0.01)
    assert height == pytest.approx(1.0, abs=0.01)
