from __future__ import annotations

import re
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest
from openamundsen_da.methods.viz.reports.project_collection_pdf import (
    _content_rows,
    _format_page_range,
    _image_size_inches,
    _project_pdf_sections,
    _save_pdf_page,
    _summary_line_segments,
    _wet_snow_classification_summary,
    _wrapped_line_sources,
    build_project_collection_pdf,
    cli_main,
    collect_project_pdf_items,
    collect_project_report_summary,
)
from openamundsen_da.methods.viz.theme import EXPORT_DPI


def _write_project_yaml(project_dir: Path, event_count: int = 2) -> None:
    events = "\n".join(
        f"  - date: '2023-01-{idx:02d}'\n    variable: scf\n    product: TEST" for idx in range(1, event_count + 1)
    )
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{project_dir.name}.yml").write_text(
        "start_date: '2023-01-01'\n"
        "end_date: '2023-01-31'\n"
        "data_assimilation:\n"
        "  wet_snow:\n"
        "    classification_method: liquid_water_amount\n"
        "    liquid_water_amount_threshold_mm: 2.0\n"
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
    (tmp_path / "setup").mkdir(parents=True, exist_ok=True)
    (tmp_path / "setup" / "demo.yml").write_text(
        "domain: rofental\n"
        "resolution: 100\n"
        "timestep: 3H\n"
        "crs: epsg:25832\n"
        "meteo:\n"
        "  interpolation:\n"
        "    temperature:\n"
        "      trend_method: fixed\n"
        "    precipitation:\n"
        "      trend_method: fractional\n"
        "    humidity:\n"
        "      trend_method: fixed\n"
        "    cloudiness:\n"
        "      day_method: clear_sky_fraction\n"
        "      night_method: humidity\n"
        "    wind_speed:\n"
        "      trend_method: regression\n"
        "  precipitation_correction:\n"
        "    - method: kochendorfer\n"
        "      gauge: us_un\n"
        "    - method: srf\n"
        "snow:\n"
        "  model: multilayer\n"
        "  liquid_water_content:\n"
        "    method: pore_volume_fraction\n"
        "    max: 0.03\n"
        "  melt:\n"
        "    method: energy_balance\n"
        "canopy:\n"
        "  enabled: false\n",
        encoding="utf-8",
    )
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
    # Legacy output from older runs must not create a duplicate overview page.
    _write_png(project_dir / "results/plots/results/result_overview_custom.png", width=80, height=80)
    _write_png(project_dir / "results/plots/assim/weights/setup_weights_overview_2023_page_02.png", width=70, height=90)
    _write_png(project_dir / "results/plots/assim/scores/performance_scores.png", width=90, height=40)
    _write_png(project_dir / "results/plots/perf/project_perf.png", width=90, height=40)
    _write_png(project_dir / "results/plots/points/setup_results_point_latschbloder_snow_depth_2023.png")
    _write_png(project_dir / "results/plots/points/setup_results_point_proviantdepot_snow_depth_2023.png")
    _write_png(project_dir / "results/plots/points/setup_results_point_snow_depth_roi_snow_depth_2023.png")
    _write_png(project_dir / "results/maps/custom_map.png", width=90, height=60)

    plan = collect_project_pdf_items(project_dir)

    assert plan.missing_paths == ()
    assert [item.path.name for item in plan.front_items] == [
        "result_overview.png",
        "setup_overview.png",
        "setup_weights_overview_2023.png",
        "setup_weights_overview_2023_page_02.png",
    ]
    assert [item.path.name for item in plan.station_snow_depth_items] == [
        "setup_results_point_latschbloder_snow_depth_2023.png",
        "setup_results_point_proviantdepot_snow_depth_2023.png",
    ]
    assert plan.performance_scores_item is not None
    assert plan.performance_scores_item.path.name == "performance_scores.png"
    assert plan.project_perf_item is not None
    assert plan.project_perf_item.path.name == "project_perf.png"
    assert [item.map_path.name for item in plan.da_steps] == ["da_1.png", "da_2.png"]
    assert plan.appendix_items == ()
    assert plan.page_count == 9


def test_project_pdf_sections_follow_temporal_report_order(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=2)
    _write_png(project_dir / "results/plots/assim/scores/performance_scores.png", width=90, height=40)
    _write_png(project_dir / "results/plots/perf/project_perf.png", width=90, height=40)
    _write_png(project_dir / "results/plots/points/setup_results_point_latschbloder_snow_depth_2023.png")

    sections = _project_pdf_sections(collect_project_pdf_items(project_dir))

    assert [(section.title, _format_page_range(section.start_page, section.end_page)) for section in sections] == [
        ("Project summary and setup", "1"),
        ("result overview", "2"),
        ("setup overview map", "3"),
        ("setup weights overview", "4"),
        ("station snow-depth plots", "5"),
        ("performance scores", "6"),
        ("project performance", "7"),
        ("DA-event maps", "8"),
    ]
    assert _content_rows(sections) == (
        ("1", "Project summary and setup"),
        ("2", "result overview"),
        ("3", "setup overview map"),
        ("4", "setup weights overview"),
        ("5", "station snow-depth plots"),
        ("6", "performance scores"),
        ("7", "project performance"),
        ("8", "DA-event maps"),
    )


def test_project_pdf_sections_paginate_many_station_snow_depth_plots(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=1)
    for idx in range(1, 7):
        _write_png(
            project_dir / f"results/plots/points/setup_results_point_{idx:02d}_snow_depth_2023.png",
            width=4200,
            height=1408,
        )

    plan = collect_project_pdf_items(project_dir)
    sections = _project_pdf_sections(plan)

    assert plan.page_count == 7
    assert [(section.title, _format_page_range(section.start_page, section.end_page)) for section in sections] == [
        ("Project summary and setup", "1"),
        ("result overview", "2"),
        ("setup overview map", "3"),
        ("setup weights overview", "4"),
        ("station snow-depth plots", "5-6"),
        ("DA-event maps", "7"),
    ]


def test_summary_wrapped_lines_can_render_without_truncation() -> None:
    lines = ["Liquid water content: method=pore_volume_fraction, max=0.03"]

    assert [line for line, _source in _wrapped_line_sources(lines, width=24, max_lines=None)] == [
        "Liquid water content:",
        "method=pore_volume_fraction,",
        "max=0.03",
    ]


def test_summary_line_segments_bold_important_report_values() -> None:
    assert _summary_line_segments("Run mode: single", source="Run mode: single") == (("Run mode: single", True),)
    assert _summary_line_segments(
        "Domain: rofental, resolution=100 m, timestep=3H,",
        source="Domain: rofental, resolution=100 m, timestep=3H, CRS=epsg:25832",
    ) == (
        ("Domain: rofental, ", False),
        ("resolution=100 m", True),
        (", ", False),
        ("timestep=3H", True),
        (",", False),
    )
    assert _summary_line_segments("ensemble_size=30, seed=42", source="Prior forcing: ensemble_size=30, seed=42") == (
        ("ensemble_size=30", True),
        (", seed=42", False),
    )
    assert _summary_line_segments(
        "ess_ratio=0.7, seed=42",
        source="Resampling: algorithm=systematic, ess_ratio=0.7, seed=42",
    ) == (
        ("ess_ratio=0.7", True),
        (", seed=42", False),
    )
    assert _summary_line_segments(
        "station_hs x5, wet_snow_line x5",
        source="By variable: station_hs x5, wet_snow_line x5",
    ) == (
        ("station_hs x5, wet_snow_line x5", True),
    )


def test_wet_snow_classification_summary_reports_method_specific_threshold() -> None:
    assert (
        _wet_snow_classification_summary(
            {"classification_method": "liquid_water_amount", "liquid_water_amount_threshold_mm": 2.0}
        )
        == "method=liquid_water_amount, threshold_abs_mm=2"
    )
    assert (
        _wet_snow_classification_summary(
            {"classification_method": "liquid_water_fraction", "classification_threshold_percent": 0.4}
        )
        == "method=liquid_water_fraction, threshold_pct=0.4"
    )


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
    setup = next(section for section in summary.sections if section.title == "openAMUNDSEN Setup")
    core = next(section for section in summary.sections if section.title == "Core DA Settings")

    assert "Max workers/cores: 8" in cost.lines
    assert "Runtime: 2m 03s" in cost.lines
    assert "Peak CPU: 99.5%" in cost.lines
    assert "Peak RAM: 5.8 GB (50.0%)" in cost.lines
    assert "Total RAM: 16.0 GB" in cost.lines
    assert "Domain: rofental, resolution=100 m, timestep=3H, CRS=epsg:25832" in setup.lines
    assert (
        "Meteo interpolation: temp=fixed, precip=fractional, humidity=fixed, "
        "wind=regression, cloud=clear_sky_fraction/humidity"
    ) in setup.lines
    assert "Precip correction: kochendorfer (gauge=us_un), srf" in setup.lines
    assert "Snow model: multilayer, melt=energy_balance" in setup.lines
    assert "Liquid water content: method=pore_volume_fraction, max=0.03" in setup.lines
    assert "Canopy enabled: false" in setup.lines
    assert "Wet snow classification: method=liquid_water_amount, threshold_abs_mm=2" in core.lines


def test_collect_project_report_summary_handles_missing_cost_stats(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=1)

    summary = collect_project_report_summary(project_dir)
    cost = next(section for section in summary.sections if section.title == "Computing Cost")

    assert "Max workers/cores: n/a" in cost.lines
    assert "Runtime: n/a" in cost.lines


def test_collect_project_report_summary_includes_subdomain_outputs(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=1)
    results = project_dir / "results"
    (results / "subdomain_overview.csv").write_text(
        "subdomain_id,status,duration_seconds\nsd_01,success,12\nsd_02,success,18\n",
        encoding="utf-8",
    )
    (results / "subdomain_assimilation_aggregate.csv").write_text(
        "subdomain_id,ess_norm_mean,ess_norm_min\nsd_01,0.9,0.5\nsd_02,0.7,0.2\n",
        encoding="utf-8",
    )
    (results / "subdomain_dropped_events.csv").write_text(
        "subdomain_id,date,variable,reason\nsd_02,2023-01-01,scf,cloud\n",
        encoding="utf-8",
    )

    summary = collect_project_report_summary(project_dir)
    subdomains = next(section for section in summary.sections if section.title == "Subdomains")

    assert "Statuses: success x2" in subdomains.lines
    assert "Subdomains: 2" in subdomains.lines
    assert "Slowest subdomain: 18s" in subdomains.lines
    assert "Mean ESS/n range: 0.700 to 0.900" in subdomains.lines
    assert "Weakest ESS/n: sd_02 = 0.200" in subdomains.lines
    assert "Dropped subdomain events: 1" in subdomains.lines


def test_build_project_collection_pdf_writes_summary_with_missing_artifacts(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2023"
    _write_project_yaml(project_dir, event_count=1)
    output = tmp_path / "out.pdf"

    written = build_project_collection_pdf(project_dir=project_dir, output=output)

    assert written == output
    assert output.is_file()
    assert _pdf_page_count(output) == 1


def test_build_project_collection_pdf_handles_subdomain_summary_section(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2023"
    _write_project_yaml(project_dir, event_count=1)
    results = project_dir / "results"
    results.mkdir(parents=True)
    (results / "subdomain_overview.csv").write_text(
        "subdomain_id,status,duration_seconds\nsd_01,success,12\nsd_02,success,18\n",
        encoding="utf-8",
    )
    (results / "subdomain_assimilation_aggregate.csv").write_text(
        "subdomain_id,ess_norm_mean,ess_norm_min\nsd_01,0.9,0.5\nsd_02,0.7,0.2\n",
        encoding="utf-8",
    )
    (results / "subdomain_dropped_events.csv").write_text(
        "subdomain_id,date,variable,reason\nsd_02,2023-01-01,scf,cloud\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.pdf"

    written = build_project_collection_pdf(project_dir=project_dir, output=output)

    assert written == output
    assert output.is_file()
    assert _pdf_page_count(output) == 1


def test_build_project_collection_pdf_writes_expected_page_count(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=1)
    _write_png(project_dir / "results/plots/perf/project_perf.png", width=90, height=40)
    output = tmp_path / "collection.pdf"

    written = build_project_collection_pdf(project_dir=project_dir, output=output)

    assert written == output
    assert output.is_file()
    assert _pdf_page_count(output) == 6


def test_build_project_collection_pdf_groups_wide_da_maps(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=4)
    for idx in range(1, 5):
        _write_png(project_dir / f"results/maps/da_events/da_{idx}.png", width=4200, height=3000)
    output = tmp_path / "collection.pdf"

    written = build_project_collection_pdf(project_dir=project_dir, output=output)

    assert written == output
    assert collect_project_pdf_items(project_dir).page_count == 6
    assert _pdf_page_count(output) == 6


def test_build_project_collection_pdf_paginates_many_station_snow_depth_plots(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=1)
    for idx in range(1, 7):
        _write_png(
            project_dir / f"results/plots/points/setup_results_point_{idx:02d}_snow_depth_2023.png",
            width=4200,
            height=1408,
        )
    output = tmp_path / "collection.pdf"

    written = build_project_collection_pdf(project_dir=project_dir, output=output)

    assert written == output
    assert collect_project_pdf_items(project_dir).page_count == 7
    assert _pdf_page_count(output) == 7


def test_cli_main_writes_project_collection_pdf(tmp_path: Path) -> None:
    project_dir = _create_project(tmp_path, event_count=1)
    output = tmp_path / "cli.pdf"

    rc = cli_main(["--project-dir", str(project_dir), "--output", str(output)])

    assert rc == 0
    assert output.is_file()


def test_python_module_entry_point_calls_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    from openamundsen_da.methods.viz.reports import __main__ as reports_main

    calls = []

    def fake_cli_main() -> int:
        calls.append(True)
        return 17

    monkeypatch.setattr(reports_main, "cli_main", fake_cli_main)

    assert reports_main.main() == 17
    assert calls == [True]


def test_image_size_inches_uses_shared_export_dpi(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _write_png(path, width=1200, height=600)

    width, height = _image_size_inches(path)

    assert width == pytest.approx(2.0)
    assert height == pytest.approx(1.0)


def test_save_pdf_page_uses_shared_export_dpi() -> None:
    class FakePdf:
        def __init__(self) -> None:
            self.kwargs = None

        def savefig(self, fig: object, **kwargs: object) -> None:
            self.kwargs = kwargs

    pdf = FakePdf()

    _save_pdf_page(pdf, object())  # type: ignore[arg-type]

    assert pdf.kwargs == {"dpi": EXPORT_DPI}
