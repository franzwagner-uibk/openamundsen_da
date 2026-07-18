from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from openamundsen_da import clean_project, prepare_project, render_project, run_project
from openamundsen_da.exceptions import (
    ObservationPreprocessingError,
    ProjectCleanupError,
    ProjectPreparationError,
    ProjectRunError,
)
from openamundsen_da.manifests import load_manifest, workflow_manifest_path, write_manifest_atomic
from openamundsen_da.observations import preprocess_snow_cover
from openamundsen_da.results import ObservationProduct, RenderResult, WorkflowStatus


def _write_public_project(tmp_path: Path) -> Path:
    setup_dir = tmp_path / "alpine"
    project_dir = setup_dir / "projects" / "winter"
    project_dir.mkdir(parents=True)
    (setup_dir / "alpine.yml").write_text(
        """
input_data:
  grids:
    dir: grids
  meteo:
    dir: meteo
output_data:
  grids:
    format: netcdf
""".lstrip(),
        encoding="utf-8",
    )
    (project_dir / "winter.yml").write_text(
        """
run_mode: single
start_date: '2022-10-01'
end_date: '2023-06-30'
obs:
  stations:
    dir: obs/stations
  snowcover:
    dir: obs/snowcover
    format: geotiff
    product_tag: SNOWCOVER
    summary_csv: obs/summaries/winter/scf_summary.csv
    classes:
      valid: [0, 1]
      cloud: [2]
      water: [3]
      nodata: [255]
data_assimilation:
  uncertainty:
    scf:
      enabled: true
      input_dir: obs/snowcover
      ingest: {scf_variable: fsc}
      assimilation: {sigma_mode: formula}
  output:
    grids:
      format: netcdf
  assimilation_events:
    - date: '2023-04-26'
      variable: scf
      product: SNOWCOVER
""".lstrip(),
        encoding="utf-8",
    )
    (setup_dir / "obs" / "stations").mkdir(parents=True)
    (setup_dir / "obs" / "snowcover").mkdir(parents=True)
    summary = setup_dir / "obs" / "summaries" / "winter" / "scf_summary.csv"
    summary.parent.mkdir(parents=True)
    summary.write_text("date,scf,source\n2023-04-26,0.5,snow.tif\n", encoding="utf-8")
    return project_dir


def test_all_six_public_operations_are_available_from_top_level() -> None:
    import openamundsen_da

    assert callable(openamundsen_da.preprocess_snow_cover)
    assert callable(openamundsen_da.preprocess_wet_snow)
    assert callable(openamundsen_da.prepare_project)
    assert callable(openamundsen_da.run_project)
    assert callable(openamundsen_da.render_project)
    assert callable(openamundsen_da.clean_project)


def test_prepare_project_returns_frozen_result_and_reuses_matching_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _write_public_project(tmp_path)

    def fake_skeleton(_setup_dir: Path, target: Path, *, overwrite: bool) -> None:
        step = target / "steps" / "step_00_init"
        step.mkdir(parents=True, exist_ok=True)
        (step / "00.yml").write_text(
            "start_date: '2022-10-01'\nend_date: '2023-04-26'\n",
            encoding="utf-8",
        )

    def fake_observations(target: Path, _summary: Path, *, product: str, overwrite: bool) -> None:
        obs = target / "steps" / "step_00_init" / "obs" / "obs_scf_SNOWCOVER_20230426.csv"
        obs.parent.mkdir(parents=True, exist_ok=True)
        obs.write_text("date,scf\n2023-04-26,0.5\n", encoding="utf-8")

    monkeypatch.setattr("openamundsen_da.api.create_project_skeleton", fake_skeleton)
    monkeypatch.setattr("openamundsen_da.api.prepare_scf_observations", fake_observations)

    completed = prepare_project(project_dir)
    reused = prepare_project(project_dir)

    assert completed.status is WorkflowStatus.COMPLETED
    assert reused.status is WorkflowStatus.REUSED
    assert completed.observation_paths == reused.observation_paths
    with pytest.raises(FrozenInstanceError):
        completed.status = WorkflowStatus.REUSED  # type: ignore[misc]


def test_prepare_project_refuses_mismatched_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _write_public_project(tmp_path)

    def fake_skeleton(_setup_dir: Path, target: Path, *, overwrite: bool) -> None:
        step = target / "steps" / "step_00_init"
        step.mkdir(parents=True, exist_ok=True)
        (step / "00.yml").write_text("start_date: '2022-10-01'\n", encoding="utf-8")

    monkeypatch.setattr("openamundsen_da.api.create_project_skeleton", fake_skeleton)
    monkeypatch.setattr("openamundsen_da.api.prepare_scf_observations", lambda *args, **kwargs: None)
    prepare_project(project_dir)
    summary = project_dir.parent.parent / "obs" / "summaries" / "winter" / "scf_summary.csv"
    summary.write_text("date,scf,source\n2023-04-26,0.9,changed.tif\n", encoding="utf-8")

    with pytest.raises(ProjectPreparationError, match="inputs differ"):
        prepare_project(project_dir)


def test_prepare_project_overwrite_removes_stale_step_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _write_public_project(tmp_path)
    stale = project_dir / "steps" / "step_99_stale" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="utf-8")

    def fake_skeleton(_setup_dir: Path, target: Path, *, overwrite: bool) -> None:
        assert overwrite is True
        step = target / "steps" / "step_00_init"
        step.mkdir(parents=True)
        (step / "00.yml").write_text("start_date: '2022-10-01'\n", encoding="utf-8")

    monkeypatch.setattr("openamundsen_da.api.create_project_skeleton", fake_skeleton)
    monkeypatch.setattr("openamundsen_da.api.prepare_scf_observations", lambda *args, **kwargs: None)

    prepare_project(project_dir, overwrite=True)

    assert not stale.exists()
    assert (project_dir / "steps" / "step_00_init" / "00.yml").is_file()


def test_prepare_project_does_not_overwrite_completed_run(tmp_path: Path) -> None:
    project_dir = _write_public_project(tmp_path)
    run_manifest = project_dir / "results" / "run_manifest.json"
    write_manifest_atomic(run_manifest, {"operation": "run-project", "status": "success"})

    with pytest.raises(ProjectPreparationError, match="immutable"):
        prepare_project(project_dir, overwrite=True)


def test_single_domain_operations_reject_subdomain_projects(tmp_path: Path) -> None:
    project_dir = _write_public_project(tmp_path)
    project_yaml = project_dir / "winter.yml"
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace("run_mode: single", "run_mode: subdomain"),
        encoding="utf-8",
    )

    with pytest.raises(ProjectPreparationError, match="subdomains command tree"):
        prepare_project(project_dir)
    with pytest.raises(ProjectRunError, match="subdomains command tree"):
        run_project(project_dir)
    with pytest.raises(ProjectCleanupError, match="after merge"):
        clean_project(project_dir)
    with pytest.raises(ObservationPreprocessingError, match="subdomains command tree"):
        preprocess_snow_cover(project_dir)


def test_preprocess_snow_cover_returns_typed_result_and_reuses_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _write_public_project(tmp_path)
    setup_dir = project_dir.parent.parent
    raster = setup_dir / "obs" / "snowcover" / "snow.tif"
    raster.write_bytes(b"raster")
    roi = setup_dir / "env" / "roi.gpkg"
    roi.parent.mkdir()
    roi.write_bytes(b"roi")
    summary = setup_dir / "obs" / "summaries" / "winter" / "scf_summary.csv"
    summary.unlink()

    monkeypatch.setattr("openamundsen_da.observations.ensure_setup_roi_vector", lambda _setup: roi)

    def fake_summarize(**kwargs) -> list[Path]:
        target = kwargs["output_root"] / kwargs["project_label"] / "scf_summary.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("date,scf,source\n2023-04-26,0.5,snow.tif\n", encoding="utf-8")
        return [raster]

    monkeypatch.setattr("openamundsen_da.observations.summarize_snowcover_directory", fake_summarize)

    completed = preprocess_snow_cover(project_dir)
    reused = preprocess_snow_cover(project_dir)

    assert completed.product is ObservationProduct.SNOW_COVER
    assert completed.status is WorkflowStatus.COMPLETED
    assert completed.processed_count == 1
    assert reused.status is WorkflowStatus.REUSED


def test_render_project_returns_deterministically_ordered_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _write_public_project(tmp_path)
    results = project_dir / "results"

    def fake_plots(**_kwargs) -> list[str]:
        for name in ("z.png", "a.png"):
            path = results / "plots" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"plot")
        return ["plots"]

    def fake_maps(**_kwargs) -> list[Path]:
        path = results / "maps" / "map.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"map")
        return [path]

    def fake_report(**_kwargs) -> Path:
        path = results / "reports" / "project_report.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pdf")
        return path

    monkeypatch.setattr("openamundsen_da.pipeline.rendering.render_project_plots", fake_plots)
    monkeypatch.setattr("openamundsen_da.pipeline.rendering.project_maps_enabled", lambda _project: True)
    monkeypatch.setattr("openamundsen_da.pipeline.rendering.render_project_maps", fake_maps)
    monkeypatch.setattr("openamundsen_da.pipeline.rendering.build_project_collection_pdf", fake_report)

    result = render_project(project_dir, max_workers=2)

    assert [path.name for path in result.plot_paths] == ["a.png", "z.png"]
    assert [path.name for path in result.map_paths] == ["map.png"]
    assert [path.name for path in result.report_paths] == ["project_report.pdf"]


def _mark_prepared(project_dir: Path) -> None:
    step = project_dir / "steps" / "step_00_init"
    step.mkdir(parents=True, exist_ok=True)
    (step / "00.yml").write_text(
        "start_date: '2022-10-01'\nend_date: '2023-06-30'\n",
        encoding="utf-8",
    )
    write_manifest_atomic(
        workflow_manifest_path(project_dir, "preparation"),
        {"operation": "prepare-project", "status": "success"},
    )


def _fake_successful_execution(project_dir: Path) -> RenderResult:
    results = project_dir / "results"
    compact = results / "grids" / "da_output_grids.nc"
    benchmark = results / "benchmark" / "manifest.json"
    plot = results / "plots" / "overview.png"
    report = results / "reports" / "project_report.pdf"
    perf_csv = results / "plots" / "perf" / "project_perf_metrics.csv"
    perf_plot = results / "plots" / "perf" / "project_perf.png"
    for path, content in (
        (compact, b"netcdf"),
        (benchmark, b'{"status": "success"}\n'),
        (plot, b"plot"),
        (report, b"pdf"),
        (perf_csv, b"time,cpu\n"),
        (perf_plot, b"perf"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    member_results = (
        project_dir
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
        / "results"
    )
    member_results.mkdir(parents=True, exist_ok=True)
    (member_results / "member_run.json").write_text('{"status": "success"}\n', encoding="utf-8")
    (member_results / "model_state.pickle.gz").write_bytes(b"state")
    (member_results.parent / "state_pointer.json").write_text(
        '{"path": "results/model_state.pickle.gz"}\n',
        encoding="utf-8",
    )
    return RenderResult(
        project_dir=project_dir.resolve(),
        status=WorkflowStatus.COMPLETED,
        plot_paths=(plot.resolve(),),
        map_paths=(),
        report_paths=(report.resolve(),),
    )


def test_run_project_finalizes_manifest_then_cleans_restart_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _write_public_project(tmp_path)
    _mark_prepared(project_dir)
    calls = 0
    lifecycle: list[str] = []

    def fake_execute(_config) -> RenderResult:
        nonlocal calls
        calls += 1
        return _fake_successful_execution(project_dir)

    monkeypatch.setattr("openamundsen_da.pipeline.project.run_project", fake_execute)

    state = (
        project_dir
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
        / "results"
        / "model_state.pickle.gz"
    )

    def fake_snapshot(_config) -> bool:
        assert not state.exists()
        lifecycle.append("post-cleanup-snapshot")
        return True

    def fake_report(*, project_dir: Path, output: Path) -> Path:
        lifecycle.append("report-refresh")
        output.write_bytes(b"refreshed-pdf")
        return output

    monkeypatch.setattr("openamundsen_da.api.capture_perf_snapshot", fake_snapshot)
    monkeypatch.setattr("openamundsen_da.api.build_project_collection_pdf", fake_report)

    completed = run_project(project_dir, max_workers=2)
    reused = run_project(project_dir, max_workers=2)

    manifest = load_manifest(completed.manifest_path)
    assert calls == 1
    assert completed.status is WorkflowStatus.COMPLETED
    assert reused.status is WorkflowStatus.REUSED
    assert not state.exists()
    assert manifest is not None
    assert manifest["status"] == "success"
    assert manifest["stages"] == {"execution": "success", "render": "success", "cleanup": "success"}
    assert manifest["cleanup"]["deleted_count"] == 2
    assert lifecycle == ["post-cleanup-snapshot", "report-refresh"]
    assert (project_dir / "results" / "reports" / "project_report.pdf").read_bytes() == b"refreshed-pdf"


def test_run_project_retains_restart_states_when_required_output_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _write_public_project(tmp_path)
    _mark_prepared(project_dir)

    def fake_execute(_config) -> RenderResult:
        result = _fake_successful_execution(project_dir)
        result.report_paths[0].unlink()
        return result

    monkeypatch.setattr("openamundsen_da.pipeline.project.run_project", fake_execute)
    snapshot_calls: list[Path] = []
    monkeypatch.setattr(
        "openamundsen_da.api.capture_perf_snapshot",
        lambda config: snapshot_calls.append(config.project_dir) or True,
    )

    with pytest.raises(Exception, match="report validation"):
        run_project(project_dir)

    state = (
        project_dir
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
        / "results"
        / "model_state.pickle.gz"
    )
    manifest = load_manifest(project_dir / "results" / "run_manifest.json")
    assert state.is_file()
    assert manifest is not None
    assert manifest["status"] == "failed"
    assert manifest["stages"]["execution"] == "failed"
    assert manifest["stages"]["cleanup"] == "pending"
    assert snapshot_calls == []


def test_run_project_rejects_mismatched_resume_before_manifest_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _write_public_project(tmp_path)
    _mark_prepared(project_dir)

    def fake_failure(_config):
        raise RuntimeError("interrupted")

    monkeypatch.setattr("openamundsen_da.pipeline.project.run_project", fake_failure)
    with pytest.raises(Exception, match="interrupted"):
        run_project(project_dir)
    manifest_path = project_dir / "results" / "run_manifest.json"
    summary = project_dir.parent.parent / "obs" / "summaries" / "winter" / "scf_summary.csv"
    summary.write_text("date,scf,source\n2023-04-26,0.8,changed.tif\n", encoding="utf-8")
    before = manifest_path.read_bytes()

    with pytest.raises(Exception, match="inputs differ"):
        run_project(project_dir)

    assert manifest_path.read_bytes() == before


def test_run_project_records_interruption_and_retains_restart_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _write_public_project(tmp_path)
    _mark_prepared(project_dir)

    def fake_interrupt(_config) -> None:
        state = (
            project_dir
            / "steps"
            / "step_00_init"
            / "ensembles"
            / "prior"
            / "member_001"
            / "results"
            / "model_state.pickle.gz"
        )
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(b"restart")
        raise KeyboardInterrupt

    monkeypatch.setattr("openamundsen_da.pipeline.project.run_project", fake_interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_project(project_dir)

    manifest = load_manifest(project_dir / "results" / "run_manifest.json")
    assert manifest is not None
    assert manifest["status"] == "interrupted"
    assert manifest["stages"]["execution"] == "interrupted"
    assert manifest["stages"]["cleanup"] == "pending"
    assert (
        project_dir
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
        / "results"
        / "model_state.pickle.gz"
    ).is_file()


def test_run_project_resumes_failed_matching_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = _write_public_project(tmp_path)
    _mark_prepared(project_dir)

    monkeypatch.setattr(
        "openamundsen_da.pipeline.project.run_project",
        lambda _config: (_ for _ in ()).throw(RuntimeError("first attempt failed")),
    )
    with pytest.raises(ProjectRunError, match="first attempt failed"):
        run_project(project_dir)

    monkeypatch.setattr(
        "openamundsen_da.pipeline.project.run_project",
        lambda _config: _fake_successful_execution(project_dir),
    )
    resumed = run_project(project_dir)
    manifest = load_manifest(resumed.manifest_path)

    assert resumed.status is WorkflowStatus.COMPLETED
    assert manifest is not None
    assert manifest["status"] == "success"
    assert manifest["resumed_from_status"] == "failed"
