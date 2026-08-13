from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from openamundsen_da.exceptions import ProjectValidationError
from openamundsen_da.pipeline import plot_tasks, project as project_cli


def _write_project_yaml(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{project_dir.name}.yml").write_text(
        "run_mode: single\nstart_date: '2022-10-01'\nend_date: '2022-10-02'\n"
        "data_assimilation: {}\n",
        encoding="utf-8",
    )


def test_cli_enables_perf_monitor_by_default(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    setup_dir.mkdir(parents=True, exist_ok=True)
    _write_project_yaml(project_dir)

    called: dict = {}

    def _fake_run_project(cfg):
        called["cfg"] = cfg

    monkeypatch.setattr("openamundsen_da.pipeline.project.run_project", _fake_run_project)

    rc = project_cli.cli(
        [
            "--setup-dir",
            str(setup_dir),
            "--project-dir",
            str(project_dir),
        ]
    )

    assert rc == 0
    assert called["cfg"].monitor_perf is True


def test_cli_disables_perf_monitor_with_flag(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    setup_dir.mkdir(parents=True, exist_ok=True)
    _write_project_yaml(project_dir)

    called: dict = {}

    def _fake_run_project(cfg):
        called["cfg"] = cfg

    monkeypatch.setattr("openamundsen_da.pipeline.project.run_project", _fake_run_project)

    rc = project_cli.cli(
        [
            "--setup-dir",
            str(setup_dir),
            "--project-dir",
            str(project_dir),
            "--no-monitor-perf",
        ]
    )

    assert rc == 0
    assert called["cfg"].monitor_perf is False


def test_cli_missing_run_mode_fails_without_rewriting_project_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects/winter"
    _write_project_yaml(project_dir)
    project_yaml = project_dir / "winter.yml"
    original = project_yaml.read_text(encoding="utf-8").replace("run_mode: single\n", "")
    project_yaml.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        project_cli,
        "run_project",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("run must not start")),
    )

    with pytest.raises(ValueError, match="no 'run_mode' marker"):
        project_cli.cli(
            ["--setup-dir", str(setup_dir), "--project-dir", str(project_dir)]
        )

    assert project_yaml.read_text(encoding="utf-8") == original


def test_post_run_plot_tasks_include_setup_weights_overview(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    step_dir = project_dir / "steps" / "step_01_event"
    assim_dir = step_dir / "assim"
    assim_dir.mkdir(parents=True, exist_ok=True)
    _write_project_yaml(project_dir)
    (setup_dir / "demo.yml").write_text("name: setup\n", encoding="utf-8")
    (assim_dir / "weights_station_hs_20230221.csv").write_text(
        "member_id,residual,sigma,log_weight,weight\nmember_001,0.1,0.2,-1.0,1.0\n",
        encoding="utf-8",
    )
    (assim_dir / "station_diagnostics_station_hs_20230221.csv").write_text(
        "station_id,member_id,residual,sigma\nstation_a,member_001,0.1,0.2\n",
        encoding="utf-8",
    )

    cfg = project_cli.OrchestratorConfig(
        project_dir=project_dir,
        setup_dir=setup_dir,
    )

    tasks = plot_tasks.build_post_run_plot_tasks(cfg, [step_dir])

    names = [task.name for task in tasks]
    assert "setup_weights_overview" in names

    overview_task = next(task for task in tasks if task.name == "setup_weights_overview")
    assert overview_task.func is plot_tasks.plot_setup_weights_overview
    assert overview_task.args == (project_dir,)
    assert overview_task.kwargs == {}


def test_configured_overview_needs_benchmark_scores_detects_score_panels(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)
    (project_dir / "plots.yml").write_text(
        "panels:\n  - panel: fSC\n  - panel: scores-crpss\n",
        encoding="utf-8",
    )

    assert plot_tasks.configured_overview_needs_benchmark_scores(project_dir) is True


def test_post_run_plot_tasks_can_defer_fraction_overlay_for_score_panels(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    step_dir = project_dir / "steps" / "step_01_event"
    step_dir.mkdir(parents=True, exist_ok=True)
    _write_project_yaml(project_dir)
    (project_dir / "plots.yml").write_text(
        "panels:\n  - panel: scores-crpss\n",
        encoding="utf-8",
    )

    cfg = project_cli.OrchestratorConfig(
        project_dir=project_dir,
        setup_dir=setup_dir,
    )

    needs_scores = plot_tasks.configured_overview_needs_benchmark_scores(project_dir)
    tasks = plot_tasks.build_post_run_plot_tasks(
        cfg,
        [step_dir],
        include_fraction_overlay=not needs_scores,
    )

    names = [task.name for task in tasks]
    assert needs_scores is True
    assert "fraction_overlay" not in names

    deferred = plot_tasks.build_fraction_overlay_task(cfg)
    assert deferred.name == "fraction_overlay"
    assert deferred.func is plot_tasks.plot_result_overview_cli
    assert deferred.args == (
        [
            "--project-dir",
            str(project_dir),
            "--setup-dir",
            str(setup_dir),
        ],
    )
    assert deferred.kwargs == {"configure_logger": False}


def test_project_pipeline_runs_report_after_final_artifact_stages() -> None:
    source = inspect.getsource(project_cli._run_project_impl)

    assert source.index("write_project_da_output_grids(") < source.index("run_project_benchmark(")
    assert source.index("run_project_benchmark(") < source.index("render_required_project_outputs(")
    assert source.index("render_required_project_outputs(") < source.index("Project processing complete:")


def test_project_pipeline_validates_configuration_before_discovery(monkeypatch, tmp_path: Path) -> None:
    def reject_config(_project_dir: Path):
        raise ProjectValidationError(["invalid scientific configuration"])

    monkeypatch.setattr(project_cli, "load_project_configuration", reject_config)
    monkeypatch.setattr(
        project_cli,
        "_list_steps_sorted",
        lambda _project_dir: (_ for _ in ()).throw(AssertionError("step discovery must not run")),
    )
    cfg = project_cli.OrchestratorConfig(
        project_dir=tmp_path / "setup" / "projects" / "project_demo",
        setup_dir=tmp_path / "setup",
        monitor_perf=False,
    )

    with pytest.raises(ProjectValidationError, match="invalid scientific configuration"):
        project_cli._run_project_impl(cfg, run_start=project_cli.datetime.utcnow())


def test_project_pipeline_stops_monitor_and_captures_final_snapshot_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class FakeHandle:
        def stop_and_join(self) -> None:
            calls.append("stop")

        def capture_now(self) -> None:
            calls.append("capture")

    monkeypatch.setattr(project_cli, "start_perf_monitor", lambda _cfg: FakeHandle())
    monkeypatch.setattr(project_cli, "preadmit_project_storage", lambda cfg: cfg)
    monkeypatch.setattr(
        project_cli,
        "_run_project_impl",
        lambda _cfg, *, run_start: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    cfg = project_cli.OrchestratorConfig(
        project_dir=tmp_path / "project",
        setup_dir=tmp_path,
        monitor_perf=True,
    )

    with pytest.raises(RuntimeError, match="failed"):
        project_cli.run_project(cfg)

    assert calls == ["stop", "capture"]


def test_project_storage_preadmission_handoff_is_not_rechecked_after_runtime_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "setup/projects/project"
    step = project_dir / "steps/step_00"
    step.mkdir(parents=True)
    calls: list[str] = []

    class _Client:
        leaf_id = "project"

        @staticmethod
        def admit_step(step_name: str, **_kwargs):
            calls.append(step_name)
            return SimpleNamespace(
                used_fraction=0.1,
                estimated_growth_bytes=100,
                operational_reserve_bytes=50,
            )

    monkeypatch.setattr(
        project_cli,
        "load_project_configuration",
        lambda _project: SimpleNamespace(project_dir=project_dir),
    )
    monkeypatch.setattr(project_cli, "_list_steps_sorted", lambda _project: [step])
    cfg = project_cli.OrchestratorConfig(
        project_dir=project_dir,
        setup_dir=tmp_path / "setup",
        storage_admission_client=_Client(),
    )

    admitted = project_cli.preadmit_project_storage(cfg)
    replay = project_cli.preadmit_project_storage(admitted)

    assert calls == ["step_00"]
    assert replay.initial_step_preadmitted is True
    assert replay.initial_storage_budget is admitted.initial_storage_budget


def test_project_impl_consumes_preadmitted_budget_before_opening_runtime_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects/project"
    step = project_dir / "steps/step_00"
    step.mkdir(parents=True)
    (setup_dir / "meteo").mkdir()

    class _Client:
        leaf_id = "project"

        @staticmethod
        def admit_step(*_args, **_kwargs):
            raise AssertionError("preadmitted step zero must not be rechecked after runtime starts")

    budget = SimpleNamespace(
        used_fraction=0.1,
        estimated_growth_bytes=100,
        operational_reserve_bytes=50,
    )
    config = SimpleNamespace(
        project_dir=project_dir.resolve(),
        setup_dir=setup_dir.resolve(),
    )
    monkeypatch.setattr(project_cli, "load_project_configuration", lambda _project: config)
    monkeypatch.setattr(project_cli, "_list_steps_sorted", lambda _project: [step])
    monkeypatch.setattr(project_cli, "load_assimilation_events", lambda _project: [])
    monkeypatch.setattr(project_cli, "output_retention_mode", lambda _project: "compact")
    monkeypatch.setattr(project_cli, "validate_assimilation_requirements", lambda **_kwargs: None)
    monkeypatch.setattr(
        project_cli,
        "_setup_logger",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("runtime log opened")),
    )
    cfg = project_cli.OrchestratorConfig(
        project_dir=project_dir,
        setup_dir=setup_dir,
        storage_admission_client=_Client(),
        initial_step_preadmitted=True,
        initial_storage_budget=budget,
    )

    with pytest.raises(RuntimeError, match="runtime log opened"):
        project_cli._run_project_impl(cfg, run_start=project_cli.datetime.utcnow())
