from __future__ import annotations

from pathlib import Path

from openamundsen_da.pipeline import project as project_cli


def _write_project_yaml(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{project_dir.name}.yml").write_text(
        "start_date: '2022-10-01'\nend_date: '2022-10-02'\ndata_assimilation: {}\n",
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


def test_post_run_plot_tasks_include_setup_weights_overview(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    step_dir = project_dir / "steps" / "step_01_event"
    assim_dir = step_dir / "assim"
    assim_dir.mkdir(parents=True, exist_ok=True)
    _write_project_yaml(project_dir)
    (setup_dir / "setup.yml").write_text("name: setup\n", encoding="utf-8")
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

    tasks = project_cli._build_post_run_plot_tasks(cfg, [step_dir])

    names = [task.name for task in tasks]
    assert "setup_weights_overview" in names

    overview_task = next(task for task in tasks if task.name == "setup_weights_overview")
    assert overview_task.func is project_cli.plot_setup_weights_overview
    assert overview_task.args == (project_dir,)
    assert overview_task.kwargs == {}


def test_custom_overview_needs_benchmark_scores_detects_score_panels(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)
    (project_dir / "plots.yml").write_text(
        "panels:\n  - panel: fSC\n  - panel: scores-crpss\n",
        encoding="utf-8",
    )

    assert project_cli._custom_overview_needs_benchmark_scores(project_dir) is True


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

    needs_scores = project_cli._custom_overview_needs_benchmark_scores(project_dir)
    tasks = project_cli._build_post_run_plot_tasks(
        cfg,
        [step_dir],
        include_fraction_overlay=not needs_scores,
    )

    names = [task.name for task in tasks]
    assert needs_scores is True
    assert "fraction_overlay" not in names

    deferred = project_cli._build_fraction_overlay_task(cfg)
    assert deferred.name == "fraction_overlay"
    assert deferred.func is project_cli.plot_result_overview_cli
    assert deferred.args == (
        [
            "--project-dir",
            str(project_dir),
            "--setup-dir",
            str(setup_dir),
        ],
    )
    assert deferred.kwargs == {"configure_logger": False}
