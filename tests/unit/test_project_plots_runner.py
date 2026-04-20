from __future__ import annotations

from pathlib import Path

from openamundsen_da.methods.viz.plots import runner as plots_runner


def _write_project_yaml(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{project_dir.name}.yml").write_text(
        "start_date: '2022-10-01'\nend_date: '2022-10-02'\ndata_assimilation: {}\n",
        encoding="utf-8",
    )


def _write_step_yaml(step_dir: Path) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "01.yml").write_text(
        "start_date: '2022-10-01'\nend_date: '2022-10-02'\n",
        encoding="utf-8",
    )


def test_default_project_plots_rerun_command_uses_runner_module(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"

    command = plots_runner.default_project_plots_rerun_command(project_dir)

    assert command == (
        "python -m openamundsen_da.methods.viz.plots.runner "
        f"--project-dir {project_dir.resolve()}"
    )


def test_render_project_plots_runs_post_processing_tasks(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    step_dir = project_dir / "steps" / "step_01_event"
    _write_step_yaml(step_dir)
    (setup_dir / "setup.yml").write_text("name: setup\n", encoding="utf-8")
    _write_project_yaml(project_dir)

    calls: dict[str, object] = {}

    monkeypatch.setattr(plots_runner, "apply_env_from_project", lambda path: calls.setdefault("project_yaml", path))
    monkeypatch.setattr(plots_runner, "ensure_gdal_proj_from_conda", lambda: calls.setdefault("gdal", True))
    monkeypatch.setattr(plots_runner, "apply_numeric_thread_defaults", lambda: calls.setdefault("threads", True))
    monkeypatch.setattr(plots_runner, "aggregate_fraction_envelopes", lambda **kwargs: calls.setdefault("aggregate", kwargs))
    monkeypatch.setattr(plots_runner, "custom_overview_needs_benchmark_scores", lambda project_dir: False)
    monkeypatch.setattr(
        plots_runner,
        "build_post_run_plot_tasks",
        lambda cfg, steps, include_fraction_overlay: [type("Task", (), {"name": "setup_results_swe"})()],
    )

    def _fake_run(tasks, plot_workers, max_workers):
        calls.setdefault("runs", []).append(([task.name for task in tasks], plot_workers, max_workers))

    monkeypatch.setattr(plots_runner, "run_plot_tasks_parallel", _fake_run)

    outputs = plots_runner.render_project_plots(
        project_dir=project_dir,
        plot_workers=7,
        max_workers=11,
    )

    assert outputs == ["setup_results_swe"]
    assert calls["project_yaml"] == project_dir / "project_2022_2023.yml"
    assert calls["aggregate"]["project_dir"] == project_dir
    assert calls["runs"] == [(["setup_results_swe"], 7, 11)]


def test_render_project_plots_runs_deferred_overlay_when_needed(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    step_dir = project_dir / "steps" / "step_01_event"
    _write_step_yaml(step_dir)
    (setup_dir / "setup.yml").write_text("name: setup\n", encoding="utf-8")
    _write_project_yaml(project_dir)

    monkeypatch.setattr(plots_runner, "apply_env_from_project", lambda path: {})
    monkeypatch.setattr(plots_runner, "ensure_gdal_proj_from_conda", lambda: None)
    monkeypatch.setattr(plots_runner, "apply_numeric_thread_defaults", lambda: None)
    monkeypatch.setattr(plots_runner, "aggregate_fraction_envelopes", lambda **kwargs: None)
    monkeypatch.setattr(plots_runner, "custom_overview_needs_benchmark_scores", lambda project_dir: True)
    monkeypatch.setattr(
        plots_runner,
        "build_post_run_plot_tasks",
        lambda cfg, steps, include_fraction_overlay: [type("Task", (), {"name": "setup_weights_overview"})()],
    )
    monkeypatch.setattr(plots_runner, "build_fraction_overlay_task", lambda cfg: type("Task", (), {"name": "fraction_overlay"})())

    runs: list[tuple[list[str], int | None, int | None]] = []

    def _fake_run(tasks, plot_workers, max_workers):
        runs.append(([task.name for task in tasks], plot_workers, max_workers))

    monkeypatch.setattr(plots_runner, "run_plot_tasks_parallel", _fake_run)

    outputs = plots_runner.render_project_plots(project_dir=project_dir)

    assert outputs == ["setup_weights_overview", "fraction_overlay"]
    assert runs == [
        (["setup_weights_overview"], None, None),
        (["fraction_overlay"], None, None),
    ]
