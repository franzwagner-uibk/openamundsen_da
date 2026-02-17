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
