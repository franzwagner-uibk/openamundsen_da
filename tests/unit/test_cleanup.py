from __future__ import annotations

from pathlib import Path

from openamundsen_da.pipeline.cleanup import cleanup_setup_dir


def _write_project_yaml(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_2022_2023.yml").write_text(
        "\n".join(
            [
                "run_mode: subdomain",
                "start_date: '2022-10-01'",
                "end_date: '2022-10-02'",
                "data_assimilation:",
                "  restart:",
                "    state_pattern: model_state.pickle.gz",
                "    cleanup_after_setup: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_cleanup_deletes_subdomain_state_files_but_keeps_outputs(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)

    root_results = project_dir / "steps" / "step_00_init" / "ensembles" / "prior" / "member_001" / "results"
    sub_results = (
        project_dir
        / "subdomains"
        / "sd_01"
        / "projects"
        / "project_2022_2023"
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
        / "results"
    )
    for results_dir in (root_results, sub_results):
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "model_state.pickle.gz").write_bytes(b"state")
        (results_dir / "state_pointer.json").write_text('{"path": "model_state.pickle.gz"}\n', encoding="utf-8")
        (results_dir / "output_grids.nc").write_bytes(b"grid")
        (results_dir / "snowdepth_daily_2022-10-01T0000.tif").write_bytes(b"tif")

    subdomain_log = project_dir / "subdomain_run.log"
    subdomain_log.write_text("subdomain log\n", encoding="utf-8")

    summary = cleanup_setup_dir(setup_dir=project_dir)

    assert summary.files_deleted == 2
    assert summary.failures == 0
    assert not (root_results / "model_state.pickle.gz").exists()
    assert not (sub_results / "model_state.pickle.gz").exists()
    assert (project_dir / "subdomains").is_dir()
    assert subdomain_log.is_file()
    for results_dir in (root_results, sub_results):
        assert (results_dir / "state_pointer.json").is_file()
        assert (results_dir / "output_grids.nc").is_file()
        assert (results_dir / "snowdepth_daily_2022-10-01T0000.tif").is_file()
