from __future__ import annotations

from pathlib import Path

from openamundsen_da.pipeline.cleanup import clean_project_artifacts


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
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_public_cleanup_previews_then_deletes_single_domain_restart_artifacts(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)
    results_dir = project_dir / "steps" / "step_00_init" / "ensembles" / "prior" / "member_001" / "results"
    results_dir.mkdir(parents=True)
    state = results_dir / "model_state.pickle.gz"
    pointer = results_dir.parent / "state_pointer.json"
    grid = results_dir / "output_grids.nc"
    state.write_bytes(b"state")
    pointer.write_text('{"path": "results/model_state.pickle.gz"}\n', encoding="utf-8")
    grid.write_bytes(b"grid")

    preview = clean_project_artifacts(project_dir, apply=False)

    assert preview.applied is False
    assert preview.eligible_paths == (state.resolve(), pointer.resolve())
    assert state.is_file()
    assert pointer.is_file()

    applied = clean_project_artifacts(project_dir, apply=True)

    assert applied.applied is True
    assert applied.deleted_paths == (state.resolve(), pointer.resolve())
    assert not state.exists()
    assert not pointer.exists()
    assert grid.is_file()


def test_public_cleanup_does_not_descend_into_subdomain_tree(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)
    nested = (
        project_dir
        / "subdomains"
        / "S1"
        / "projects"
        / project_dir.name
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
        / "results"
        / "model_state.pickle.gz"
    )
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"state")

    result = clean_project_artifacts(project_dir, apply=True)

    assert result.eligible_paths == ()
    assert nested.is_file()
