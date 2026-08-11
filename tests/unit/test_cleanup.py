from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import pytest

from openamundsen_da.pipeline.cleanup import clean_predecessor_checkpoint, clean_project_artifacts


def _write_state(path: Path, value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as stream:
        pickle.dump({"snow": {"swe": value}}, stream)


def _write_project_yaml(project_dir: Path, *, retention: str | None = None) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_2022_2023.yml").write_text(
        "\n".join(
            [
                "run_mode: subdomain",
                "start_date: '2022-10-01'",
                "end_date: '2022-10-02'",
                "data_assimilation:",
                "  prior_forcing:",
                "    ensemble_size: 2",
                "  restart:",
                "    state_pattern: model_state.pickle.gz",
                *( ["  output:", f"    retention: {retention}"] if retention else [] ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_public_cleanup_previews_then_deletes_single_domain_restart_artifacts(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    results_dir = project_dir / "steps" / "step_00_init" / "ensembles" / "prior" / "member_001" / "results"
    results_dir.mkdir(parents=True)
    state = results_dir / "model_state.pickle.gz"
    pointer = results_dir.parent / "state_pointer.json"
    grid = results_dir / "output_grids.nc"
    state.write_bytes(b"state")
    (results_dir / "member_run.json").write_text('{"status": "success"}\n', encoding="utf-8")
    compact = project_dir / "results" / "points" / "ensemble_points.nc"
    compact.parent.mkdir(parents=True, exist_ok=True)
    compact.write_bytes(b"accepted")
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


def test_full_retention_preserves_restart_and_member_artifacts(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="full")
    results_dir = (
        project_dir
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
        / "results"
    )
    results_dir.mkdir(parents=True)
    state = results_dir / "model_state.pickle.gz"
    state.write_bytes(b"state")

    result = clean_project_artifacts(project_dir, apply=True)

    assert result.eligible_paths == ()
    assert result.deleted_paths == ()
    assert state.is_file()


def test_compact_cleanup_removes_point_csv_only_after_lossless_store_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    point_csv = (
        project_dir
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
        / "results"
        / "point_station.csv"
    )
    point_csv.parent.mkdir(parents=True)
    point_csv.write_text("date,snow_depth\n2023-01-01,1.0\n", encoding="utf-8")
    (point_csv.parent / "member_run.json").write_text('{"status": "success"}\n', encoding="utf-8")

    assert point_csv not in clean_project_artifacts(project_dir, apply=False).eligible_paths
    retained = project_dir / "results" / "points" / "ensemble_points.nc"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(b"validated-store")
    monkeypatch.setattr(
        "openamundsen_da.pipeline.cleanup.validate_project_ensemble_points",
        lambda *_args, **_kwargs: retained,
    )

    result = clean_project_artifacts(project_dir, apply=True)
    assert point_csv.resolve() in result.deleted_paths
    assert retained.is_file()


def test_compact_cleanup_preserves_station_metadata_when_forcing_is_compacted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    meteo = (
        project_dir
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
        / "meteo"
    )
    meteo.mkdir(parents=True)
    station = meteo / "station.csv"
    metadata = meteo / "stations.csv"
    station.write_text("date,temp\n2023-01-01,273\n", encoding="utf-8")
    (station.parents[1] / "results").mkdir(parents=True, exist_ok=True)
    (station.parents[1] / "results" / "member_run.json").write_text(
        '{"status": "success"}\n', encoding="utf-8"
    )
    metadata.write_text("id,name,x,y,alt\nstation,S,0,0,0\n", encoding="utf-8")
    retained = project_dir / "results" / "forcing" / "ensemble_forcing.nc"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(b"validated-store")
    monkeypatch.setattr(
        "openamundsen_da.pipeline.cleanup.validate_project_ensemble_forcing",
        lambda *_args, **_kwargs: retained,
    )

    result = clean_project_artifacts(project_dir, apply=True)

    assert station.resolve() in result.deleted_paths
    assert not station.exists()
    assert metadata.is_file()


def test_compact_predecessor_cleanup_waits_for_explicit_successor_gate(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    step = project_dir / "steps" / "step_00_init"
    state = step / "ensembles" / "prior" / "member_001" / "results" / "model_state.pickle.gz"
    _write_state(state)
    (state.parent / "member_run.json").write_text('{"status": "success"}\n', encoding="utf-8")
    successor = project_dir / "steps" / "step_01"
    for name in ("open_loop", "member_001", "member_002"):
        _write_state(
            successor / "ensembles" / "prior" / name / "results" / "model_state.pickle.gz"
        )

    preview = clean_predecessor_checkpoint(project_dir, step, apply=False)
    assert preview == (state.resolve(),)
    assert state.is_file()
    removed = clean_predecessor_checkpoint(
        project_dir,
        step,
        successor_step=successor,
        apply=True,
    )
    assert removed == preview
    assert not state.exists()


def test_predecessor_cleanup_requires_every_successor_state(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    predecessor = project_dir / "steps" / "step_00_init"
    state = predecessor / "ensembles" / "prior" / "member_001" / "results" / "model_state.pickle.gz"
    _write_state(state)
    (state.parent / "member_run.json").write_text('{"status": "success"}\n', encoding="utf-8")
    successor = project_dir / "steps" / "step_01"
    _write_state(successor / "ensembles" / "prior" / "open_loop" / "results" / "model_state.pickle.gz")
    broken = successor / "ensembles" / "prior" / "member_001" / "results" / "model_state.pickle.gz"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a checkpoint")
    _write_state(successor / "ensembles" / "prior" / "member_002" / "results" / "model_state.pickle.gz")

    with pytest.raises(RuntimeError, match="Restart state is unreadable"):
        clean_predecessor_checkpoint(
            project_dir,
            predecessor,
            successor_step=successor,
            apply=True,
        )

    assert state.is_file()


def test_predecessor_cleanup_rejects_missing_or_extra_successor_members(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    predecessor = project_dir / "steps" / "step_00_init"
    state = predecessor / "ensembles" / "prior" / "member_001" / "results" / "model_state.pickle.gz"
    _write_state(state)
    (state.parent / "member_run.json").write_text('{"status": "success"}\n', encoding="utf-8")
    successor = project_dir / "steps" / "step_01"
    for name in ("open_loop", "member_001"):
        _write_state(successor / "ensembles" / "prior" / name / "results" / "model_state.pickle.gz")

    with pytest.raises(RuntimeError, match="membership differs"):
        clean_predecessor_checkpoint(
            project_dir, predecessor, successor_step=successor, apply=True
        )
    _write_state(successor / "ensembles" / "prior" / "member_002" / "results" / "model_state.pickle.gz")
    _write_state(successor / "ensembles" / "prior" / "member_003" / "results" / "model_state.pickle.gz")
    with pytest.raises(RuntimeError, match="membership differs"):
        clean_predecessor_checkpoint(
            project_dir, predecessor, successor_step=successor, apply=True
        )
    assert state.is_file()
