from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import pytest

from openamundsen_da.pipeline.cleanup import (
    _member_run_manifests,
    clean_predecessor_checkpoint,
    clean_project_artifacts,
)


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
    (results_dir / "member_run.json").write_text(
        '{"member": "member_001", "status": "success"}\n', encoding="utf-8"
    )
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
    plot = project_dir / "steps" / "step_00_init" / "plots" / "forcing" / "station.png"
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b"plot")

    result = clean_project_artifacts(project_dir, apply=True)

    assert result.eligible_paths == ()
    assert result.deleted_paths == ()
    assert state.is_file()
    assert plot.is_file()


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
    (point_csv.parent / "member_run.json").write_text(
        '{"member": "member_001", "status": "success"}\n', encoding="utf-8"
    )

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
        '{"member": "member_001", "status": "success"}\n', encoding="utf-8"
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


def test_compact_cleanup_deletes_derived_forcing_plots_after_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    member = (
        project_dir
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
    )
    forcing = member / "meteo" / "station.csv"
    forcing.parent.mkdir(parents=True)
    forcing.write_text("date,temp\n2023-01-01,273\n", encoding="utf-8")
    (member / "results").mkdir(parents=True)
    (member / "results" / "member_run.json").write_text(
        '{"member": "member_001", "status": "success"}\n',
        encoding="utf-8",
    )
    plot = project_dir / "steps" / "step_00_init" / "plots" / "forcing" / "station.png"
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b"derived plot")
    retained = project_dir / "results" / "forcing" / "ensemble_forcing.nc"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(b"validated compact forcing")
    report = project_dir / "results" / "reports" / "project_report.pdf"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"accepted report")
    validations: list[bool] = []

    def validate_while_raw_exists(*_args, **_kwargs):
        validations.append(forcing.is_file())
        return retained

    monkeypatch.setattr(
        "openamundsen_da.pipeline.cleanup.validate_project_ensemble_forcing",
        validate_while_raw_exists,
    )

    result = clean_project_artifacts(project_dir, apply=True)

    assert validations and all(validations)
    assert plot.resolve() in result.deleted_paths
    assert forcing.resolve() in result.deleted_paths
    assert retained.is_file()
    assert report.is_file()
    ledger = json.loads((project_dir / "results" / "retention_manifest.json").read_text())
    plot_batch = next(
        batch for batch in ledger["batches"]
        if batch["artifact_class"] == "derived_forcing_plot"
    )
    assert plot_batch["status"] == "complete"
    assert plot_batch["consumer_inventory_sha256"]
    assert plot_batch["producer_digest"]


def test_compact_cleanup_keeps_forcing_and_plots_until_report_succeeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    member = (
        project_dir
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
    )
    forcing = member / "meteo" / "station.csv"
    forcing.parent.mkdir(parents=True)
    forcing.write_text("date,temp\n2023-01-01,273\n", encoding="utf-8")
    plot = project_dir / "steps" / "step_00_init" / "plots" / "forcing" / "station.png"
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b"derived plot")
    retained = project_dir / "results" / "forcing" / "ensemble_forcing.nc"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(b"validated compact forcing")
    monkeypatch.setattr(
        "openamundsen_da.pipeline.cleanup.validate_project_ensemble_forcing",
        lambda *_args, **_kwargs: retained,
    )

    result = clean_project_artifacts(project_dir, apply=True)

    assert plot.resolve() not in result.eligible_paths
    assert forcing.resolve() not in result.eligible_paths
    assert plot.is_file()
    assert forcing.is_file()


def test_compact_cleanup_keeps_forcing_plot_when_compact_validation_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    plot = project_dir / "steps" / "step_00_init" / "plots" / "forcing" / "station.png"
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b"derived plot")
    retained = project_dir / "results" / "forcing" / "ensemble_forcing.nc"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(b"invalid compact forcing")
    report = project_dir / "results" / "reports" / "project_report.pdf"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"accepted report")
    monkeypatch.setattr(
        "openamundsen_da.pipeline.cleanup.validate_project_ensemble_forcing",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad compact forcing")),
    )

    with pytest.raises(ValueError, match="bad compact forcing"):
        clean_project_artifacts(project_dir, apply=True)
    assert plot.is_file()


def test_interrupted_forcing_plot_cleanup_never_deletes_raw_forcing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    member = (
        project_dir
        / "steps"
        / "step_00_init"
        / "ensembles"
        / "prior"
        / "member_001"
    )
    forcing = member / "meteo" / "station.csv"
    forcing.parent.mkdir(parents=True)
    forcing.write_text("date,temp\n2023-01-01,273\n", encoding="utf-8")
    (member / "results").mkdir(parents=True)
    (member / "results" / "member_run.json").write_text(
        '{"member": "member_001", "status": "success"}\n',
        encoding="utf-8",
    )
    plot = project_dir / "steps" / "step_00_init" / "plots" / "forcing" / "station.png"
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b"derived plot")
    retained = project_dir / "results" / "forcing" / "ensemble_forcing.nc"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(b"validated compact forcing")
    report = project_dir / "results" / "reports" / "project_report.pdf"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"accepted report")
    monkeypatch.setattr(
        "openamundsen_da.pipeline.cleanup.validate_project_ensemble_forcing",
        lambda *_args, **_kwargs: retained,
    )
    monkeypatch.setattr(
        "openamundsen_da.util.retention._unlink_path",
        lambda path: (_ for _ in ()).throw(OSError("interrupted"))
        if path == plot.resolve()
        else path.unlink(),
    )

    result = clean_project_artifacts(project_dir, apply=True)

    assert result.failures
    assert plot.is_file()
    assert forcing.is_file()


def test_compact_predecessor_cleanup_waits_for_explicit_successor_gate(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    step = project_dir / "steps" / "step_00_init"
    state = step / "ensembles" / "prior" / "member_001" / "results" / "model_state.pickle.gz"
    _write_state(state)
    (state.parent / "member_run.json").write_text(
        '{"member": "member_001", "status": "success"}\n', encoding="utf-8"
    )
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
    (state.parent / "member_run.json").write_text(
        '{"member": "member_001", "status": "success"}\n', encoding="utf-8"
    )
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
    (state.parent / "member_run.json").write_text(
        '{"member": "member_001", "status": "success"}\n', encoding="utf-8"
    )
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


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "unreadable"),
        ("{not-json", "unreadable"),
        ('{"member": "member_001", "status": "failed"}\n', "not successful"),
        ('{"member": "member_002", "status": "success"}\n', "identity differs"),
    ],
)
def test_cleanup_rejects_invalid_or_mismatched_member_producer_manifest(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    project = tmp_path / "project"
    artifact = (
        project
        / "steps"
        / "step_00"
        / "ensembles"
        / "prior"
        / "member_001"
        / "results"
        / "point_a.csv"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("date,swe\n2023-01-01,1\n", encoding="utf-8")
    (artifact.parent / "member_run.json").write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        _member_run_manifests(project, (artifact,))
    assert artifact.is_file()
