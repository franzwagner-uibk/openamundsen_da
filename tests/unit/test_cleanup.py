from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import pytest

from openamundsen_da.exceptions import CleanupSafetyError
from openamundsen_da.manifests import write_manifest_atomic
from openamundsen_da.pipeline import cleanup as cleanup_mod
from openamundsen_da.pipeline.cleanup import (
    _member_run_manifests,
    clean_predecessor_checkpoint,
    clean_project_artifacts,
)
from openamundsen_da.pipeline.rendering import render_completion_manifest_path
from openamundsen_da.methods.pf.resample import resample_from_weights
from openamundsen_da.util.retention import validate_retained_consumers


def _write_state(path: Path, value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as stream:
        pickle.dump({"snow": {"swe": value}}, stream)


def _write_render_completion(project_dir: Path) -> Path:
    path = render_completion_manifest_path(project_dir)
    return write_manifest_atomic(
        path,
        {
            "contract": "project-render-v1",
            "status": "success",
            "project_dir": str(project_dir.resolve()),
        },
    )


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


def test_cross_class_power_failure_refuses_overwrite_before_any_new_delete(
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
    point = member / "results" / "point_station.csv"
    forcing = member / "meteo" / "station.csv"
    producer = member / "results" / "member_run.json"
    point.parent.mkdir(parents=True)
    forcing.parent.mkdir(parents=True)
    point.write_text("time,swe\n2023-01-01,1\n", encoding="utf-8")
    forcing.write_text("time,temp\n2023-01-01,273\n", encoding="utf-8")
    producer.write_text(
        '{"member": "member_001", "status": "success"}\n',
        encoding="utf-8",
    )
    compact_point = project_dir / "results" / "points" / "ensemble_points.nc"
    compact_forcing = project_dir / "results" / "forcing" / "ensemble_forcing.nc"
    compact_point.parent.mkdir(parents=True)
    compact_forcing.parent.mkdir(parents=True)
    compact_point.write_bytes(b"point generation one")
    compact_forcing.write_bytes(b"forcing generation one")
    monkeypatch.setattr(
        cleanup_mod,
        "validate_project_ensemble_points",
        lambda *_args, **_kwargs: compact_point,
    )
    monkeypatch.setattr(
        cleanup_mod,
        "validate_project_ensemble_forcing",
        lambda *_args, **_kwargs: compact_forcing,
    )
    real_apply = cleanup_mod.apply_retention_batch

    def stop_between_classes(*args, artifact_class: str, **kwargs):
        if artifact_class == "member_forcing_csv":
            raise RuntimeError("power failure between cleanup classes")
        return real_apply(*args, artifact_class=artifact_class, **kwargs)

    monkeypatch.setattr(cleanup_mod, "apply_retention_batch", stop_between_classes)
    interrupted = clean_project_artifacts(project_dir, apply=True)
    assert interrupted.failures
    assert not point.exists()
    assert forcing.is_file()

    point.write_text("time,swe\n2023-01-01,2\n", encoding="utf-8")
    forcing.write_text("time,temp\n2023-01-01,274\n", encoding="utf-8")
    compact_point.write_bytes(b"point generation two")
    compact_forcing.write_bytes(b"forcing generation two")
    producer.write_text(
        '{"member": "member_001", "status": "success", "run": 2}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cleanup_mod, "apply_retention_batch", real_apply)

    with pytest.raises(CleanupSafetyError, match="changed after planning|identity"):
        clean_project_artifacts(project_dir, apply=True)

    assert point.is_file()
    assert forcing.is_file()


def test_compact_cleanup_deletes_derived_forcing_plots_after_render_completion(
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
    report.write_bytes(b"initial accepted report")
    render_completion = _write_render_completion(project_dir)
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
    assert render_completion.is_file()
    ledger = json.loads((project_dir / "results" / "retention_manifest.json").read_text())
    plot_batch = next(
        batch for batch in ledger["batches"]
        if batch["artifact_class"] == "derived_forcing_plot"
    )
    assert plot_batch["status"] == "complete"
    assert plot_batch["consumer_inventory_sha256"]
    assert plot_batch["producer_digest"]
    assert ledger["active_generation"] == 1
    assert ledger["generations"][0]["status"] == "complete"
    assert {batch["generation"] for batch in ledger["batches"]} == {1}
    report.write_bytes(b"final performance refresh")
    assert validate_retained_consumers(project_dir, require_complete=True)


def test_compact_cleanup_keeps_forcing_and_plots_until_render_succeeds(
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
    _write_render_completion(project_dir)
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
    _write_render_completion(project_dir)
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


def test_predecessor_cleanup_resolves_real_pf_posterior_pointer_producers(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, retention="compact")
    predecessor = project_dir / "steps" / "step_00_init"
    prior = predecessor / "ensembles" / "prior"
    states: dict[str, Path] = {}
    manifests: list[Path] = []
    for index, name in enumerate(("open_loop", "member_001", "member_002"), start=1):
        state = prior / name / "results" / "model_state.pickle.gz"
        _write_state(state, value=index)
        manifest = state.parent / "member_run.json"
        manifest.write_text(
            f'{{"member": "{name}", "status": "success"}}\n',
            encoding="utf-8",
        )
        states[name] = state
        manifests.append(manifest)

    weights = predecessor / "assim" / "weights_station_hs_20221001.csv"
    weights.parent.mkdir(parents=True)
    weights.write_text(
        "member_id,weight\nmember_001,0.0\nmember_002,1.0\n",
        encoding="utf-8",
    )
    resample_from_weights(
        step_dir=predecessor,
        source_ensemble="prior",
        weights_csv=weights,
        target_ensemble="posterior",
        seed=17,
        algorithm="systematic",
        ess_threshold=1.5,
        ess_threshold_ratio=None,
        overwrite=False,
    )
    posterior = predecessor / "ensembles" / "posterior"
    posterior_pointers = sorted(posterior.glob("member_*/state_pointer.json"))
    assert len(posterior_pointers) == 2
    assert all(
        json.loads((pointer.parent / "source_pointer.json").read_text(encoding="utf-8"))[
            "member_dir"
        ].endswith("member_002")
        for pointer in posterior_pointers
    )
    posterior_pointers[0].write_text(
        json.dumps(
            {
                "path": (
                    "/setup/projects/project_2022_2023/steps/step_00_init/"
                    "ensembles/prior/member_002/results/model_state.pickle.gz"
                )
            }
        ),
        encoding="utf-8",
    )

    successor = project_dir / "steps" / "step_01_da"
    successor_pointers: list[Path] = []
    for index, name in enumerate(("open_loop", "member_001", "member_002"), start=10):
        member = successor / "ensembles" / "prior" / name
        state = member / "results" / "model_state.pickle.gz"
        _write_state(state, value=index)
        source_name = "open_loop" if name == "open_loop" else "member_002"
        pointer = member / "state_pointer.json"
        pointer.write_text(
            json.dumps({"path": str(states[source_name])}),
            encoding="utf-8",
        )
        successor_pointers.append(pointer)

    removed = clean_predecessor_checkpoint(
        project_dir,
        predecessor,
        successor_step=successor,
        apply=True,
    )

    assert set(removed) == {
        *(state.resolve() for state in states.values()),
        *(pointer.resolve() for pointer in posterior_pointers),
        *(pointer.resolve() for pointer in successor_pointers),
    }
    assert all(not path.exists() for path in removed)
    assert all(manifest.is_file() for manifest in manifests)
    assert all(
        (successor / "ensembles" / "prior" / name / "results" / "model_state.pickle.gz").is_file()
        for name in ("open_loop", "member_001", "member_002")
    )
    validate_retained_consumers(project_dir, require_complete=True)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "unreadable"),
        ("{not-json", "unreadable"),
        ("{}", "no valid path"),
        ('{"path": "results/missing.pickle.gz"}', "missing or outside"),
    ],
)
def test_cleanup_rejects_invalid_checkpoint_pointer_provenance(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    project = tmp_path / "project"
    _write_project_yaml(project, retention="compact")
    pointer = (
        project
        / "steps"
        / "step_00"
        / "ensembles"
        / "posterior"
        / "member_001"
        / "state_pointer.json"
    )
    pointer.parent.mkdir(parents=True)
    pointer.write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        _member_run_manifests(project, (pointer,))
    assert pointer.is_file()


def test_cleanup_rejects_checkpoint_pointer_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project_yaml(project, retention="compact")
    external = tmp_path / "external" / "model_state.pickle.gz"
    _write_state(external)
    pointer = (
        project
        / "steps"
        / "step_00"
        / "ensembles"
        / "posterior"
        / "member_001"
        / "state_pointer.json"
    )
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"path": str(external)}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing or outside"):
        _member_run_manifests(project, (pointer,))
    assert pointer.is_file()
    assert external.is_file()


def test_predecessor_cleanup_fails_before_deletion_for_malformed_pointer(
    tmp_path: Path,
) -> None:
    project = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project, retention="compact")
    predecessor = project / "steps" / "step_00"
    state = predecessor / "ensembles" / "prior" / "member_001" / "results" / "model_state.pickle.gz"
    _write_state(state)
    (state.parent / "member_run.json").write_text(
        '{"member": "member_001", "status": "success"}\n',
        encoding="utf-8",
    )
    malformed = (
        project
        / "steps"
        / "step_01"
        / "ensembles"
        / "prior"
        / "member_001"
        / "state_pointer.json"
    )
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unreadable"):
        clean_predecessor_checkpoint(project, predecessor, apply=False)
    assert state.is_file()
    assert malformed.is_file()


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
