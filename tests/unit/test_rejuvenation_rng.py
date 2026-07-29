from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import openamundsen_da.methods.pf.rejuvenate as rejuvenate_module
from openamundsen_da.methods.pf.rejuvenate import rejuvenate, validate_rejuvenation_manifest


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _prepare_project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    setup = tmp_path / "setup"
    project = setup / "projects" / "project_2022_2023"
    step_0 = project / "steps" / "step_00_init"
    step_1 = project / "steps" / "step_01_next"
    step_2 = project / "steps" / "step_02_next"
    _write_yaml(
        project / "project_2022_2023.yml",
        {
            "end_date": "2023-01-03 23:00:00",
            "data_assimilation": {
                "prior_forcing": {
                    "ensemble_size": 2,
                    "random_seed": 11,
                    "sigma_t": 0.5,
                    "mu_p": 0.1,
                    "sigma_p": 0.2,
                    "sigma_rh": 0.3,
                    "sigma_sw": 0.05,
                },
                "rejuvenation": {"seed": 113},
            },
        },
    )
    _write_yaml(step_1 / "step_01.yml", {"start_date": "2023-01-02", "end_date": "2023-01-02 23:00:00"})
    _write_yaml(step_2 / "step_02.yml", {"start_date": "2023-01-03", "end_date": "2023-01-03 23:00:00"})
    meteo = setup / "meteo"
    meteo.mkdir(parents=True)
    pd.DataFrame({"id": ["station_a"]}).to_csv(meteo / "stations.csv", index=False)
    pd.DataFrame(
        {
            "time": pd.date_range("2023-01-01", "2023-01-03 23:00:00", freq="h"),
            "temp": [273.15] * 72,
            "precip": [1.0] * 72,
            "rel_hum": [80.0] * 72,
            "sw_in": [100.0] * 72,
        }
    ).to_csv(meteo / "station_a.csv", index=False)
    for step in (step_0, step_1):
        for idx in range(1, 3):
            (step / "ensembles" / "posterior" / f"member_{idx:03d}").mkdir(parents=True, exist_ok=True)
    return setup, step_0, step_1, step_2


def test_rejuvenation_uses_distinct_event_keyed_perturbations_and_validates_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup, step_0, step_1, step_2 = _prepare_project(tmp_path)

    def sequential(func, tasks, **_kwargs):
        return [func(*task) for task in tasks]

    monkeypatch.setattr(rejuvenate_module, "run_tasks_with_pool", sequential)
    monkeypatch.setattr(rejuvenate_module, "pick_max_workers", lambda *args, **kwargs: 1)

    rejuvenate(setup_dir=setup, prev_step_dir=step_0, next_step_dir=step_1)
    first = validate_rejuvenation_manifest(setup_dir=setup, prev_step_dir=step_0, next_step_dir=step_1)
    rejuvenate(setup_dir=setup, prev_step_dir=step_1, next_step_dir=step_2)
    second = validate_rejuvenation_manifest(setup_dir=setup, prev_step_dir=step_1, next_step_dir=step_2)

    first_vector = [
        (row["delta_T"], row["f_p"], row["delta_dew_point"], row["f_sw"])
        for row in first["members"]
    ]
    second_vector = [
        (row["delta_T"], row["f_p"], row["delta_dew_point"], row["f_sw"])
        for row in second["members"]
    ]
    assert first_vector != second_vector
    assert first["rng_scheme"] == "keyed-v1"
    assert first["mu_p"] == pytest.approx(0.1)

    target_csv = step_1 / "ensembles" / "prior" / "member_001" / "meteo" / "station_a.csv"
    frame = pd.read_csv(target_csv)
    frame.loc[0, "temp"] += 1.0
    frame.to_csv(target_csv, index=False)
    with pytest.raises(RuntimeError, match="output_inventory_sha256"):
        validate_rejuvenation_manifest(setup_dir=setup, prev_step_dir=step_0, next_step_dir=step_1)


def test_rejuvenation_resume_ignores_diagnostics_but_binds_weight_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup, step_0, step_1, _step_2 = _prepare_project(tmp_path)
    previous_assim = step_0 / "assim"
    previous_assim.mkdir(parents=True, exist_ok=True)
    weights = previous_assim / "weights_station_hs_20230101.csv"
    weights.write_text("member_id,weight\nmember_001,0.5\nmember_002,0.5\n", encoding="utf-8")
    (previous_assim / "weights_station_hs_20230101_manifest.json").write_text(
        '{"status":"complete"}\n',
        encoding="utf-8",
    )
    (previous_assim / "resample_indices_20230101.csv").write_text(
        "child_member_id,source_member_id\nmember_001,member_001\nmember_002,member_002\n",
        encoding="utf-8",
    )
    (previous_assim / "resample_manifest_20230101.json").write_text(
        '{"status":"complete"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rejuvenate_module,
        "run_tasks_with_pool",
        lambda func, tasks, **_kwargs: [func(*task) for task in tasks],
    )
    monkeypatch.setattr(rejuvenate_module, "pick_max_workers", lambda *args, **kwargs: 1)

    rejuvenate(setup_dir=setup, prev_step_dir=step_0, next_step_dir=step_1)
    diagnostic = previous_assim / "station_diagnostics_station_hs_20230101.png"
    diagnostic.write_bytes(b"rendered later")
    validate_rejuvenation_manifest(setup_dir=setup, prev_step_dir=step_0, next_step_dir=step_1)

    weights.write_text("member_id,weight\nmember_001,0.9\nmember_002,0.1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input_inventory_sha256"):
        validate_rejuvenation_manifest(setup_dir=setup, prev_step_dir=step_0, next_step_dir=step_1)


def test_rejuvenation_validation_allows_pointers_removed_by_completed_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup, step_0, step_1, _step_2 = _prepare_project(tmp_path)
    source_pointers = []
    for idx in range(1, 3):
        pointer = step_0 / "ensembles" / "posterior" / f"member_{idx:03d}" / "state_pointer.json"
        pointer.write_text(json.dumps({"path": f"/states/member_{idx:03d}.pickle.gz"}), encoding="utf-8")
        source_pointers.append(pointer)

    monkeypatch.setattr(
        rejuvenate_module,
        "run_tasks_with_pool",
        lambda func, tasks, **_kwargs: [func(*task) for task in tasks],
    )
    monkeypatch.setattr(rejuvenate_module, "pick_max_workers", lambda *args, **kwargs: 1)

    rejuvenate(setup_dir=setup, prev_step_dir=step_0, next_step_dir=step_1)
    target_pointers = sorted((step_1 / "ensembles" / "prior").glob("member_*/state_pointer.json"))
    pointers = [*source_pointers, *target_pointers]
    assert len(target_pointers) == 2
    validate_rejuvenation_manifest(setup_dir=setup, prev_step_dir=step_0, next_step_dir=step_1)

    pointers[0].unlink()
    with pytest.raises(RuntimeError, match="input_inventory_sha256"):
        validate_rejuvenation_manifest(setup_dir=setup, prev_step_dir=step_0, next_step_dir=step_1)

    project = step_1.parents[1]
    cleanup_paths = [pointer.relative_to(project).as_posix() for pointer in pointers]
    for pointer in pointers[1:]:
        pointer.unlink()
    run_manifest = {
        "schema_version": 1,
        "status": "success",
        "stages": {"cleanup": "success"},
        "cleanup": {"deleted_paths": cleanup_paths, "failures": []},
    }
    manifest_path = project / "results" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")

    validate_rejuvenation_manifest(setup_dir=setup, prev_step_dir=step_0, next_step_dir=step_1)
