from __future__ import annotations

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
