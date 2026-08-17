from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from openamundsen_da.io.paths import meteo_dir_for_member
from openamundsen_da.util import forcing_output as forcing_output_mod
from openamundsen_da.util.forcing_output import (
    _collapsed_forcing_frame,
    _read_forcing_csv,
    compact_forcing_members,
    compact_forcing_stations,
    load_compact_forcing_series,
    validate_project_ensemble_forcing,
    write_project_ensemble_forcing,
)
from openamundsen_da.util.runtime_generation import ensure_runtime_generation


def test_forcing_csv_rejects_unknown_text_but_accepts_missing_tokens(tmp_path: Path) -> None:
    path = tmp_path / "forcing.csv"
    path.write_text("date,temp\n2023-01-01,NA\n2023-01-02,bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognized nonnumeric"):
        _read_forcing_csv(path)
    path.write_text("date,temp\n2023-01-01,NA\n2023-01-02,\n", encoding="utf-8")
    _times, frame = _read_forcing_csv(path)
    assert frame["temp"].isna().all()


def _write_forcing(root: Path, date: str, temp: float, precip: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "stations.csv").write_text("id,name,x,y,alt\na,A,0,0,0\n", encoding="utf-8")
    (root / "a.csv").write_text(
        f"date,temp,precip\n{date} 00:00:00,{temp},{precip}\n",
        encoding="utf-8",
    )


def test_forcing_output_retains_consumed_open_loop_and_member_values(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    project.mkdir(parents=True)
    (project / "demo.yml").write_text("start_date: 2023-01-01\nend_date: 2023-01-02\n", encoding="utf-8")
    for step_name, date, offset in (("step_00", "2023-01-01", 0), ("step_01", "2023-01-02", 10)):
        step = project / "steps" / step_name
        step.mkdir(parents=True)
        (step / f"{step_name}.yml").write_text(
            f"start_date: {date}T00:00:00\nend_date: {date}T21:00:00\n",
            encoding="utf-8",
        )
        _write_forcing(step / "ensembles" / "prior" / "open_loop" / "meteo", date, 270 + offset, 1 + offset)
        for member_idx in (1, 2):
            _write_forcing(
                step / "ensembles" / "prior" / f"member_{member_idx:03d}" / "meteo",
                date,
                270 + offset + member_idx,
                1 + offset + member_idx,
            )

    output = write_project_ensemble_forcing(project)

    assert compact_forcing_stations(project) == ["a.csv"]
    assert compact_forcing_members(project) == ["member_001", "member_002"]
    series = load_compact_forcing_series(
        project,
        station_filename="a.csv",
        member="member_002",
        variables=["temp", "precip"],
    )
    assert series is not None
    np.testing.assert_array_equal(series["temp"].to_numpy(), [272.0, 282.0])
    with netCDF4.Dataset(output) as dataset:
        assert dataset.variables["temp"].units == "K"
        assert dataset.variables["precip"].filters()["zlib"] is True


def test_generation_routed_forcing_compacts_identically_to_legacy_layout(
    tmp_path: Path,
) -> None:
    outputs: list[np.ndarray] = []
    for layout in ("legacy", "generation"):
        project = tmp_path / layout / "projects/demo"
        project.mkdir(parents=True)
        (project / "demo.yml").write_text(
            "start_date: 2023-01-01\nend_date: 2023-01-02\n",
            encoding="utf-8",
        )
        if layout == "generation":
            ensure_runtime_generation(project)
        step = project / "steps/step_00"
        step.mkdir(parents=True)
        (step / "step_00.yml").write_text(
            "start_date: 2023-01-01T00:00:00\nend_date: 2023-01-01T21:00:00\n",
            encoding="utf-8",
        )
        for index, member_name in enumerate(("open_loop", "member_001")):
            member = step / "ensembles/prior" / member_name
            member.mkdir(parents=True)
            _write_forcing(
                meteo_dir_for_member(member),
                "2023-01-01",
                270 + index,
                1 + index,
            )

        write_project_ensemble_forcing(project)
        series = load_compact_forcing_series(
            project,
            station_filename="a.csv",
            member="member_001",
            variables=["temp", "precip"],
        )
        assert series is not None
        outputs.append(series[["temp", "precip"]].to_numpy())

    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_forcing_output_refuses_incomplete_member_schema(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    step = project / "steps" / "step_00"
    step.mkdir(parents=True)
    (step / "step_00.yml").write_text(
        "start_date: 2023-01-01T00:00:00\nend_date: 2023-01-01T21:00:00\n",
        encoding="utf-8",
    )
    _write_forcing(step / "ensembles" / "prior" / "open_loop" / "meteo", "2023-01-01", 270, 1)
    (step / "ensembles" / "prior" / "member_001" / "meteo").mkdir(parents=True)

    with pytest.raises(ValueError, match="Forcing station files differ"):
        write_project_ensemble_forcing(project)


def test_forcing_output_mean_collapses_overlapping_step_boundaries(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    project.mkdir(parents=True)
    (project / "demo.yml").write_text("start_date: 2023-01-01\n", encoding="utf-8")
    rows_by_step = {
        "step_00": (
            "date,temp,precip\n"
            "2023-01-01,270,1\n"
            "2023-01-02,280,10\n"
        ),
        "step_01": (
            "date,temp,precip\n"
            "2023-01-02,284,14\n"
            "2023-01-03,273,3\n"
        ),
    }
    for step_name, contents in rows_by_step.items():
        step = project / "steps" / step_name
        step.mkdir(parents=True)
        (step / f"{step_name}.yml").write_text(
            "start_date: 2023-01-01\nend_date: 2023-01-03\n",
            encoding="utf-8",
        )
        for member in ("open_loop", "member_001"):
            meteo = step / "ensembles" / "prior" / member / "meteo"
            meteo.mkdir(parents=True)
            (meteo / "stations.csv").write_text(
                "id,name,x,y,alt\na,A,0,0,0\n",
                encoding="utf-8",
            )
            (meteo / "a.csv").write_text(contents, encoding="utf-8")

    output = write_project_ensemble_forcing(project)
    compact = load_compact_forcing_series(
        project,
        station_filename="a.csv",
        member="member_001",
        variables=["temp", "precip"],
    )
    assert compact is not None
    np.testing.assert_array_equal(compact["temp"].to_numpy(), [270.0, 282.0, 273.0])
    np.testing.assert_array_equal(compact["precip"].to_numpy(), [1.0, 12.0, 3.0])

    with netCDF4.Dataset(output, "a") as dataset:
        dataset.variables["temp"][1, 1, 0] = 999.0
    with pytest.raises(ValueError, match="values do not match mean-collapsed"):
        validate_project_ensemble_forcing(project, output_nc=output)


def test_forcing_collapse_excludes_empty_all_na_frames(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    for step_name, rows in (
        ("step_00", "date,temp,precip\n2023-01-01,,\n"),
        ("step_01", "date,temp,precip\n2023-01-02,273.15,1.0\n"),
    ):
        step = project / "steps" / step_name
        step.mkdir(parents=True)
        (step / f"{step_name}.yml").write_text(
            "start_date: 2023-01-01\nend_date: 2023-01-02\n",
            encoding="utf-8",
        )
        for member in ("open_loop", "member_001"):
            meteo = step / "ensembles" / "prior" / member / "meteo"
            meteo.mkdir(parents=True)
            (meteo / "stations.csv").write_text(
                "id,name,x,y,alt\na,A,0,0,0\n",
                encoding="utf-8",
            )
            (meteo / "a.csv").write_text(rows, encoding="utf-8")

    frame = _collapsed_forcing_frame(
        sorted((project / "steps").iterdir()),
        member="member_001",
        station="a",
    )

    assert frame.index.tolist() == [np.datetime64("2023-01-02")]
    np.testing.assert_array_equal(frame[["temp", "precip"]].to_numpy(), [[273.15, 1.0]])


def test_forcing_temp_validation_failure_preserves_accepted_target(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    step = project / "steps" / "step_00"
    step.mkdir(parents=True)
    (step / "step.yml").write_text("start_date: 2023-01-01\n", encoding="utf-8")
    for member in ("open_loop", "member_001"):
        _write_forcing(step / "ensembles" / "prior" / member / "meteo", "2023-01-01", 270, 1)
    target = project / "results" / "forcing" / "ensemble_forcing.nc"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"accepted")
    monkeypatch.setattr(
        forcing_output_mod,
        "validate_project_ensemble_forcing",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad temp")),
    )

    with pytest.raises(ValueError, match="bad temp"):
        write_project_ensemble_forcing(project)
    assert target.read_bytes() == b"accepted"
