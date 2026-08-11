from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from openamundsen_da.util import point_output as point_output_mod
from openamundsen_da.util.point_output import (
    _read_point_csv,
    compact_point_filenames,
    load_compact_point_series,
    validate_project_ensemble_points,
    write_project_ensemble_points,
)


def test_point_csv_rejects_unknown_text_but_accepts_missing_tokens(tmp_path: Path) -> None:
    path = tmp_path / "point.csv"
    path.write_text("date,swe\n2023-01-01,NA\n2023-01-02,bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognized nonnumeric"):
        _read_point_csv(path)
    path.write_text("date,swe\n2023-01-01,NA\n2023-01-02,\n", encoding="utf-8")
    _times, frame = _read_point_csv(path)
    assert frame["swe"].isna().all()
from openamundsen_da.methods.viz.fraction_series import (
    load_member_series,
    load_open_loop_fraction_series,
)


def _write_point(root: Path, rows: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "point_station.csv").write_text("date,snow_depth,swe\n" + rows, encoding="utf-8")


def test_point_output_retains_open_loop_and_all_members_losslessly(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    (project / "demo.yml").parent.mkdir(parents=True)
    (project / "demo.yml").write_text("start_date: 2023-01-01\nend_date: 2023-01-02\n", encoding="utf-8")
    for step_name, date, offset in (("step_00", "2023-01-01", 0), ("step_01", "2023-01-02", 10)):
        step = project / "steps" / step_name
        (step / f"{step_name}.yml").parent.mkdir(parents=True)
        (step / f"{step_name}.yml").write_text(
            f"start_date: {date}T00:00:00\nend_date: {date}T21:00:00\n",
            encoding="utf-8",
        )
        _write_point(
            step / "ensembles" / "prior" / "open_loop" / "results",
            f"{date} 00:00:00,{1.0 + offset},{100.0 + offset}\n",
        )
        for member_idx in (1, 2):
            _write_point(
                step / "ensembles" / "prior" / f"member_{member_idx:03d}" / "results",
                f"{date} 00:00:00,{member_idx + offset},{100 + member_idx + offset}\n",
            )

    output = write_project_ensemble_points(project)

    assert output.is_file()
    assert compact_point_filenames(project) == ["point_station.csv"]
    member = load_compact_point_series(
        project,
        point_filename="point_station.csv",
        member="member_002",
        variable="snow_depth",
    )
    assert member is not None
    np.testing.assert_array_equal(member.to_numpy(), [2.0, 12.0])
    with netCDF4.Dataset(output) as dataset:
        assert dataset.dimensions.keys() == {"time", "member", "point"}
        assert dataset.variables["snow_depth"].units == "m"
        assert dataset.variables["swe"].units == "kg m-2"
        assert dataset.variables["snow_depth"].filters()["zlib"] is True

    for csv_path in project.glob("steps/step_*/ensembles/prior/*/results/point_*.csv"):
        csv_path.unlink()
    open_loop = load_open_loop_fraction_series(project, "point_station.csv", "snow_depth")
    members = load_member_series(project, "point_station.csv", "snow_depth")
    assert open_loop is not None
    np.testing.assert_array_equal(open_loop["snow_depth"].to_numpy(), [1.0, 11.0])
    assert len(members) == 2
    np.testing.assert_array_equal(members[1].to_numpy(), [2.0, 12.0])


def test_point_output_refuses_incomplete_member_schema(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    step = project / "steps" / "step_00"
    (step / "step_00.yml").parent.mkdir(parents=True)
    (step / "step_00.yml").write_text(
        "start_date: 2023-01-01T00:00:00\nend_date: 2023-01-01T21:00:00\n",
        encoding="utf-8",
    )
    _write_point(step / "ensembles" / "prior" / "open_loop" / "results", "2023-01-01,1,100\n")
    member_results = step / "ensembles" / "prior" / "member_001" / "results"
    member_results.mkdir(parents=True)

    with pytest.raises(ValueError, match="Point files differ"):
        write_project_ensemble_points(project)


def test_point_output_mean_collapses_overlapping_step_boundaries(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    project.mkdir(parents=True)
    (project / "demo.yml").write_text("start_date: 2023-01-01\n", encoding="utf-8")
    rows_by_step = {
        "step_00": "2023-01-01,1,100\n2023-01-02,10,110\n",
        "step_01": "2023-01-02,14,114\n2023-01-03,3,103\n",
    }
    for step_name, rows in rows_by_step.items():
        step = project / "steps" / step_name
        step.mkdir(parents=True)
        (step / f"{step_name}.yml").write_text(
            "start_date: 2023-01-01\nend_date: 2023-01-03\n",
            encoding="utf-8",
        )
        for member in ("open_loop", "member_001"):
            _write_point(step / "ensembles" / "prior" / member / "results", rows)

    output = write_project_ensemble_points(project)
    compact = load_compact_point_series(
        project,
        point_filename="point_station.csv",
        member="member_001",
        variable="snow_depth",
    )
    assert compact is not None
    np.testing.assert_array_equal(compact.to_numpy(), [1.0, 12.0, 3.0])

    with netCDF4.Dataset(output, "a") as dataset:
        dataset.variables["snow_depth"][1, 1, 0] = 99.0
    with pytest.raises(ValueError, match="values do not match mean-collapsed"):
        validate_project_ensemble_points(project, output_nc=output)


def test_point_temp_validation_failure_preserves_accepted_target(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    step = project / "steps" / "step_00"
    step.mkdir(parents=True)
    (step / "step.yml").write_text("start_date: 2023-01-01\n", encoding="utf-8")
    for member in ("open_loop", "member_001"):
        _write_point(step / "ensembles" / "prior" / member / "results", "2023-01-01,1,100\n")
    target = project / "results" / "points" / "ensemble_points.nc"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"accepted")
    monkeypatch.setattr(
        point_output_mod,
        "validate_project_ensemble_points",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad temp")),
    )

    with pytest.raises(ValueError, match="bad temp"):
        write_project_ensemble_points(project)
    assert target.read_bytes() == b"accepted"
