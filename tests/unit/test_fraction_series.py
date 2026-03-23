from __future__ import annotations

from pathlib import Path

import pandas as pd

from openamundsen_da.methods.viz.fraction_series import (
    load_member_series,
    load_open_loop_fraction_series,
)


def _write_series_csv(path: Path, value_col: str, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": [t for t, _ in rows], value_col: [v for _, v in rows]}).to_csv(path, index=False)


def test_load_member_series_stitches_members_across_steps(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step_00 = project_dir / "steps" / "step_00_init" / "ensembles" / "prior"
    step_01 = project_dir / "steps" / "step_01_next" / "ensembles" / "prior"

    _write_series_csv(
        step_00 / "member_001" / "results" / "point_swe_roi.csv",
        "swe",
        [("2023-01-01", 10.0), ("2023-01-02", 11.0)],
    )
    _write_series_csv(
        step_01 / "member_001" / "results" / "point_swe_roi.csv",
        "swe",
        [("2023-01-02", 13.0), ("2023-01-03", 14.0)],
    )
    _write_series_csv(
        step_00 / "member_002" / "results" / "point_swe_roi.csv",
        "swe",
        [("2023-01-01", 20.0), ("2023-01-02", 21.0)],
    )
    _write_series_csv(
        step_01 / "member_002" / "results" / "point_swe_roi.csv",
        "swe",
        [("2023-01-02", 23.0), ("2023-01-03", 24.0)],
    )

    member_series = load_member_series(project_dir, "point_swe_roi.csv", "swe")

    assert len(member_series) == 2
    assert list(member_series[0].index) == list(pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))
    assert list(member_series[0].values) == [10.0, 12.0, 14.0]
    assert list(member_series[1].values) == [20.0, 22.0, 24.0]


def test_load_open_loop_fraction_series_collapses_step_overlap(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step_00 = project_dir / "steps" / "step_00_init" / "ensembles" / "prior" / "open_loop" / "results"
    step_01 = project_dir / "steps" / "step_01_next" / "ensembles" / "prior" / "open_loop" / "results"

    _write_series_csv(
        step_00 / "point_swe_roi.csv",
        "swe",
        [("2023-01-01", 100.0), ("2023-01-02", 110.0)],
    )
    _write_series_csv(
        step_01 / "point_swe_roi.csv",
        "swe",
        [("2023-01-02", 130.0), ("2023-01-03", 140.0)],
    )

    series = load_open_loop_fraction_series(project_dir, "point_swe_roi.csv", "swe")

    assert series is not None
    assert list(series["date"]) == list(pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))
    assert list(series["swe"]) == [100.0, 120.0, 140.0]
