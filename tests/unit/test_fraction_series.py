from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openamundsen_da.methods.viz.fraction_series import (
    default_result_overview_output,
    load_fraction_series,
    load_member_series,
    load_named_member_series,
    load_open_loop_fraction_series,
    load_weighted_member_envelope,
)
from openamundsen_da.methods.pf.weights import write_prior_weights


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


def test_load_weighted_member_envelope_uses_each_step_ledger(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    for step_name, day, values, weights in (
        ("step_00_init", "2023-01-01", (0.0, 10.0), (0.8, 0.2)),
        ("step_01_next", "2023-01-02", (2.0, 12.0), (0.25, 0.75)),
    ):
        step = project_dir / "steps" / step_name
        for member_idx, value in enumerate(values, start=1):
            _write_series_csv(
                step / "ensembles" / "prior" / f"member_{member_idx:03d}" / "results" / "point_swe_roi.csv",
                "swe",
                [(day, value)],
            )
        write_prior_weights(
            step,
            member_ids=["member_001", "member_002"],
            weights=weights,
            mode="carried_posterior",
        )

    envelope = load_weighted_member_envelope(project_dir, "point_swe_roi.csv", "swe")

    assert envelope is not None
    assert list(envelope["value_mean"]) == pytest.approx([2.0, 9.5])
    assert list(envelope["value_min"]) == [0.0, 2.0]
    assert list(envelope["value_max"]) == [10.0, 12.0]
    assert set(envelope["weighting"]) == {"pf_prior_ledger"}


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


def test_default_result_overview_output_uses_new_filename(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"

    out_path = default_result_overview_output(project_dir, None)

    assert out_path == project_dir / "results" / "plots" / "results" / "result_overview.png"


def test_load_named_member_series_returns_member_mapping(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step_00 = project_dir / "steps" / "step_00_init" / "ensembles" / "prior"
    step_01 = project_dir / "steps" / "step_01_next" / "ensembles" / "prior"

    _write_series_csv(
        step_00 / "member_001" / "results" / "point_snow_depth_roi.csv",
        "snow_depth",
        [("2023-01-01", 0.1)],
    )
    _write_series_csv(
        step_01 / "member_001" / "results" / "point_snow_depth_roi.csv",
        "snow_depth",
        [("2023-01-02", 0.2)],
    )
    _write_series_csv(
        step_00 / "member_002" / "results" / "point_snow_depth_roi.csv",
        "snow_depth",
        [("2023-01-01", 0.3)],
    )
    _write_series_csv(
        step_01 / "member_002" / "results" / "point_snow_depth_roi.csv",
        "snow_depth",
        [("2023-01-02", 0.4)],
    )

    member_series = load_named_member_series(project_dir, "point_snow_depth_roi.csv", "snow_depth")

    assert sorted(member_series) == ["member_001", "member_002"]
    assert list(member_series["member_001"].values) == [0.1, 0.2]
    assert list(member_series["member_002"].values) == [0.3, 0.4]


def test_load_open_loop_fraction_series_supports_mixed_timestamp_formats(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step_00 = project_dir / "steps" / "step_00_init" / "ensembles" / "prior" / "open_loop" / "results"
    step_01 = project_dir / "steps" / "step_01_next" / "ensembles" / "prior" / "open_loop" / "results"

    _write_series_csv(
        step_00 / "point_swe_roi.csv",
        "swe",
        [("2023-01-01", 100.0), ("2023-01-01 03:00:00", 110.0)],
    )
    _write_series_csv(
        step_01 / "point_swe_roi.csv",
        "swe",
        [("2023-01-01 03:00:00", 130.0), ("2023-01-02", 140.0)],
    )

    series = load_open_loop_fraction_series(project_dir, "point_swe_roi.csv", "swe")

    assert series is not None
    assert list(series["date"]) == list(
        pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 03:00:00", "2023-01-02 00:00:00"])
    )
    assert list(series["swe"]) == [100.0, 120.0, 140.0]


def test_load_named_member_series_supports_mixed_timestamp_formats(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step_00 = project_dir / "steps" / "step_00_init" / "ensembles" / "prior"
    step_01 = project_dir / "steps" / "step_01_next" / "ensembles" / "prior"

    _write_series_csv(
        step_00 / "member_001" / "results" / "point_snow_depth_roi.csv",
        "snow_depth",
        [("2023-01-01", 0.1), ("2023-01-01 03:00:00", 0.2)],
    )
    _write_series_csv(
        step_01 / "member_001" / "results" / "point_snow_depth_roi.csv",
        "snow_depth",
        [("2023-01-01 03:00:00", 0.4), ("2023-01-02", 0.5)],
    )

    member_series = load_named_member_series(project_dir, "point_snow_depth_roi.csv", "snow_depth")

    assert list(member_series["member_001"].index) == list(
        pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 03:00:00", "2023-01-02 00:00:00"])
    )
    assert list(member_series["member_001"].values) == pytest.approx([0.1, 0.3, 0.5])


def test_load_fraction_series_can_preserve_missing_wsl_values_without_changing_default(tmp_path: Path) -> None:
    csv_path = tmp_path / "wet_snow_line_diagnostics.csv"
    pd.DataFrame(
        {
            "date": ["2023-04-29", "2023-05-11", "2023-05-15"],
            "wet_snow_line": [2450.0, float("nan"), 2550.0],
        }
    ).to_csv(csv_path, index=False)

    default_loaded = load_fraction_series(csv_path, "wet_snow_line")
    preserved_loaded = load_fraction_series(csv_path, "wet_snow_line", preserve_missing_values=True)

    assert default_loaded is not None
    assert preserved_loaded is not None
    assert list(default_loaded["date"]) == list(pd.to_datetime(["2023-04-29", "2023-05-15"]))
    assert list(default_loaded["wet_snow_line"]) == [2450.0, 2550.0]
    assert list(preserved_loaded["date"]) == list(pd.to_datetime(["2023-04-29", "2023-05-11", "2023-05-15"]))
    assert pd.isna(preserved_loaded.iloc[1]["wet_snow_line"])


def test_load_open_loop_fraction_series_can_preserve_missing_wsl_gaps_across_steps(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step_00 = project_dir / "steps" / "step_00_init" / "ensembles" / "prior" / "open_loop" / "results"
    step_01 = project_dir / "steps" / "step_01_next" / "ensembles" / "prior" / "open_loop" / "results"
    step_00.mkdir(parents=True, exist_ok=True)
    step_01.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "time": ["2023-04-29", "2023-05-11"],
            "wet_snow_line": [2450.0, float("nan")],
        }
    ).to_csv(step_00 / "point_wet_snow_line_roi.csv", index=False)
    pd.DataFrame(
        {
            "time": ["2023-05-11", "2023-05-15"],
            "wet_snow_line": [float("nan"), 2550.0],
        }
    ).to_csv(step_01 / "point_wet_snow_line_roi.csv", index=False)

    default_loaded = load_open_loop_fraction_series(project_dir, "point_wet_snow_line_roi.csv", "wet_snow_line")
    preserved_loaded = load_open_loop_fraction_series(
        project_dir,
        "point_wet_snow_line_roi.csv",
        "wet_snow_line",
        preserve_missing_values=True,
    )

    assert default_loaded is not None
    assert preserved_loaded is not None
    assert list(default_loaded["date"]) == list(pd.to_datetime(["2023-04-29", "2023-05-15"]))
    assert list(default_loaded["wet_snow_line"]) == [2450.0, 2550.0]
    assert list(preserved_loaded["date"]) == list(pd.to_datetime(["2023-04-29", "2023-05-11", "2023-05-15"]))
    assert pd.isna(preserved_loaded.iloc[1]["wet_snow_line"])


def test_load_member_series_can_preserve_missing_wsl_gaps_across_steps(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step_00 = project_dir / "steps" / "step_00_init" / "ensembles" / "prior"
    step_01 = project_dir / "steps" / "step_01_next" / "ensembles" / "prior"

    _write_series_csv(
        step_00 / "member_001" / "results" / "point_wet_snow_line_roi.csv",
        "wet_snow_line",
        [("2023-04-29", 2450.0), ("2023-05-11", float("nan"))],
    )
    _write_series_csv(
        step_01 / "member_001" / "results" / "point_wet_snow_line_roi.csv",
        "wet_snow_line",
        [("2023-05-11", float("nan")), ("2023-05-15", 2550.0)],
    )

    default_loaded = load_member_series(project_dir, "point_wet_snow_line_roi.csv", "wet_snow_line")
    preserved_loaded = load_member_series(
        project_dir,
        "point_wet_snow_line_roi.csv",
        "wet_snow_line",
        preserve_missing_values=True,
    )

    assert len(default_loaded) == 1
    assert len(preserved_loaded) == 1
    assert list(default_loaded[0].index) == list(pd.to_datetime(["2023-04-29", "2023-05-15"]))
    assert list(default_loaded[0].values) == [2450.0, 2550.0]
    assert list(preserved_loaded[0].index) == list(pd.to_datetime(["2023-04-29", "2023-05-11", "2023-05-15"]))
    assert pd.isna(preserved_loaded[0].iloc[1])


def test_load_named_member_series_can_preserve_missing_wsl_gaps_across_steps(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step_00 = project_dir / "steps" / "step_00_init" / "ensembles" / "prior"
    step_01 = project_dir / "steps" / "step_01_next" / "ensembles" / "prior"

    _write_series_csv(
        step_00 / "member_001" / "results" / "point_wet_snow_line_roi.csv",
        "wet_snow_line",
        [("2023-04-29", 2450.0), ("2023-05-11", float("nan"))],
    )
    _write_series_csv(
        step_01 / "member_001" / "results" / "point_wet_snow_line_roi.csv",
        "wet_snow_line",
        [("2023-05-11", float("nan")), ("2023-05-15", 2550.0)],
    )

    default_loaded = load_named_member_series(project_dir, "point_wet_snow_line_roi.csv", "wet_snow_line")
    preserved_loaded = load_named_member_series(
        project_dir,
        "point_wet_snow_line_roi.csv",
        "wet_snow_line",
        preserve_missing_values=True,
    )

    assert list(default_loaded["member_001"].index) == list(pd.to_datetime(["2023-04-29", "2023-05-15"]))
    assert list(default_loaded["member_001"].values) == [2450.0, 2550.0]
    assert list(preserved_loaded["member_001"].index) == list(
        pd.to_datetime(["2023-04-29", "2023-05-11", "2023-05-15"])
    )
    assert pd.isna(preserved_loaded["member_001"].iloc[1])
