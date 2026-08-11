from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pytest

from openamundsen_da.exceptions import LowDiskEmergencyError, LowDiskPauseError
from openamundsen_da.util.storage_budget import (
    EMERGENCY_USED_FRACTION,
    SOFT_USED_FRACTION,
    check_step_admission,
    estimate_compact_timeseries_bytes,
    estimate_step_forcing_bytes,
)


def _usage(*, total: int, used: int) -> shutil._ntuple_diskusage:
    return shutil._ntuple_diskusage(total, used, total - used)


def test_step_admission_uses_fixed_soft_and_emergency_limits(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    project.mkdir(parents=True)

    ok = check_step_admission(project, estimated_growth_bytes=5, usage=_usage(total=100, used=70))
    assert ok.used_fraction == pytest.approx(0.70)
    assert SOFT_USED_FRACTION == 0.80
    assert EMERGENCY_USED_FRACTION == 0.90

    with pytest.raises(LowDiskPauseError, match="80%"):
        check_step_admission(project, usage=_usage(total=100, used=80))
    draining = check_step_admission(
        project,
        usage=_usage(total=100, used=82),
        allow_existing_step_drain=True,
    )
    assert draining.projected_used_fraction == pytest.approx(0.87)
    with pytest.raises(LowDiskPauseError, match="completion estimate"):
        check_step_admission(
            project,
            usage=_usage(total=100, used=85),
            allow_existing_step_drain=True,
        )
    with pytest.raises(LowDiskPauseError, match="completion estimate"):
        check_step_admission(project, estimated_growth_bytes=21, usage=_usage(total=100, used=70))
    with pytest.raises(LowDiskEmergencyError, match="90%"):
        check_step_admission(project, usage=_usage(total=100, used=90))


def test_step_admission_rejects_invalid_estimates(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="non-negative"):
        check_step_admission(project, estimated_growth_bytes=-1, usage=_usage(total=100, used=1))


def test_forcing_estimate_scales_to_step_window_and_ensemble(tmp_path: Path) -> None:
    meteo = tmp_path / "meteo"
    meteo.mkdir()
    (meteo / "stations.csv").write_text("id,name,x,y,alt\na,A,0,0,0\n", encoding="utf-8")
    (meteo / "a.csv").write_text(
        "date,temp\n"
        "2023-01-01 00:00:00,273\n"
        "2023-01-02 00:00:00,274\n"
        "2023-01-03 00:00:00,275\n"
        "2023-01-04 00:00:00,276\n",
        encoding="utf-8",
    )

    short = estimate_step_forcing_bytes(
        meteo,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 2),
        ensemble_size=2,
    )
    full = estimate_step_forcing_bytes(
        meteo,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 4),
        ensemble_size=2,
    )

    assert 0 < short < full


def test_compact_export_estimate_counts_owned_csvs_with_margin(tmp_path: Path) -> None:
    project = tmp_path / "project"
    point = project / "steps" / "step_00" / "ensembles" / "prior" / "member_001" / "results" / "point_a.csv"
    forcing = point.parents[1] / "meteo" / "a.csv"
    point.parent.mkdir(parents=True)
    forcing.parent.mkdir(parents=True)
    point.write_bytes(b"12345")
    forcing.write_bytes(b"12345")
    unrelated = project / "results" / "table.csv"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"ignored")

    assert estimate_compact_timeseries_bytes(project) == 11
