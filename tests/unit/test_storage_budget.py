from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pytest

from openamundsen_da.exceptions import LowDiskEmergencyError, LowDiskPauseError
from openamundsen_da.util.storage_budget import (
    DEFAULT_POINT_VARIABLE_COUNT,
    EMERGENCY_USED_FRACTION,
    ProjectStorageEstimate,
    SOFT_USED_FRACTION,
    StorageReservationProject,
    check_step_admission,
    estimate_compact_timeseries_bytes,
    estimate_coordinated_storage_reserve,
    estimate_parent_compact_merge_bytes,
    estimate_project_storage_components,
    estimate_project_storage_reserve,
    estimate_step_forcing_bytes,
    _point_storage_bound,
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


def test_forcing_estimate_scales_each_station_coverage_independently(tmp_path: Path) -> None:
    meteo = tmp_path / "meteo"
    meteo.mkdir()
    (meteo / "stations.csv").write_text("id\na\nb\n", encoding="utf-8")
    (meteo / "a.csv").write_text(
        "date,temp\n2023-01-01,273\n2023-01-11,274\n",
        encoding="utf-8",
    )
    short_record = "date,temp\n2023-01-01,273\n2023-01-02,274\n"
    (meteo / "b.csv").write_text(short_record, encoding="utf-8")

    estimated = estimate_step_forcing_bytes(
        meteo,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 2),
        ensemble_size=1,
    )

    short_full_bound = int(len(short_record.encode()) * 1.35) * 2
    assert estimated >= short_full_bound


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


def test_point_bound_counts_default_columns_and_explicit_layers(tmp_path: Path) -> None:
    setup = tmp_path / "setup"
    meteo = setup / "meteo"
    meteo.mkdir(parents=True)
    (meteo / "stations.csv").write_text("id\na\n", encoding="utf-8")
    steps = [(tmp_path / "step", datetime(2023, 1, 1), datetime(2023, 1, 1))]
    default_only = _point_storage_bound(
        setup_dir=setup,
        setup_cfg={},
        output_data={"timeseries": {"add_default_points": True, "add_default_variables": True}},
        steps=steps,
        model_timestep="1D",
        member_count=1,
    )
    explicit_layers = _point_storage_bound(
        setup_dir=setup,
        setup_cfg={"snow": {"min_thickness": [0.1, 0.2, 0.4]}},
        output_data={
            "timeseries": {
                "add_default_points": True,
                "add_default_variables": True,
                "variables": [{"var": "snow.temp"}],
            }
        },
        steps=steps,
        model_timestep="1D",
        member_count=1,
    )

    assert DEFAULT_POINT_VARIABLE_COUNT == 40
    assert explicit_layers > default_only


def test_project_reserve_covers_pending_steps_and_final_compaction(tmp_path: Path) -> None:
    setup = tmp_path / "setup"
    project = setup / "projects" / "demo"
    meteo = setup / "meteo"
    meteo.mkdir(parents=True)
    (meteo / "stations.csv").write_text("id,name,x,y,alt\na,A,0,0,0\n", encoding="utf-8")
    (meteo / "a.csv").write_text(
        "date,temp\n"
        "2023-01-01,273\n"
        "2023-01-02,274\n"
        "2023-01-03,275\n"
        "2023-01-04,276\n",
        encoding="utf-8",
    )
    project.mkdir(parents=True)
    (setup / "north_tyrol.yml").write_text(
        "domain: demo\n"
        "resolution: 100\n"
        "timestep: 1D\n"
        "input_data:\n  grids:\n    dir: grids\n"
        "output_data:\n"
        "  timeseries:\n"
        "    add_default_points: false\n"
        "    add_default_variables: false\n"
        "  grids:\n"
        "    format: netcdf\n"
        "    variables:\n"
        "      - var: snow.depth\n"
        "        name: snowdepth_daily\n"
        "        freq: 1D\n",
        encoding="utf-8",
    )
    (project / "demo.yml").write_text(
        "data_assimilation:\n"
        "  prior_forcing:\n"
        "    ensemble_size: 2\n"
        "  output:\n"
        "    retention: compact\n"
        "    grids:\n"
        "      variables:\n"
        "        - var: snowdepth_daily\n"
        "          metrics: [open_loop, ens_mean]\n",
        encoding="utf-8",
    )
    windows = (
        ("step_00", "2023-01-01", "2023-01-02"),
        ("step_01", "2023-01-03", "2023-01-04"),
    )
    estimates = []
    for name, start, end in windows:
        step = project / "steps" / name
        step.mkdir(parents=True)
        (step / f"{name}.yml").write_text(
            f"start_date: {start}\nend_date: {end}\n",
            encoding="utf-8",
        )
        estimates.append(
            estimate_step_forcing_bytes(
                meteo,
                start=datetime.fromisoformat(start),
                end=datetime.fromisoformat(end),
                ensemble_size=2,
            )
        )

    reserve = estimate_project_storage_reserve(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=4,
    )
    components = estimate_project_storage_components(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=4,
    )
    assert reserve == components.total_bytes
    assert components.forcing_bytes == sum(estimates)
    assert components.member_grid_bytes > 0
    assert components.restart_baseline_bytes > 0
    assert components.restart_transition_bytes > 0
    assert components.compact_timeseries_bytes >= int(sum(estimates) * 1.10)
    assert components.compact_grid_bytes > 0

    (project / "demo.yml").write_text(
        "data_assimilation:\n"
        "  prior_forcing:\n"
        "    ensemble_size: 2\n"
        "  output:\n"
        "    retention: full\n"
        "    grids:\n"
        "      variables:\n"
        "        - var: snowdepth_daily\n"
        "          metrics: [open_loop, ens_mean]\n",
        encoding="utf-8",
    )
    full = estimate_project_storage_components(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=4,
    )
    assert full.forcing_bytes == sum(estimates)
    assert full.compact_timeseries_bytes == 0
    assert full.restart_transition_bytes == 0
    assert full.restart_baseline_bytes > components.restart_baseline_bytes
    (project / "demo.yml").write_text(
        "data_assimilation:\n"
        "  prior_forcing:\n"
        "    ensemble_size: 2\n"
        "  output:\n"
        "    retention: compact\n"
        "    grids:\n"
        "      variables:\n"
        "        - var: snowdepth_daily\n"
        "          metrics: [open_loop, ens_mean]\n",
        encoding="utf-8",
    )

    for name, _start, _end in windows:
        for member in ("open_loop", "member_001", "member_002"):
            member_root = project / "steps" / name / "ensembles" / "prior" / member
            meteo_out = member_root / "meteo" / "a.csv"
            grid_out = member_root / "results" / "output_grids.nc"
            state_out = member_root / "results" / "model_state.pickle.gz"
            meteo_out.parent.mkdir(parents=True, exist_ok=True)
            grid_out.parent.mkdir(parents=True, exist_ok=True)
            meteo_out.write_bytes(b"forcing")
            grid_out.write_bytes(b"grid")
            state_out.write_bytes(b"state")
    for path in (
        project / "results" / "points" / "ensemble_points.nc",
        project / "results" / "forcing" / "ensemble_forcing.nc",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")

    completed = estimate_project_storage_components(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=4,
    )
    assert completed.compact_timeseries_bytes == 0
    assert completed.total_bytes < components.total_bytes

    overwritten = estimate_project_storage_components(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=4,
        overwrite=True,
    )
    assert overwritten.compact_timeseries_bytes > 0
    assert overwritten.compact_grid_bytes > 0


def test_coordinator_reserves_all_growth_concurrent_states_and_parent_merge(
    monkeypatch,
    tmp_path: Path,
) -> None:
    projects = tuple(
        StorageReservationProject(
            setup_dir=tmp_path / f"S{index}",
            project_dir=tmp_path / f"P{index}",
            grid_cell_count=10,
        )
        for index in range(3)
    )
    estimates = {
        str(projects[0].project_dir): ProjectStorageEstimate(10, 20, 30, 40, 5, 50, 60),
        str(projects[1].project_dir): ProjectStorageEstimate(11, 21, 31, 41, 15, 51, 61),
        str(projects[2].project_dir): ProjectStorageEstimate(12, 22, 32, 42, 25, 52, 62),
    }

    monkeypatch.setattr(
        "openamundsen_da.util.storage_budget.estimate_project_storage_components",
        lambda *, project_dir, **_kwargs: estimates[str(project_dir)],
    )
    reserve, observed = estimate_coordinated_storage_reserve(
        projects,
        outer_workers=2,
        parent_merge_reserve_bytes=100,
    )

    expected_non_transition = sum(item.non_transition_bytes for item in estimates.values())
    assert reserve == expected_non_transition + 25 + 15 + 100
    assert observed == estimates

    project_root = tmp_path / "shared"
    project_root.mkdir()
    with pytest.raises(LowDiskPauseError, match="completion estimate"):
        check_step_admission(
            project_root,
            estimated_growth_bytes=reserve,
            usage=_usage(total=1_000, used=500),
        )


def test_parent_merge_reserves_one_atomic_full_grid_temporary(tmp_path: Path) -> None:
    setup = tmp_path / "setup"
    project = setup / "projects" / "demo"
    project.mkdir(parents=True)
    (setup / "north_tyrol.yml").write_text(
        "domain: demo\nresolution: 100\ntimestep: 1D\n"
        "output_data:\n"
        "  grids:\n"
        "    variables:\n"
        "      - {var: snow.depth, name: snowdepth_daily, freq: 1D}\n",
        encoding="utf-8",
    )
    (project / "demo.yml").write_text(
        "data_assimilation:\n"
        "  prior_forcing: {ensemble_size: 2}\n"
        "  output:\n"
        "    grids:\n"
        "      variables:\n"
        "        - {var: snowdepth_daily, metrics: [open_loop, ens_mean]}\n",
        encoding="utf-8",
    )
    for name, start, end in (
        ("step_00", "2023-01-01", "2023-01-02"),
        ("step_01", "2023-01-03", "2023-01-04"),
    ):
        step = project / "steps" / name
        step.mkdir(parents=True)
        (step / f"{name}.yml").write_text(
            f"start_date: {start}\nend_date: {end}\n",
            encoding="utf-8",
        )

    reserve = estimate_parent_compact_merge_bytes(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=100,
    )
    assert reserve > 100 * 2 * 4 * 8
