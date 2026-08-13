from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from openamundsen_da.exceptions import LowDiskEmergencyError, LowDiskPauseError
from openamundsen_da.pipeline.project_skeleton import create_project_skeleton
from openamundsen_da.util.storage_budget import (
    DEFAULT_POINT_VARIABLE_COUNT,
    EMERGENCY_USED_FRACTION,
    EUREGIO_ES30_RETAINED_DIAGNOSTICS_BYTES,
    FORCING_PLOT_BYTES_PER_STATION_MEMBER_DAY,
    ProjectStorageEstimate,
    SOFT_USED_FRACTION,
    StorageReservationProject,
    check_step_admission,
    estimate_compact_timeseries_bytes,
    estimate_coordinated_storage_reserve,
    estimate_parent_compact_merge_bytes,
    estimate_parent_render_bytes,
    estimate_project_storage_components,
    estimate_project_storage_reserve,
    estimate_step_forcing_bytes,
    _point_storage_bound,
    _forcing_plot_storage_bound,
    _owned_file_bytes,
    _restart_storage_bound,
    _retained_diagnostics_storage_bound,
    _selected_forcing_source_bytes,
)


def _usage(*, total: int, used: int) -> shutil._ntuple_diskusage:
    return shutil._ntuple_diskusage(total, used, total - used)


def _unmaterialized_project(tmp_path: Path) -> tuple[Path, Path]:
    setup = tmp_path / "setup"
    project = setup / "projects" / "demo"
    meteo = setup / "meteo"
    meteo.mkdir(parents=True)
    project.mkdir(parents=True)
    (meteo / "stations.csv").write_text(
        "id,name,x,y,alt\na,A,0,0,0\n",
        encoding="utf-8",
    )
    (meteo / "a.csv").write_text(
        "date,temp\n"
        "2023-01-01,273\n"
        "2023-01-02,274\n"
        "2023-01-03,275\n"
        "2023-01-04,276\n",
        encoding="utf-8",
    )
    (setup / "north.yml").write_text(
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
        "      - {var: snow.depth, name: snowdepth_daily, freq: 1D}\n",
        encoding="utf-8",
    )
    (project / "demo.yml").write_text(
        "start_date: '2023-01-01'\n"
        "end_date: '2023-01-04'\n"
        "data_assimilation:\n"
        "  prior_forcing: {ensemble_size: 2}\n"
        "  assimilation_events:\n"
        "    - {date: '2023-01-02', variable: station_hs}\n"
        "  output:\n"
        "    retention: compact\n"
        "    grids:\n"
        "      variables:\n"
        "        - {var: snowdepth_daily, metrics: [open_loop, ens_mean]}\n",
        encoding="utf-8",
    )
    return setup, project


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


def test_forcing_estimate_counts_only_exact_selected_rows_conservatively(tmp_path: Path) -> None:
    path = tmp_path / "station.csv"
    outside = "2020-01-01," + ("1" * 2_000) + "\n"
    selected = "2023-01-02," + ("2" * 700) + "\n"
    path.write_text("date,temp\n" + outside + selected + outside, encoding="utf-8")

    estimated = _selected_forcing_source_bytes(
        path,
        start=datetime(2023, 1, 2),
        end=datetime(2023, 1, 2, 23, 59),
    )

    assert estimated == len(b"date,temp\n") + int(len(selected.encode()) * 1.35 + 0.999999)
    assert estimated < path.stat().st_size


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


def test_owned_file_snapshot_deduplicates_hardlinks_and_ignores_symlinks(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.bin"
    hardlink = tmp_path / "hardlink.bin"
    symlink = tmp_path / "symlink.bin"
    original.write_bytes(b"12345")
    os.link(original, hardlink)
    symlink.symlink_to(original)

    assert _owned_file_bytes([original, hardlink, symlink]) == 5


@pytest.mark.parametrize("tolerate_disappearance", [False, True])
def test_owned_file_snapshot_handles_identity_replacement_conservatively(
    monkeypatch,
    tmp_path: Path,
    tolerate_disappearance: bool,
) -> None:
    target = tmp_path / "owned.bin"
    replacement = tmp_path / "replacement.bin"
    target.write_bytes(b"old")
    replacement.write_bytes(b"replacement")
    original_lstat = Path.lstat
    calls = 0

    def replacing_lstat(path: Path, *args, **kwargs):
        nonlocal calls
        if path == target:
            calls += 1
            if calls == 2:
                replacement.replace(target)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", replacing_lstat)

    if tolerate_disappearance:
        assert _owned_file_bytes(
            [target],
            tolerate_disappearance=True,
        ) == 0
    else:
        with pytest.raises(RuntimeError, match="identity changed"):
            _owned_file_bytes([target])


def test_owned_file_snapshot_does_not_credit_new_file_between_passes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "owned.bin"
    newcomer = tmp_path / "new.bin"
    target.write_bytes(b"old")
    original_lstat = Path.lstat
    calls = 0

    def creating_lstat(path: Path, *args, **kwargs):
        nonlocal calls
        if path == target:
            calls += 1
            if calls == 2:
                newcomer.write_bytes(b"a much larger newly materialized file")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", creating_lstat)

    assert _owned_file_bytes([target], tolerate_disappearance=True) == 3
    assert newcomer.stat().st_size > 3


def test_owned_file_snapshot_never_hides_other_metadata_errors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "owned.bin"
    target.write_bytes(b"data")

    def denied_lstat(_path: Path, *_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "lstat", denied_lstat)

    with pytest.raises(PermissionError, match="denied"):
        _owned_file_bytes([target], tolerate_disappearance=True)


def test_forcing_plot_bound_uses_euregio_calibration_and_refits_existing(
    tmp_path: Path,
) -> None:
    setup = tmp_path / "setup"
    (setup / "meteo").mkdir(parents=True)
    (setup / "meteo" / "stations.csv").write_text("id\na\nb\n", encoding="utf-8")
    project = setup / "projects" / "demo"
    steps = [
        (project / "steps" / "step_00", datetime(2023, 1, 1), datetime(2023, 1, 10)),
    ]

    bound = _forcing_plot_storage_bound(
        setup_dir=setup,
        project_dir=project,
        steps=steps,
        member_count=51,
    )

    assert FORCING_PLOT_BYTES_PER_STATION_MEMBER_DAY == 4_400
    assert bound == int(2 * 51 * 10 * 4_400 * 1.25)
    existing = project / "steps" / "step_00" / "plots" / "forcing" / "a.png"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"x" * 10_000)
    resumed = _forcing_plot_storage_bound(
        setup_dir=setup,
        project_dir=project,
        steps=steps,
        member_count=51,
    )
    assert resumed == bound - 10_000


def test_retained_diagnostics_allowance_scales_es30_audit_to_es50(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    per_leaf = _retained_diagnostics_storage_bound(
        project_dir=project,
        member_count=51,
    )

    expected_total = EUREGIO_ES30_RETAINED_DIAGNOSTICS_BYTES * (51 / 31) * 1.25
    assert per_leaf * 90 >= int(expected_total) - 90
    forcing_plot = project / "steps" / "step_00" / "plots" / "forcing" / "large.png"
    forcing_plot.parent.mkdir(parents=True)
    with forcing_plot.open("wb") as stream:
        stream.truncate(per_leaf + 1)
    assert _retained_diagnostics_storage_bound(
        project_dir=project,
        member_count=51,
    ) == per_leaf
    diagnostic = project / "results" / "benchmark" / "large.csv"
    diagnostic.parent.mkdir(parents=True)
    with diagnostic.open("wb") as stream:
        stream.truncate(per_leaf + 1)
    refitted = _retained_diagnostics_storage_bound(
        project_dir=project,
        member_count=51,
    )
    assert refitted >= int(diagnostic.stat().st_size * 0.25)


def test_overwrite_reserves_full_checkpoint_replacement_coexistence(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    existing_checkpoint_bytes = 16 * 1024 * 1024
    for member in ("open_loop", "member_001", "member_002"):
        state = (
            project
            / "steps"
            / "step_00"
            / "ensembles"
            / "prior"
            / member
            / "results"
            / "model_state.pickle.gz"
        )
        state.parent.mkdir(parents=True)
        with state.open("wb") as stream:
            stream.truncate(existing_checkpoint_bytes)

    resumed = _restart_storage_bound(
        project_dir=project,
        grid_cell_count=1_000,
        member_count=3,
        step_count=2,
        compact=True,
        state_pattern="model_state.pickle.gz",
        overwrite=False,
    )
    overwritten = _restart_storage_bound(
        project_dir=project,
        grid_cell_count=1_000,
        member_count=3,
        step_count=2,
        compact=True,
        state_pattern="model_state.pickle.gz",
        overwrite=True,
    )
    checkpoint = int(existing_checkpoint_bytes * 3 * 1.25)

    assert overwritten == (checkpoint, checkpoint)
    assert sum(overwritten) > sum(resumed)


def test_compact_restart_reserve_does_not_credit_cleanup_eligible_predecessor(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    members = ("open_loop", "member_001", "member_002")
    states_by_step: dict[str, list[Path]] = {}
    for index, step_name in enumerate(("step_00", "step_01")):
        step = project / "steps" / step_name
        step.mkdir(parents=True)
        (step / f"{step_name}.yml").write_text(
            f"start_date: 2023-01-0{index + 1}\n"
            f"end_date: 2023-01-0{index + 1}\n",
            encoding="utf-8",
        )
        states_by_step[step_name] = []
        for member in members:
            state = (
                step
                / "ensembles"
                / "prior"
                / member
                / "results"
                / "model_state.pickle.gz"
            )
            state.parent.mkdir(parents=True)
            with state.open("wb") as stream:
                stream.truncate(1024 * 1024)
            states_by_step[step_name].append(state)

    during_cleanup = _restart_storage_bound(
        project_dir=project,
        grid_cell_count=1_000,
        member_count=3,
        step_count=2,
        compact=True,
        state_pattern="model_state.pickle.gz",
    )
    for state in states_by_step["step_00"]:
        state.unlink()
    after_cleanup = _restart_storage_bound(
        project_dir=project,
        grid_cell_count=1_000,
        member_count=3,
        step_count=2,
        compact=True,
        state_pattern="model_state.pickle.gz",
    )

    assert during_cleanup == after_cleanup
    assert after_cleanup[0] > 0
    assert after_cleanup[1] > 0


def test_compact_restart_uses_predecessors_only_as_observed_size_high_water(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    members = ("open_loop", "member_001", "member_002")
    for index, (step_name, state_size) in enumerate(
        (("step_00", 16 * 1024 * 1024), ("step_01", 1024 * 1024))
    ):
        step = project / "steps" / step_name
        step.mkdir(parents=True)
        (step / f"{step_name}.yml").write_text(
            f"start_date: 2023-01-0{index + 1}\n"
            f"end_date: 2023-01-0{index + 1}\n",
            encoding="utf-8",
        )
        for member in members:
            state = (
                step
                / "ensembles"
                / "prior"
                / member
                / "results"
                / "model_state.pickle.gz"
            )
            state.parent.mkdir(parents=True)
            with state.open("wb") as stream:
                stream.truncate(state_size)

    baseline, transition = _restart_storage_bound(
        project_dir=project,
        grid_cell_count=1_000,
        member_count=3,
        step_count=2,
        compact=True,
        state_pattern="model_state.pickle.gz",
    )
    observed_checkpoint = int(16 * 1024 * 1024 * 3 * 1.25)
    newest_existing = 3 * 1024 * 1024

    assert baseline == observed_checkpoint - newest_existing
    assert transition == observed_checkpoint


def _storage_race_project(
    tmp_path: Path,
    *,
    retention: str,
) -> tuple[Path, Path, dict[str, Path]]:
    setup, project = _unmaterialized_project(tmp_path)
    project_yaml = project / "demo.yml"
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8").replace(
            "retention: compact",
            f"retention: {retention}",
        ),
        encoding="utf-8",
    )
    create_project_skeleton(setup, project)
    step = sorted((project / "steps").glob("step_*"))[0]
    member = step / "ensembles" / "prior" / "member_001"
    artifacts = {
        "forcing": member / "meteo" / "a.csv",
        "point": member / "results" / "point_a.csv",
        "grid": member / "results" / "output_grids.nc",
        "forcing_plot": step / "plots" / "forcing" / "member_001.png",
        "state": member / "results" / "model_state.pickle.gz",
    }
    for artifact_class, path in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            b"date,value\n2023-01-01,1\n"
            if artifact_class == "point"
            else artifact_class.encode("ascii")
        )
        path.write_bytes(payload)
    return setup, project, artifacts


@pytest.mark.parametrize(
    "artifact_class",
    ["forcing", "point", "grid", "forcing_plot", "state"],
)
@pytest.mark.parametrize("disappearance", ["before_stat", "after_stat"])
def test_compact_estimate_tolerates_cleanup_race_for_owned_artifact_classes(
    monkeypatch,
    tmp_path: Path,
    artifact_class: str,
    disappearance: str,
) -> None:
    setup, project, artifacts = _storage_race_project(
        tmp_path,
        retention="compact",
    )
    target = artifacts[artifact_class]
    stable_estimate = estimate_project_storage_components(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=4,
    )
    original_lstat = Path.lstat
    deleted = False

    def racing_lstat(path: Path, *args, **kwargs):
        nonlocal deleted
        if path == target and not deleted:
            deleted = True
            if disappearance == "before_stat":
                target.unlink()
                raise FileNotFoundError(target)
            metadata = original_lstat(path, *args, **kwargs)
            target.unlink()
            return metadata
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", racing_lstat)

    estimate = estimate_project_storage_components(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=4,
    )

    assert deleted
    assert estimate.total_bytes > 0
    assert estimate.total_bytes >= stable_estimate.total_bytes
    assert all(
        value >= 0
        for value in (
            estimate.forcing_bytes,
            estimate.member_grid_bytes,
            estimate.point_bytes,
            estimate.restart_baseline_bytes,
            estimate.restart_transition_bytes,
            estimate.derived_forcing_plot_bytes,
        )
    )


@pytest.mark.parametrize(
    "artifact_class",
    ["forcing", "point", "grid", "forcing_plot", "state"],
)
@pytest.mark.parametrize("disappearance", ["before_stat", "after_stat"])
def test_full_retention_estimate_rejects_disappearing_owned_artifacts(
    monkeypatch,
    tmp_path: Path,
    artifact_class: str,
    disappearance: str,
) -> None:
    setup, project, artifacts = _storage_race_project(
        tmp_path,
        retention="full",
    )
    target = artifacts[artifact_class]
    original_lstat = Path.lstat
    deleted = False

    def racing_lstat(path: Path, *args, **kwargs):
        nonlocal deleted
        if path == target and not deleted:
            deleted = True
            if disappearance == "before_stat":
                target.unlink()
                raise FileNotFoundError(target)
            metadata = original_lstat(path, *args, **kwargs)
            target.unlink()
            return metadata
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", racing_lstat)

    with pytest.raises(FileNotFoundError):
        estimate_project_storage_components(
            setup_dir=setup,
            project_dir=project,
            grid_cell_count=4,
        )
    assert deleted


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


def test_point_bound_counts_layered_defaults_and_refits_only_upward(tmp_path: Path) -> None:
    setup = tmp_path / "setup"
    (setup / "meteo").mkdir(parents=True)
    (setup / "meteo" / "stations.csv").write_text("id\na\n", encoding="utf-8")
    steps = [(tmp_path / "step", datetime(2023, 1, 1), datetime(2023, 1, 2))]
    base = _point_storage_bound(
        setup_dir=setup,
        setup_cfg={"snow": {"min_thickness": [0.1, 0.2, 0.4]}},
        output_data={"timeseries": {"add_default_points": True, "add_default_variables": True}},
        steps=steps,
        model_timestep="1D",
        member_count=1,
    )
    project = tmp_path / "project"
    observed = project / "steps" / "step_00" / "ensembles" / "prior" / "open_loop" / "results" / "point_a.csv"
    observed.parent.mkdir(parents=True)
    observed.write_text("date,swe\n2023-01-01," + ("9" * 10_000) + "\n", encoding="utf-8")
    refit = _point_storage_bound(
        setup_dir=setup,
        setup_cfg={"snow": {"min_thickness": [0.1, 0.2, 0.4]}},
        output_data={"timeseries": {"add_default_points": True, "add_default_variables": True}},
        steps=steps,
        model_timestep="1D",
        member_count=1,
        project_dir=project,
    )

    assert refit > base


def test_euregio_point_bound_uses_actual_filtered_leaf_inventory(tmp_path: Path) -> None:
    """Regression calibration from the 90-leaf prepared Euregio audit."""
    setup = tmp_path / "filtered"
    meteo = setup / "meteo"
    meteo.mkdir(parents=True)
    station_rows = ["id", *(f"station_{index}" for index in range(4555))]
    (meteo / "stations.csv").write_text("\n".join(station_rows) + "\n", encoding="utf-8")
    steps = [(tmp_path / "step", datetime(2023, 1, 1), datetime(2023, 1, 1))]
    output_data = {
        "timeseries": {
            "add_default_points": True,
            "add_default_variables": True,
            "points": [{"name": f"snow_{index}"} for index in range(78)],
        }
    }
    filtered = _point_storage_bound(
        setup_dir=setup,
        setup_cfg={},
        output_data=output_data,
        steps=steps,
        model_timestep="1D",
        member_count=1,
    )

    one = tmp_path / "one"
    (one / "meteo").mkdir(parents=True)
    (one / "meteo" / "stations.csv").write_text("id\nstation\n", encoding="utf-8")
    per_point = _point_storage_bound(
        setup_dir=one,
        setup_cfg={},
        output_data={
            "timeseries": {
                "add_default_points": True,
                "add_default_variables": True,
            }
        },
        steps=steps,
        model_timestep="1D",
        member_count=1,
    )

    assert filtered == per_point * (4555 + 78)


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


def test_project_estimate_plans_pristine_unmaterialized_steps_without_writes(
    tmp_path: Path,
) -> None:
    setup, project = _unmaterialized_project(tmp_path)

    planned = estimate_project_storage_components(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=4,
    )

    assert planned.total_bytes > 0
    assert not (project / "steps").exists()

    create_project_skeleton(setup, project)
    materialized = estimate_project_storage_components(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=4,
    )
    assert materialized == planned


@pytest.mark.parametrize("status", ["running", "failed", "paused_low_disk", "success"])
def test_project_estimate_rejects_missing_steps_after_leaf_started(
    tmp_path: Path,
    status: str,
) -> None:
    setup, project = _unmaterialized_project(tmp_path)
    (setup / "run_manifest.json").write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing after the leaf project started"):
        estimate_project_storage_components(
            setup_dir=setup,
            project_dir=project,
            grid_cell_count=4,
        )
    assert not (project / "steps").exists()


def test_project_estimate_rejects_existing_empty_steps_directory(
    tmp_path: Path,
) -> None:
    setup, project = _unmaterialized_project(tmp_path)
    (project / "steps").mkdir()

    with pytest.raises(FileNotFoundError, match="steps directory is empty or invalid"):
        estimate_project_storage_components(
            setup_dir=setup,
            project_dir=project,
            grid_cell_count=4,
        )


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
        parent_finalization_reserve_bytes=100,
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
        "start_date: 2023-01-01\n"
        "end_date: 2023-01-04\n"
        "data_assimilation:\n"
        "  prior_forcing: {ensemble_size: 2}\n"
        "  output:\n"
        "    grids:\n"
        "      variables:\n"
        "        - {var: snowdepth_daily, metrics: [open_loop, ens_mean]}\n",
        encoding="utf-8",
    )

    reserve = estimate_parent_compact_merge_bytes(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=100,
    )
    assert reserve > 100 * 2 * 4 * 8


def test_parent_render_reserve_refits_existing_outputs(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    project.mkdir(parents=True)
    (project / "demo.yml").write_text(
        "data_assimilation:\n"
        "  assimilation_events:\n"
        "    - {date: 2023-01-01, variable: scf, product: test}\n"
        "    - {date: 2023-01-07, variable: station_hs, product: test}\n",
        encoding="utf-8",
    )

    first = estimate_parent_render_bytes(project_dir=project, grid_cell_count=100)
    assert first >= 512 * 1024**2

    report = project / "results" / "reports" / "project_report.pdf"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"x" * 100)
    resumed = estimate_parent_render_bytes(project_dir=project, grid_cell_count=100)
    overwritten = estimate_parent_render_bytes(
        project_dir=project,
        grid_cell_count=100,
        overwrite=True,
    )
    assert resumed == first - 100
    assert overwritten == first


def test_map_support_reserve_is_explicit_and_stage_aware(tmp_path: Path) -> None:
    setup = tmp_path / "setup"
    project = setup / "projects" / "demo"
    (setup / "meteo").mkdir(parents=True)
    (setup / "meteo" / "stations.csv").write_text("id\n", encoding="utf-8")
    (setup / "meteo" / "a.csv").write_text("date,temp\n2023-01-01,273\n", encoding="utf-8")
    (setup / "north.yml").write_text(
        "timestep: 1D\noutput_data:\n  timeseries: {add_default_points: false, add_default_variables: false}\n"
        "  grids:\n    variables: [{var: snow.depth, name: snowdepth_daily, freq: 1D}]\n",
        encoding="utf-8",
    )
    project.mkdir(parents=True)
    (project / "demo.yml").write_text(
        "data_assimilation:\n"
        "  prior_forcing: {ensemble_size: 1}\n"
        "  assimilation_events:\n"
        "    - {date: '2023-01-01', variable: scf, product: TEST}\n"
        "  output:\n"
        "    retention: compact\n"
        "    grids:\n"
        "      variables: [{var: snowdepth_daily, metrics: [open_loop]}]\n",
        encoding="utf-8",
    )
    step = project / "steps" / "step_00"
    step.mkdir(parents=True)
    (step / "step.yml").write_text(
        "start_date: '2023-01-01'\nend_date: '2023-01-01'\n", encoding="utf-8"
    )
    pending = estimate_project_storage_components(
        setup_dir=setup, project_dir=project, grid_cell_count=100
    )
    assert pending.map_support_bytes > 3 * 100 * 4
    support = project / "results" / "grids" / "da_map_support.nc"
    support.parent.mkdir(parents=True)
    support.write_bytes(b"accepted")
    accepted = estimate_project_storage_components(
        setup_dir=setup, project_dir=project, grid_cell_count=100
    )
    overwritten = estimate_project_storage_components(
        setup_dir=setup, project_dir=project, grid_cell_count=100, overwrite=True
    )
    assert accepted.map_support_bytes == 0
    assert overwritten.map_support_bytes == pending.map_support_bytes
