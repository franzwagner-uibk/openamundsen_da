from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from openamundsen_da.benchmark.pipeline import core as pipeline_mod


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")


def _setup_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    step_dir = project_dir / "steps" / "step_00_init"

    _write_yaml(
        setup_dir / "setup.yml",
        """
        resolution: 100
        """,
    )
    _write_yaml(
        project_dir / "project_2022_2023.yml",
        """
        start_date: '2023-01-01'
        end_date: '2023-01-02'
        data_assimilation:
          wet_snow:
            classification_threshold_percent: 12.5
        """,
    )
    _write_yaml(
        step_dir / "step_00.yml",
        """
        start_date: '2023-01-01 00:00:00'
        end_date: '2023-01-02 23:00:00'
        """,
    )
    return setup_dir, project_dir, step_dir


def _write_complete_fraction_outputs(step_dir: Path, filename: str) -> None:
    base = step_dir / "ensembles" / "prior"
    _touch(base / "open_loop" / "results" / filename)
    _touch(base / "member_001" / "results" / filename)
    _touch(base / "member_002" / "results" / filename)


def test_benchmark_prerequisites_reuse_existing_outputs_during_project_run(tmp_path, monkeypatch) -> None:
    setup_dir, project_dir, step_dir = _setup_project(tmp_path)
    _write_complete_fraction_outputs(step_dir, "point_scf_roi.csv")
    _write_complete_fraction_outputs(step_dir, "point_wet_snow_roi.csv")

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(pipeline_mod, "ensure_setup_roi_vector", lambda _setup_dir: _setup_dir / "env" / "roi.gpkg")
    monkeypatch.setattr(pipeline_mod, "resolve_landcover_mask", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        pipeline_mod,
        "compute_step_scf_daily_for_all_members",
        lambda **kwargs: calls.append(("scf", bool(kwargs["overwrite"]))),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "classify_step_wet_snow",
        lambda **kwargs: calls.append(("wet_classify", bool(kwargs["overwrite"]))),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "compute_step_wet_snow_daily_for_all_members",
        lambda **kwargs: calls.append(("wet_daily", bool(kwargs["overwrite"]))),
    )

    pipeline_mod.ensure_benchmark_prerequisites(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("scf", "wet_snow"),
        overwrite=True,
        reuse_existing_prerequisites=True,
    )

    assert calls == []


def test_benchmark_prerequisites_backfill_missing_outputs_without_overwrite(tmp_path, monkeypatch) -> None:
    setup_dir, project_dir, step_dir = _setup_project(tmp_path)
    _write_complete_fraction_outputs(step_dir, "point_scf_roi.csv")
    _touch(step_dir / "ensembles" / "prior" / "open_loop" / "results" / "point_wet_snow_roi.csv")
    _touch(step_dir / "ensembles" / "prior" / "member_001" / "results" / "point_wet_snow_roi.csv")

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(pipeline_mod, "ensure_setup_roi_vector", lambda _setup_dir: _setup_dir / "env" / "roi.gpkg")
    monkeypatch.setattr(pipeline_mod, "resolve_landcover_mask", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        pipeline_mod,
        "compute_step_scf_daily_for_all_members",
        lambda **kwargs: calls.append(("scf", bool(kwargs["overwrite"]))),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "classify_step_wet_snow",
        lambda **kwargs: calls.append(("wet_classify", bool(kwargs["overwrite"]))),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "compute_step_wet_snow_daily_for_all_members",
        lambda **kwargs: calls.append(("wet_daily", bool(kwargs["overwrite"]))),
    )

    pipeline_mod.ensure_benchmark_prerequisites(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("scf", "wet_snow"),
        overwrite=True,
        reuse_existing_prerequisites=True,
    )

    assert calls == [("wet_classify", False), ("wet_daily", False)]


def test_benchmark_prerequisites_overwrite_still_forces_recompute(tmp_path, monkeypatch) -> None:
    setup_dir, project_dir, step_dir = _setup_project(tmp_path)
    _write_complete_fraction_outputs(step_dir, "point_scf_roi.csv")
    _write_complete_fraction_outputs(step_dir, "point_wet_snow_roi.csv")

    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(pipeline_mod, "ensure_setup_roi_vector", lambda _setup_dir: _setup_dir / "env" / "roi.gpkg")
    monkeypatch.setattr(pipeline_mod, "resolve_landcover_mask", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        pipeline_mod,
        "compute_step_scf_daily_for_all_members",
        lambda **kwargs: calls.append(("scf", bool(kwargs["overwrite"]))),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "classify_step_wet_snow",
        lambda **kwargs: calls.append(("wet_classify", bool(kwargs["overwrite"]))),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "compute_step_wet_snow_daily_for_all_members",
        lambda **kwargs: calls.append(("wet_daily", bool(kwargs["overwrite"]))),
    )

    pipeline_mod.ensure_benchmark_prerequisites(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("scf", "wet_snow"),
        overwrite=True,
        reuse_existing_prerequisites=False,
    )

    assert calls == [("scf", True), ("wet_classify", True), ("wet_daily", True)]


def test_load_benchmark_config_accepts_performance_score_exclusions(tmp_path: Path) -> None:
    setup_dir, project_dir, _step_dir = _setup_project(tmp_path)
    _write_yaml(
        project_dir / "project_2022_2023.yml",
        """
        start_date: '2023-01-01'
        end_date: '2023-01-02'
        data_assimilation:
          wet_snow:
            classification_threshold_percent: 12.5
          benchmark:
            independent_variables: [station_swe]
            performance_scores_exclude_variables: [station_swe, wet_snow_fraction]
            score_station_sigma_threshold: 200
        """,
    )

    cfg = pipeline_mod.load_benchmark_config(project_dir)

    assert cfg.output_dir == project_dir / "results" / "benchmark"
    assert cfg.independent_variables == ("station_swe",)
    assert cfg.performance_scores_exclude_variables == ("station_swe", "wet_snow")
    assert cfg.score_station_sigma_threshold == pytest.approx(200.0)


def test_load_benchmark_config_rejects_invalid_performance_score_exclusion(tmp_path: Path) -> None:
    _setup_project(tmp_path)
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_yaml(
        project_dir / "project_2022_2023.yml",
        """
        start_date: '2023-01-01'
        end_date: '2023-01-02'
        data_assimilation:
          wet_snow:
            classification_threshold_percent: 12.5
          benchmark:
            performance_scores_exclude_variables: [bogus_variable]
        """,
    )

    with pytest.raises(ValueError, match="bogus_variable"):
        pipeline_mod.load_benchmark_config(project_dir)


def test_load_benchmark_config_rejects_invalid_station_score_sigma_threshold(tmp_path: Path) -> None:
    _setup_project(tmp_path)
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_yaml(
        project_dir / "project_2022_2023.yml",
        """
        start_date: '2023-01-01'
        end_date: '2023-01-02'
        data_assimilation:
          wet_snow:
            classification_threshold_percent: 12.5
          benchmark:
            score_station_sigma_threshold: nope
        """,
    )

    with pytest.raises(ValueError, match="score_station_sigma_threshold"):
        pipeline_mod.load_benchmark_config(project_dir)


def test_load_benchmark_config_rejects_non_positive_station_score_sigma_threshold(tmp_path: Path) -> None:
    _setup_project(tmp_path)
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_yaml(
        project_dir / "project_2022_2023.yml",
        """
        start_date: '2023-01-01'
        end_date: '2023-01-02'
        data_assimilation:
          wet_snow:
            classification_threshold_percent: 12.5
          benchmark:
            score_station_sigma_threshold: 0
        """,
    )

    with pytest.raises(ValueError, match="score_station_sigma_threshold"):
        pipeline_mod.load_benchmark_config(project_dir)
