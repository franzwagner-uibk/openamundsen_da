from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from openamundsen_da.benchmark.cases import analysis_event_contexts, event_dates_by_variable, extract_analysis_cases, extract_continuous_cases
from openamundsen_da.benchmark.metrics import build_case_scores
from openamundsen_da.observer.summary_paths import record_fraction_summary_path


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def _write_series_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _setup_basic_project(tmp_path: Path, *, events_yaml: str) -> tuple[Path, Path]:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    events_block = events_yaml.strip("\n")

    _write_yaml(
        setup_dir / "setup.yml",
        """
        resolution: 100
        """,
    )
    _write_yaml(
        project_dir / "project_2022_2023.yml",
        f"""
        start_date: '2023-01-01'
        end_date: '2023-01-04'
        obs:
          stations:
            dir: obs/stations
          snowcover:
            dir: obs/snowcover
            product_tag: SNOWCOVER
          wetsnow:
            dir: obs/wetsnow
            product_tag: WETSNOW
        data_assimilation:
          station:
            default_station_uncertainty_pct: 25
            min_station_uncertainty_pct: 10
            single_station_factor: 2.0
          assimilation_events:
{events_block}
        """,
    )
    _write_yaml(
        project_dir / "steps" / "step_00_init" / "step_00.yml",
        """
        start_date: '2023-01-01 00:00:00'
        end_date: '2023-01-02 23:00:00'
        """,
    )
    _write_yaml(
        project_dir / "steps" / "step_01_next" / "step_01.yml",
        """
        start_date: '2023-01-03 00:00:00'
        end_date: '2023-01-04 23:00:00'
        """,
    )
    return setup_dir, project_dir


def _write_fraction_benchmark_inputs(setup_dir: Path, project_dir: Path) -> None:
    summaries_dir = setup_dir / "obs" / "summaries" / project_dir.name
    _write_series_csv(
        summaries_dir / "scf_summary.csv",
        [
            {"date": "2023-01-02", "scf": 0.30},
            {"date": "2023-01-03", "scf": 0.60},
        ],
    )
    _write_series_csv(
        summaries_dir / "wet_snow_summary.csv",
        [
            {"date": "2023-01-03", "wet_snow_fraction": 0.40},
        ],
    )
    _write_series_csv(
        summaries_dir / "wet_snow_line_diagnostics.csv",
        [
            {"date": "2023-01-03", "wet_snow_line": 2400.0},
        ],
    )

    for step_name, day in (("step_00_init", "2023-01-02"), ("step_01_next", "2023-01-03")):
        base = project_dir / "steps" / step_name / "ensembles" / "prior"
        _write_series_csv(
            base / "open_loop" / "results" / "point_scf_roi.csv",
            [{"time": day, "scf": 0.2 if day.endswith("02") else 0.5}],
        )
        _write_series_csv(
            base / "member_001" / "results" / "point_scf_roi.csv",
            [{"time": day, "scf": 0.25 if day.endswith("02") else 0.55}],
        )
        _write_series_csv(
            base / "member_002" / "results" / "point_scf_roi.csv",
            [{"time": day, "scf": 0.35 if day.endswith("02") else 0.75}],
        )

        _write_series_csv(
            base / "open_loop" / "results" / "point_wet_snow_roi.csv",
            [{"time": day, "wet_snow_fraction": 0.1 if day.endswith("02") else 0.2}],
        )
        _write_series_csv(
            base / "member_001" / "results" / "point_wet_snow_roi.csv",
            [{"time": day, "wet_snow_fraction": 0.15 if day.endswith("02") else 0.25}],
        )
        _write_series_csv(
            base / "member_002" / "results" / "point_wet_snow_roi.csv",
            [{"time": day, "wet_snow_fraction": 0.05 if day.endswith("02") else 0.35}],
        )

        _write_series_csv(
            base / "open_loop" / "results" / "point_wet_snow_line_roi.csv",
            [{"time": day, "wet_snow_line": 2300.0 if day.endswith("02") else 2350.0}],
        )
        _write_series_csv(
            base / "member_001" / "results" / "point_wet_snow_line_roi.csv",
            [{"time": day, "wet_snow_line": 2380.0 if day.endswith("02") else 2390.0}],
        )
        _write_series_csv(
            base / "member_002" / "results" / "point_wet_snow_line_roi.csv",
            [{"time": day, "wet_snow_line": 2420.0 if day.endswith("02") else 2410.0}],
        )


def _write_station_benchmark_inputs(project_dir: Path, setup_dir: Path) -> None:
    stations_dir = setup_dir / "obs" / "stations"
    _write_series_csv(
        stations_dir / "stations_da_metadata.csv",
        [
            {
                "station_id": "station_a",
                "station_uncertainty_pct": 12.0,
                "hs_sigma_abs_min": 0.15,
                "swe_sigma_abs_min": 8.0,
            },
            {
                "station_id": "station_b",
                "station_uncertainty_pct": 20.0,
                "hs_sigma_abs_min": 0.20,
                "swe_sigma_abs_min": 10.0,
            },
        ],
    )
    _write_series_csv(
        stations_dir / "station_a.csv",
        [
            {"time": "2023-01-02 00:00:00", "snow_depth": 1.1, "swe": None},
            {"time": "2023-01-03 00:00:00", "snow_depth": 1.4, "swe": None},
        ],
    )
    _write_series_csv(
        stations_dir / "station_b.csv",
        [
            {"time": "2023-01-02 00:00:00", "snow_depth": None, "swe": 10.0},
            {"time": "2023-01-03 00:00:00", "snow_depth": None, "swe": 12.0},
        ],
    )

    for step_name, day in (("step_00_init", "2023-01-02 00:00:00"), ("step_01_next", "2023-01-03 00:00:00")):
        base = project_dir / "steps" / step_name / "ensembles" / "prior"
        _write_series_csv(
            base / "open_loop" / "results" / "point_station_a.csv",
            [{"time": day, "snow_depth": 0.8 if day.startswith("2023-01-02") else 0.9, "swe": None}],
        )
        _write_series_csv(
            base / "member_001" / "results" / "point_station_a.csv",
            [{"time": day, "snow_depth": 1.0 if day.startswith("2023-01-02") else 1.1, "swe": None}],
        )
        _write_series_csv(
            base / "member_002" / "results" / "point_station_a.csv",
            [{"time": day, "snow_depth": 1.2 if day.startswith("2023-01-02") else 1.3, "swe": None}],
        )

        _write_series_csv(
            base / "open_loop" / "results" / "point_station_b.csv",
            [{"time": day, "snow_depth": None, "swe": 9.0 if day.startswith("2023-01-02") else 10.0}],
        )
        _write_series_csv(
            base / "member_001" / "results" / "point_station_b.csv",
            [{"time": day, "snow_depth": None, "swe": 10.0 if day.startswith("2023-01-02") else 11.0}],
        )
        _write_series_csv(
            base / "member_002" / "results" / "point_station_b.csv",
            [{"time": day, "snow_depth": None, "swe": 12.0 if day.startswith("2023-01-02") else 13.0}],
        )


def test_extract_continuous_cases_supports_all_benchmark_families(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-02'
              variable: scf
              product: SNOWCOVER
            - date: '2023-01-02'
              variable: station_hs
        """,
    )
    _write_fraction_benchmark_inputs(setup_dir, project_dir)
    _write_station_benchmark_inputs(project_dir, setup_dir)

    cases = extract_continuous_cases(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("scf", "wet_snow", "wet_snow_line", "station_hs", "station_swe"),
    )

    assert {case.variable for case in cases} == {"scf", "wet_snow", "wet_snow_line", "station_hs", "station_swe"}
    by_key = {(case.variable, str(case.timestamp), case.obs_id): case for case in cases}
    assert by_key[("scf", "2023-01-02 00:00:00", "roi")].stream == "assimilation_fit"
    assert by_key[("scf", "2023-01-03 00:00:00", "roi")].stream == "semi_independent"
    assert by_key[("wet_snow", "2023-01-03 00:00:00", "roi")].stream == "independent"
    assert by_key[("wet_snow_line", "2023-01-03 00:00:00", "roi")].stream == "independent"
    assert by_key[("wet_snow_line", "2023-01-03 00:00:00", "roi")].obs_value == 2400.0
    assert by_key[("station_hs", "2023-01-02 00:00:00", "station_a")].stream == "assimilation_fit"
    assert by_key[("station_hs", "2023-01-03 00:00:00", "station_a")].stream == "semi_independent"
    assert by_key[("station_swe", "2023-01-02 00:00:00", "station_b")].stream == "semi_independent"
    assert by_key[("station_swe", "2023-01-03 00:00:00", "station_b")].stream == "semi_independent"
    assert by_key[("station_hs", "2023-01-02 00:00:00", "station_a")].sigma_base == pytest.approx(
        ((1.1 * 0.12) ** 2 + 0.15 ** 2) ** 0.5
    )
    assert by_key[("station_swe", "2023-01-03 00:00:00", "station_b")].sigma_base == pytest.approx(
        ((12.0 * 0.20) ** 2 + 10.0 ** 2) ** 0.5
    )


def test_extract_continuous_cases_respects_station_benchmark_role(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-02'
              variable: station_hs
        """,
    )
    stations_dir = setup_dir / "obs" / "stations"
    _write_series_csv(
        stations_dir / "stations_da_metadata.csv",
        [
            {
                "station_id": "station_a",
                "station_uncertainty_pct": 12.0,
                "hs_sigma_abs_min": 0.15,
                "use_for_benchmark": True,
            },
            {
                "station_id": "station_b",
                "station_uncertainty_pct": 12.0,
                "hs_sigma_abs_min": 0.15,
                "use_for_benchmark": False,
            },
        ],
    )
    for station_id in ("station_a", "station_b"):
        _write_series_csv(
            stations_dir / f"{station_id}.csv",
            [{"time": "2023-01-02 00:00:00", "snow_depth": 1.0}],
        )
        base = project_dir / "steps" / "step_00_init" / "ensembles" / "prior"
        for member_id, value in (("open_loop", 0.9), ("member_001", 1.0), ("member_002", 1.1)):
            _write_series_csv(
                base / member_id / "results" / f"point_{station_id}.csv",
                [{"time": "2023-01-02 00:00:00", "snow_depth": value}],
            )

    cases = extract_continuous_cases(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("station_hs",),
    )

    assert {case.obs_id for case in cases} == {"station_a"}


def test_extract_analysis_cases_uses_weighted_station_posterior(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-02'
              variable: station_hs
        """,
    )
    _write_station_benchmark_inputs(project_dir, setup_dir)
    weights_path = project_dir / "steps" / "step_00_init" / "assim" / "weights_station_hs_20230102.csv"
    _write_series_csv(
        weights_path,
        [
            {"member_id": "member_001", "weight": 0.25},
            {"member_id": "member_002", "weight": 0.75},
        ],
    )

    cases = extract_analysis_cases(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("station_hs",),
    )

    assert len(cases) == 1
    case = cases[0]
    assert case.variable == "station_hs"
    assert case.stream == "assimilation_fit"
    assert case.obs_id == "station_a"
    assert case.timestamp == pd.Timestamp("2023-01-02 00:00:00")
    assert case.prior_values == (1.0, 1.2)
    assert case.posterior_values == (1.0, 1.2)
    assert case.posterior_weights == (0.25, 0.75)
    assert case.sigma_base == pytest.approx(((1.1 * 0.12) ** 2 + 0.15 ** 2) ** 0.5)

    case_scores = build_case_scores(cases)
    posterior = case_scores.loc[case_scores["representation"] == "posterior"].iloc[0]
    assert float(posterior["pred_mean"]) == 1.15


def test_extract_analysis_cases_include_transfer_streams_on_da_dates(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-02'
              variable: station_hs
            - date: '2023-01-03'
              variable: scf
              product: SNOWCOVER
        """,
    )
    _write_fraction_benchmark_inputs(setup_dir, project_dir)
    _write_station_benchmark_inputs(project_dir, setup_dir)
    _write_series_csv(
        project_dir / "steps" / "step_00_init" / "assim" / "weights_station_hs_20230102.csv",
        [
            {"member_id": "member_001", "weight": 0.25},
            {"member_id": "member_002", "weight": 0.75},
        ],
    )
    _write_series_csv(
        project_dir / "steps" / "step_01_next" / "assim" / "weights_scf_20230103.csv",
        [
            {"member_id": "member_001", "weight": 0.6},
            {"member_id": "member_002", "weight": 0.4},
        ],
    )

    cases = extract_analysis_cases(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("scf", "wet_snow", "wet_snow_line", "station_hs", "station_swe"),
    )

    case_lookup = {(case.variable, case.stream, str(case.timestamp), case.obs_id): case for case in cases}
    assert ("station_hs", "assimilation_fit", "2023-01-02 00:00:00", "station_a") in case_lookup
    assert ("station_swe", "semi_independent", "2023-01-02 00:00:00", "station_b") in case_lookup
    assert ("scf", "independent", "2023-01-02 00:00:00", "roi") in case_lookup
    assert ("station_hs", "semi_independent", "2023-01-03 00:00:00", "station_a") in case_lookup
    assert ("scf", "assimilation_fit", "2023-01-03 00:00:00", "roi") in case_lookup
    assert ("wet_snow", "independent", "2023-01-03 00:00:00", "roi") in case_lookup
    wsl_case = case_lookup[("wet_snow_line", "independent", "2023-01-03 00:00:00", "roi")]
    assert wsl_case.posterior_values == (2390.0, 2410.0)
    assert wsl_case.posterior_weights == (0.6, 0.4)


def test_extract_analysis_cases_skips_wet_snow_line_transfer_with_missing_weighted_member(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-03'
              variable: station_hs
        """,
    )
    _write_fraction_benchmark_inputs(setup_dir, project_dir)
    _write_station_benchmark_inputs(project_dir, setup_dir)
    _write_series_csv(
        project_dir / "steps" / "step_01_next" / "ensembles" / "prior" / "member_002" / "results" / "point_wet_snow_line_roi.csv",
        [{"time": "2023-01-03", "wet_snow_line": None}],
    )
    _write_series_csv(
        project_dir / "steps" / "step_00_init" / "assim" / "weights_station_hs_20230103.csv",
        [
            {"member_id": "member_001", "weight": 0.25},
            {"member_id": "member_002", "weight": 0.75},
        ],
    )

    cases = extract_analysis_cases(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("wet_snow_line", "station_hs"),
    )

    case_lookup = {(case.variable, case.stream, str(case.timestamp), case.obs_id): case for case in cases}
    assert ("wet_snow_line", "independent", "2023-01-03 00:00:00", "roi") not in case_lookup
    assert ("station_hs", "assimilation_fit", "2023-01-03 00:00:00", "station_a") in case_lookup


def test_extract_analysis_cases_skip_transfer_rows_without_same_day_observation(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-02'
              variable: station_hs
        """,
    )
    _write_fraction_benchmark_inputs(setup_dir, project_dir)
    _write_station_benchmark_inputs(project_dir, setup_dir)
    _write_series_csv(
        project_dir / "steps" / "step_00_init" / "assim" / "weights_station_hs_20230102.csv",
        [
            {"member_id": "member_001", "weight": 0.25},
            {"member_id": "member_002", "weight": 0.75},
        ],
    )

    cases = extract_analysis_cases(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("wet_snow",),
    )

    assert cases == []


def test_extract_continuous_station_cases_supports_mixed_model_timestamps(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-02'
              variable: station_hs
        """,
    )
    _write_station_benchmark_inputs(project_dir, setup_dir)

    for base in (
        project_dir / "steps" / "step_00_init" / "ensembles" / "prior",
        project_dir / "steps" / "step_01_next" / "ensembles" / "prior",
    ):
        mixed_rows = [
            {"time": "2023-01-02", "snow_depth": 0.8, "swe": None},
            {"time": "2023-01-02 03:00:00", "snow_depth": 0.85, "swe": None},
        ]
        if base.name == "prior" and base.parent.parent.name == "step_01_next":
            mixed_rows = [
                {"time": "2023-01-03", "snow_depth": 0.9, "swe": None},
                {"time": "2023-01-03 03:00:00", "snow_depth": 0.95, "swe": None},
            ]
        _write_series_csv(base / "open_loop" / "results" / "point_station_a.csv", mixed_rows)
        _write_series_csv(
            base / "member_001" / "results" / "point_station_a.csv",
            [{**row, "snow_depth": float(row["snow_depth"]) + 0.2, "swe": None} for row in mixed_rows],
        )
        _write_series_csv(
            base / "member_002" / "results" / "point_station_a.csv",
            [{**row, "snow_depth": float(row["snow_depth"]) + 0.4, "swe": None} for row in mixed_rows],
        )

    cases = extract_continuous_cases(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("station_hs",),
    )

    station_cases = [case for case in cases if case.variable == "station_hs" and case.obs_id == "station_a"]
    assert [case.timestamp for case in station_cases] == list(
        pd.to_datetime(["2023-01-02 00:00:00", "2023-01-03 00:00:00"])
    )


def test_extract_continuous_cases_keep_future_same_variable_events_independent(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-03'
              variable: scf
              product: SNOWCOVER
        """,
    )
    _write_fraction_benchmark_inputs(setup_dir, project_dir)

    _write_series_csv(
        setup_dir / "obs" / "summaries" / project_dir.name / "scf_summary.csv",
        [
            {"date": "2023-01-02", "scf": 0.30},
            {"date": "2023-01-03", "scf": 0.60},
            {"date": "2023-01-04", "scf": 0.65},
        ],
    )
    _write_series_csv(
        project_dir / "steps" / "step_00_init" / "ensembles" / "prior" / "open_loop" / "results" / "point_scf_roi.csv",
        [{"time": "2023-01-02", "scf": 0.2}],
    )
    _write_series_csv(
        project_dir / "steps" / "step_00_init" / "ensembles" / "prior" / "member_001" / "results" / "point_scf_roi.csv",
        [{"time": "2023-01-02", "scf": 0.25}],
    )
    _write_series_csv(
        project_dir / "steps" / "step_00_init" / "ensembles" / "prior" / "member_002" / "results" / "point_scf_roi.csv",
        [{"time": "2023-01-02", "scf": 0.35}],
    )
    _write_series_csv(
        project_dir / "steps" / "step_01_next" / "ensembles" / "prior" / "open_loop" / "results" / "point_scf_roi.csv",
        [
            {"time": "2023-01-03", "scf": 0.5},
            {"time": "2023-01-04", "scf": 0.58},
        ],
    )
    _write_series_csv(
        project_dir / "steps" / "step_01_next" / "ensembles" / "prior" / "member_001" / "results" / "point_scf_roi.csv",
        [
            {"time": "2023-01-03", "scf": 0.55},
            {"time": "2023-01-04", "scf": 0.60},
        ],
    )
    _write_series_csv(
        project_dir / "steps" / "step_01_next" / "ensembles" / "prior" / "member_002" / "results" / "point_scf_roi.csv",
        [
            {"time": "2023-01-03", "scf": 0.75},
            {"time": "2023-01-04", "scf": 0.70},
        ],
    )

    cases = extract_continuous_cases(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("scf",),
    )

    by_time = {str(case.timestamp): case.stream for case in cases}
    assert by_time["2023-01-02 00:00:00"] == "independent"
    assert by_time["2023-01-03 00:00:00"] == "assimilation_fit"
    assert by_time["2023-01-04 00:00:00"] == "semi_independent"


def test_extract_continuous_cases_uses_project_configured_summary_path(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-03'
              variable: scf
              product: SNOWCOVER
        """,
    )
    shared_summary = setup_dir / "obs" / "summaries" / "project_2022_2023_base" / "scf_summary.csv"
    _write_series_csv(shared_summary, [{"date": "2023-01-03", "scf": 0.60}])
    record_fraction_summary_path(
        setup_dir=setup_dir,
        project_dir=project_dir,
        filename="scf_summary.csv",
        summary_csv=shared_summary,
    )

    base = project_dir / "steps" / "step_01_next" / "ensembles" / "prior"
    _write_series_csv(base / "open_loop" / "results" / "point_scf_roi.csv", [{"time": "2023-01-03", "scf": 0.5}])
    _write_series_csv(base / "member_001" / "results" / "point_scf_roi.csv", [{"time": "2023-01-03", "scf": 0.55}])
    _write_series_csv(base / "member_002" / "results" / "point_scf_roi.csv", [{"time": "2023-01-03", "scf": 0.65}])

    cases = extract_continuous_cases(project_dir=project_dir, setup_dir=setup_dir, variables=("scf",))

    assert len(cases) == 1
    assert cases[0].variable == "scf"
    assert cases[0].obs_value == 0.60


def test_extract_continuous_station_cases_activate_sister_link_on_first_sister_event(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-03'
              variable: station_swe
        """,
    )
    _write_station_benchmark_inputs(project_dir, setup_dir)

    cases = extract_continuous_cases(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("station_hs",),
    )

    by_time = {
        str(case.timestamp): case.stream
        for case in cases
        if case.variable == "station_hs" and case.obs_id == "station_a"
    }
    assert by_time["2023-01-02 00:00:00"] == "independent"
    assert by_time["2023-01-03 00:00:00"] == "semi_independent"


def test_benchmark_event_contexts_keep_wet_snow_line_distinct(tmp_path: Path) -> None:
    _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-03'
              variable: wet_snow_line
              product: WETSNOW
        """,
    )
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"

    contexts = analysis_event_contexts(project_dir, variables=("wet_snow",))
    wsl_contexts = analysis_event_contexts(project_dir, variables=("wet_snow_line",))
    by_variable = event_dates_by_variable(project_dir)

    assert contexts == []
    assert len(wsl_contexts) == 1
    assert wsl_contexts[0].variable == "wet_snow_line"
    assert by_variable == {"wet_snow_line": {pd.Timestamp("2023-01-03").date()}}


def test_extract_analysis_cases_uses_wet_snow_line_prior_and_posterior(tmp_path: Path) -> None:
    setup_dir, project_dir = _setup_basic_project(
        tmp_path,
        events_yaml="""
            - date: '2023-01-03'
              variable: wet_snow_line
              product: WETSNOW
        """,
    )
    _write_fraction_benchmark_inputs(setup_dir, project_dir)
    _write_series_csv(
        project_dir / "steps" / "step_00_init" / "assim" / "weights_wet_snow_line_20230103.csv",
        [
            {"member_id": "member_001", "value_model": 2360.0, "value_obs": 2375.0, "weight": 0.25},
            {"member_id": "member_002", "value_model": 2380.0, "value_obs": 2375.0, "weight": 0.75},
        ],
    )

    cases = extract_analysis_cases(
        project_dir=project_dir,
        setup_dir=setup_dir,
        variables=("wet_snow_line",),
    )

    assert len(cases) == 1
    case = cases[0]
    assert case.variable == "wet_snow_line"
    assert case.stream == "assimilation_fit"
    assert case.obs_value == 2375.0
    assert case.open_loop_value == 2350.0
    assert case.prior_values == (2360.0, 2380.0)
    assert case.posterior_values == (2360.0, 2380.0)
    assert case.posterior_weights == (0.25, 0.75)
