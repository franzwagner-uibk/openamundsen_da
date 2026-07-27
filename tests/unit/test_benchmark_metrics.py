from __future__ import annotations

import math
import textwrap
from pathlib import Path

from openamundsen_da.benchmark.cases import RawBenchmarkCase
from openamundsen_da.benchmark.metrics import aggregate_scores, build_case_scores, enrich_case_scores, reliability_rows


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def test_build_case_scores_uses_weighted_posterior_for_analysis_cases() -> None:
    raw_case = RawBenchmarkCase(
        score_set="analysis",
        variable="station_hs",
        stream="assimilation_fit",
        timestamp="2023-01-02",
        obs_id="station_a",
        step_name="step_00_init",
        obs_value=2.0,
        open_loop_value=0.5,
        da_informed_values=None,
        prior_values=(1.0, 3.0),
        posterior_values=(1.0, 3.0),
        posterior_weights=(0.2, 0.8),
        sigma_base=0.5,
    )

    case_scores = build_case_scores([raw_case])

    assert set(case_scores["representation"]) == {"open_loop", "prior", "posterior"}

    posterior = case_scores.loc[case_scores["representation"] == "posterior"].iloc[0]
    prior = case_scores.loc[case_scores["representation"] == "prior"].iloc[0]
    open_loop = case_scores.loc[case_scores["representation"] == "open_loop"].iloc[0]

    assert math.isclose(float(prior["pred_mean"]), 2.0)
    assert math.isclose(float(posterior["pred_mean"]), 2.6)
    assert math.isclose(float(posterior["spread"]), 0.8)
    assert math.isclose(float(posterior["sigma_base"]), 0.5)
    assert math.isclose(float(posterior["z_error"]), 1.2)
    assert int(open_loop["n_members"]) == 1
    assert math.isclose(float(open_loop["crps"]), 1.5)

    project_scores = aggregate_scores(
        case_scores,
        group_cols=("score_set", "variable", "stream"),
    )
    posterior_project = project_scores.loc[project_scores["representation"] == "posterior"].iloc[0]
    assert math.isclose(float(posterior_project["rmse"]), 0.6)
    assert math.isclose(float(posterior_project["z_rmse"]), 1.2)
    assert math.isclose(float(posterior_project["ner"]), 0.6)
    assert math.isclose(float(posterior_project["zskill"]), 0.6)
    assert math.isclose(float(posterior_project["crpss"]), 0.5466666666666666)


def test_build_case_scores_keeps_zskill_undefined_for_non_station_cases() -> None:
    raw_case = RawBenchmarkCase(
        score_set="continuous",
        variable="scf",
        stream="independent",
        timestamp="2023-01-02",
        obs_id="roi",
        step_name="step_00_init",
        obs_value=0.3,
        open_loop_value=0.1,
        da_informed_values=(0.1, 0.5),
        prior_values=None,
        posterior_values=None,
        posterior_weights=None,
    )

    case_scores = build_case_scores([raw_case])
    project_scores = aggregate_scores(
        case_scores,
        group_cols=("score_set", "variable", "stream"),
    )

    assert case_scores["sigma_base"].isna().all()
    assert case_scores["z_error"].isna().all()
    assert case_scores["z_sq_error"].isna().all()
    assert project_scores["z_rmse"].isna().all()
    assert project_scores["zskill"].isna().all()


def test_reliability_rows_include_pit_histogram_for_pf_weighted_ensemble() -> None:
    raw_cases = [
        RawBenchmarkCase(
            score_set="continuous",
            variable="scf",
            stream="independent",
            timestamp="2023-01-02",
            obs_id="roi",
            step_name="step_00_init",
            obs_value=0.3,
            open_loop_value=0.1,
            da_informed_values=(0.1, 0.5),
            prior_values=None,
            posterior_values=None,
            posterior_weights=None,
        ),
        RawBenchmarkCase(
            score_set="continuous",
            variable="scf",
            stream="independent",
            timestamp="2023-01-03",
            obs_id="roi",
            step_name="step_01_next",
            obs_value=0.9,
            open_loop_value=0.8,
            da_informed_values=(0.2, 0.6),
            prior_values=None,
            posterior_values=None,
            posterior_weights=None,
        ),
    ]

    case_scores = build_case_scores(raw_cases)
    reliability = reliability_rows(
        case_scores,
        group_cols=("score_set", "variable", "stream"),
    )

    da_pit = reliability[
        (reliability["representation"] == "da_informed_ensemble")
        & (reliability["diagnostic"] == "pit_histogram")
    ].copy()
    coverage = reliability[
        (reliability["representation"] == "da_informed_ensemble")
        & (reliability["diagnostic"] == "interval_coverage")
    ].copy()

    assert sorted(da_pit["bin_index"].astype(int).tolist()) == list(range(10))
    assert int(da_pit["count"].sum()) == 2
    assert math.isclose(float(coverage.loc[coverage["nominal_level"] == 0.5, "value"].iloc[0]), 0.5)
    assert math.isclose(float(coverage.loc[coverage["nominal_level"] == 0.8, "value"].iloc[0]), 0.5)


def test_enrich_case_scores_marks_high_uncertainty_station_rows_for_non_sigma_exclusion(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    _write_text(
        project_dir / "project_2022_2023.yml",
        """
        start_date: '2023-01-01'
        end_date: '2023-01-02'
        obs:
          stations:
            dir: obs/stations
        data_assimilation:
          station:
            default_station_uncertainty_pct: 25
            min_station_uncertainty_pct: 10
            single_station_factor: 2.0
        """,
    )
    _write_text(
        setup_dir / "obs" / "stations" / "stations_da_metadata.csv",
        """
        station_id,station_uncertainty_pct,hs_sigma_abs_min,swe_sigma_abs_min
        station_a,500,0.20,8.0
        """,
    )

    case_scores = build_case_scores(
        [
            RawBenchmarkCase(
                score_set="continuous",
                variable="station_hs",
                stream="independent",
                timestamp="2023-01-02",
                obs_id="station_a",
                step_name="step_00_init",
                obs_value=1.0,
                open_loop_value=2.0,
                da_informed_values=(1.0, 1.0),
                prior_values=None,
                posterior_values=None,
                posterior_weights=None,
                sigma_base=0.2,
            )
        ]
    )

    enriched = enrich_case_scores(
        case_scores,
        project_dir=project_dir,
        setup_dir=setup_dir,
        score_station_sigma_threshold=200,
    )

    assert set(enriched["station_uncertainty_pct"].dropna().astype(float)) == {500.0}
    assert enriched["exclude_from_non_sigma_scores"].tolist() == [True, True]


def test_aggregate_scores_excludes_flagged_station_rows_from_ner_and_crpss_but_keeps_zskill() -> None:
    case_scores = build_case_scores(
        [
            RawBenchmarkCase(
                score_set="continuous",
                variable="station_hs",
                stream="independent",
                timestamp="2023-01-02",
                obs_id="station_a",
                step_name="step_00_init",
                obs_value=1.0,
                open_loop_value=2.0,
                da_informed_values=(1.0, 1.0),
                prior_values=None,
                posterior_values=None,
                posterior_weights=None,
                sigma_base=1.0,
            ),
            RawBenchmarkCase(
                score_set="continuous",
                variable="station_hs",
                stream="independent",
                timestamp="2023-01-03",
                obs_id="station_b",
                step_name="step_01_next",
                obs_value=1.0,
                open_loop_value=1.5,
                da_informed_values=(3.0, 3.0),
                prior_values=None,
                posterior_values=None,
                posterior_weights=None,
                sigma_base=1.0,
            ),
        ]
    )
    case_scores["exclude_from_non_sigma_scores"] = case_scores["obs_id"].astype(str).eq("station_b")

    project_scores = aggregate_scores(
        case_scores,
        group_cols=("score_set", "variable", "stream"),
    )
    da_row = project_scores.loc[project_scores["representation"] == "da_informed_ensemble"].iloc[0]

    assert math.isclose(float(da_row["rmse"]), 0.0)
    assert math.isclose(float(da_row["crps"]), 0.0)
    assert math.isclose(float(da_row["ner"]), 1.0)
    assert math.isclose(float(da_row["crpss"]), 1.0)
    assert math.isclose(float(da_row["z_rmse"]), math.sqrt(2.0))
    assert math.isclose(float(da_row["zskill"]), 1.0 - (math.sqrt(2.0) / math.sqrt(0.625)))
