from __future__ import annotations

import math

from openamundsen_da.benchmark.cases import RawBenchmarkCase
from openamundsen_da.benchmark.metrics import aggregate_scores, build_case_scores, reliability_rows


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


def test_reliability_rows_include_rank_histogram_for_unweighted_ensemble() -> None:
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

    da_rank = reliability[
        (reliability["representation"] == "da_informed_ensemble")
        & (reliability["diagnostic"] == "rank_histogram")
    ].copy()
    coverage = reliability[
        (reliability["representation"] == "da_informed_ensemble")
        & (reliability["diagnostic"] == "interval_coverage")
    ].copy()

    assert sorted(da_rank["bin_index"].astype(int).tolist()) == [0, 1, 2]
    assert da_rank["count"].astype(int).tolist() == [0, 1, 1]
    assert math.isclose(float(coverage.loc[coverage["nominal_level"] == 0.5, "value"].iloc[0]), 0.5)
    assert math.isclose(float(coverage.loc[coverage["nominal_level"] == 0.8, "value"].iloc[0]), 0.5)
