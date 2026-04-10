"""Benchmark table and manifest writers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common import ensure_dir


_VARIABLE_ORDER = ("scf", "wet_snow", "station_hs", "station_swe")
_STREAM_ORDER = ("assimilation_fit", "semi_independent", "independent")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sort_variable(variable: str) -> tuple[int, str]:
    token = str(variable)
    try:
        return (_VARIABLE_ORDER.index(token), token)
    except ValueError:
        return (len(_VARIABLE_ORDER), token)


def _sort_stream(stream: str) -> tuple[int, str]:
    token = str(stream)
    try:
        return (_STREAM_ORDER.index(token), token)
    except ValueError:
        return (len(_STREAM_ORDER), token)


def _ordered_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    working = df.copy()
    if "variable" in working.columns:
        working["_variable_order"] = working["variable"].map(_sort_variable)
    if "stream" in working.columns:
        working["_stream_order"] = working["stream"].map(_sort_stream)
    sort_cols = [col for col in ("assimilation_date", "_variable_order", "_stream_order") if col in working.columns]
    if sort_cols:
        working = working.sort_values(sort_cols)
    drop_cols = [col for col in ("_variable_order", "_stream_order") if col in working.columns]
    return working.drop(columns=drop_cols).reset_index(drop=True)


def write_case_tables(results_dir: Path, case_scores: pd.DataFrame) -> dict[str, Path]:
    cases_dir = ensure_dir(results_dir / "cases")
    outputs: dict[str, Path] = {}

    continuous = case_scores[case_scores["score_set"] == "continuous"].copy()
    analysis = case_scores[case_scores["score_set"] == "analysis"].copy()

    continuous_path = cases_dir / "continuous_case_scores.csv"
    continuous.to_csv(continuous_path, index=False)
    outputs["continuous_cases"] = continuous_path

    analysis_path = cases_dir / "analysis_case_scores.csv"
    analysis.to_csv(analysis_path, index=False)
    outputs["analysis_cases"] = analysis_path
    return outputs


def write_score_tables(
    results_dir: Path,
    *,
    event_scores: pd.DataFrame,
    project_scores: pd.DataFrame,
    reliability: pd.DataFrame,
) -> dict[str, Path]:
    scores_dir = ensure_dir(results_dir / "scores")
    outputs: dict[str, Path] = {}

    event_path = scores_dir / "event_scores.csv"
    event_scores.to_csv(event_path, index=False)
    outputs["event_scores"] = event_path

    project_path = scores_dir / "project_scores.csv"
    project_scores.to_csv(project_path, index=False)
    outputs["project_scores"] = project_path

    reliability_path = scores_dir / "project_reliability.csv"
    reliability.to_csv(reliability_path, index=False)
    outputs["project_reliability"] = reliability_path
    return outputs


def _analysis_daily_summary(event_scores: pd.DataFrame) -> pd.DataFrame:
    analysis = event_scores[
        (event_scores["score_set"] == "analysis")
        & (event_scores["representation"].isin(["prior", "posterior"]))
    ].copy()
    if analysis.empty:
        return pd.DataFrame(columns=["assimilation_date", "variable", "stream", "representation", "crpss", "ner", "bias"])
    analysis["assimilation_date"] = pd.to_datetime(analysis["date"]).dt.normalize()
    return (
        analysis.groupby(["assimilation_date", "variable", "stream", "representation"], dropna=False, sort=True)[
            ["crpss", "ner", "bias"]
        ]
        .mean()
        .reset_index()
    )


def build_project_summary_table(
    *,
    event_scores: pd.DataFrame,
    project_scores: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "variable",
        "stream",
        "n_project_points",
        "whole_project_crpss",
        "whole_project_ner",
        "whole_project_bias",
        "n_update_dates",
        "update_prior_crpss",
        "update_posterior_crpss",
        "update_prior_ner",
        "update_posterior_ner",
        "update_prior_bias",
        "update_posterior_bias",
    ]
    if project_scores.empty:
        return pd.DataFrame(columns=columns)

    whole_project = project_scores[
        (project_scores["score_set"] == "continuous")
        & (project_scores["representation"] == "da_informed_ensemble")
    ][["variable", "stream", "n_cases", "crpss", "ner", "bias"]].rename(
        columns={
            "n_cases": "n_project_points",
            "crpss": "whole_project_crpss",
            "ner": "whole_project_ner",
            "bias": "whole_project_bias",
        }
    )

    update_metrics = project_scores[
        (project_scores["score_set"] == "analysis")
        & (project_scores["representation"].isin(["prior", "posterior"]))
    ][["variable", "stream", "representation", "crpss", "ner", "bias"]].copy()

    if update_metrics.empty:
        update_wide = pd.DataFrame(columns=["variable", "stream"])
    else:
        update_wide = update_metrics.pivot_table(
            index=["variable", "stream"],
            columns="representation",
            values=["crpss", "ner", "bias"],
            aggfunc="first",
        ).reset_index()
        update_wide.columns = [
            "_".join([str(part) for part in col if str(part) != ""]).strip("_")
            if isinstance(col, tuple)
            else str(col)
            for col in update_wide.columns
        ]
        update_wide = update_wide.rename(
            columns={
                "crpss_prior": "update_prior_crpss",
                "crpss_posterior": "update_posterior_crpss",
                "ner_prior": "update_prior_ner",
                "ner_posterior": "update_posterior_ner",
                "bias_prior": "update_prior_bias",
                "bias_posterior": "update_posterior_bias",
            }
        )

    daily_updates = _analysis_daily_summary(event_scores)
    if daily_updates.empty:
        update_counts = pd.DataFrame(columns=["variable", "stream", "n_update_dates"])
    else:
        update_counts = (
            daily_updates.groupby(["variable", "stream"], dropna=False)["assimilation_date"]
            .nunique()
            .reset_index(name="n_update_dates")
        )

    merged = whole_project.merge(update_wide, on=["variable", "stream"], how="outer")
    merged = merged.merge(update_counts, on=["variable", "stream"], how="left")
    merged["n_update_dates"] = merged["n_update_dates"].fillna(0).astype(int)
    if "n_project_points" in merged.columns:
        merged["n_project_points"] = merged["n_project_points"].astype("Int64")
    return _ordered_sort(merged.reindex(columns=columns))


def build_update_summary_table(event_scores: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "assimilation_date",
        "variable",
        "stream",
        "prior_crpss",
        "posterior_crpss",
        "prior_ner",
        "posterior_ner",
        "prior_bias",
        "posterior_bias",
        "delta_crpss",
        "delta_ner",
        "delta_abs_bias",
    ]
    daily_updates = _analysis_daily_summary(event_scores)
    if daily_updates.empty:
        return pd.DataFrame(columns=columns)

    wide = daily_updates.pivot_table(
        index=["assimilation_date", "variable", "stream"],
        columns="representation",
        values=["crpss", "ner", "bias"],
        aggfunc="first",
    ).reset_index()
    wide.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).strip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in wide.columns
    ]
    wide = wide.rename(
        columns={
            "crpss_prior": "prior_crpss",
            "crpss_posterior": "posterior_crpss",
            "ner_prior": "prior_ner",
            "ner_posterior": "posterior_ner",
            "bias_prior": "prior_bias",
            "bias_posterior": "posterior_bias",
        }
    )
    wide["delta_crpss"] = wide["posterior_crpss"] - wide["prior_crpss"]
    wide["delta_ner"] = wide["posterior_ner"] - wide["prior_ner"]
    wide["delta_abs_bias"] = wide["prior_bias"].abs() - wide["posterior_bias"].abs()
    wide["assimilation_date"] = pd.to_datetime(wide["assimilation_date"]).dt.date
    return _ordered_sort(wide.reindex(columns=columns))


def write_summary_tables(
    results_dir: Path,
    *,
    event_scores: pd.DataFrame,
    project_scores: pd.DataFrame,
    reliability: pd.DataFrame,
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    del reliability
    tables_dir = ensure_dir(results_dir / "tables")
    outputs: dict[str, Path] = {}
    tables: dict[str, pd.DataFrame] = {}

    stale_names = (
        "project_summary_wide.csv",
        "event_summary_wide.csv",
        "reliability_summary_wide.csv",
        "improvement_summary.csv",
        "project_summary_wide.md",
        "event_summary_wide.md",
        "reliability_summary_wide.md",
        "improvement_summary.md",
    )
    for stale_name in stale_names:
        stale_path = tables_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    project_summary = build_project_summary_table(event_scores=event_scores, project_scores=project_scores)
    project_path = tables_dir / "project_summary.csv"
    project_summary.to_csv(project_path, index=False)
    outputs["project_summary"] = project_path
    tables["project_summary"] = project_summary

    update_summary = build_update_summary_table(event_scores)
    update_path = tables_dir / "update_summary.csv"
    update_summary.to_csv(update_path, index=False)
    outputs["update_summary"] = update_path
    tables["update_summary"] = update_summary
    return outputs, tables


def write_summary_markdown(
    results_dir: Path,
    *,
    project_dir: Path,
    benchmark_variables: list[str],
    independent_variables: list[str],
    project_summary: pd.DataFrame,
    update_summary: pd.DataFrame,
) -> Path:
    out = results_dir / "summary.md"
    _ensure_parent(out)

    lines = [
        "# Benchmark Summary",
        "",
        f"Project: `{project_dir.name}`",
        "",
        f"Benchmark variables: `{', '.join(benchmark_variables)}`",
        "",
        f"Independent benchmark additions: `{', '.join(independent_variables) if independent_variables else 'none'}`",
        "",
        "This is an observation-based benchmarking layer focused on DA performance over the open-loop baseline.",
        "Headline metrics are `CRPSS` and `NER`; `bias` is kept as the core support metric.",
        "Score basis remains ROI-based for `scf` / `wet_snow` and station-point based for `station_hs` / `station_swe`.",
        "These outputs do not replace future holdout, LOOCV, or OSSE validation.",
        "",
        "## Core Outputs",
        "",
        "- `plots/assim/scores/performance_scores.png` shows update-date `CRPSS` and `NER` on project assimilation dates, using `prior` and `posterior` points for assimilated and transfer-observed variables.",
        "- `results/benchmark/tables/project_summary.csv` gives the compact whole-project summary.",
        "- `results/benchmark/tables/update_summary.csv` gives prior/posterior update-date skill.",
        "",
        "## Whole-Project Highlights",
        "",
    ]

    if project_summary.empty:
        lines.append("- No whole-project benchmark rows were available.")
    else:
        for row in project_summary.itertuples(index=False):
            lines.append(
                "- "
                f"`{row.variable}` (`{row.stream}`): "
                f"`CRPSS={float(row.whole_project_crpss):.3f}` "
                f"`NER={float(row.whole_project_ner):.3f}` "
                f"`bias={float(row.whole_project_bias):.3f}`"
            )

    lines.extend(["", "## Update-Date Highlights", ""])
    if update_summary.empty:
        lines.append("- No assimilation-date benchmark rows were available.")
    else:
        best_rows = update_summary.copy()
        best_rows["sort_key"] = pd.to_numeric(best_rows["delta_crpss"], errors="coerce")
        best_rows = best_rows.sort_values("sort_key", ascending=False).head(6)
        for row in best_rows.itertuples(index=False):
            lines.append(
                "- "
                f"`{row.assimilation_date}` `{row.variable}`: "
                f"`{row.stream}` "
                f"`prior CRPSS={float(row.prior_crpss):.3f}` -> "
                f"`posterior CRPSS={float(row.posterior_crpss):.3f}`, "
                f"`prior NER={float(row.prior_ner):.3f}` -> "
                f"`posterior NER={float(row.posterior_ner):.3f}`"
            )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `assimilation_fit` means the exact variable/date pair was assimilated.",
            "- `semi_independent` means the exact variable/date pair was not assimilated, but it is still linked through same-variable reuse elsewhere in the project or through a sister station variable.",
            "- `independent` means the benchmark variable is never assimilated anywhere in the project and is not downgraded by station linkage.",
            "- Positive `CRPSS` and `NER` indicate improvement over `open_loop`.",
            "- Positive `delta_abs_bias` means the posterior reduced absolute bias relative to the prior.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_manifest(
    results_dir: Path,
    *,
    project_dir: Path,
    benchmark_variables: list[str],
    independent_variables: list[str],
    outputs: dict[str, Path],
    case_scores: pd.DataFrame,
    event_scores: pd.DataFrame,
    project_scores: pd.DataFrame,
) -> Path:
    out = results_dir / "manifest.json"
    _ensure_parent(out)
    payload: dict[str, Any] = {
        "project_dir": str(project_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_variables": benchmark_variables,
        "independent_variables": independent_variables,
        "case_rows": int(len(case_scores)),
        "event_rows": int(len(event_scores)),
        "project_rows": int(len(project_scores)),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "caveat": (
            "Observation-based benchmark outputs. "
            "Stream labels distinguish assimilation_fit, semi_independent, and independent benchmark views. "
            "These results do not replace future holdout, LOOCV, or OSSE validation."
        ),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
