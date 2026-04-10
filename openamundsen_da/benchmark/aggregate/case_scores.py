"""Case-level benchmark scoring and distribution summaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from openamundsen_da.benchmark.extract.cases import RawBenchmarkCase
from openamundsen_da.util.stats import (
    ensemble_crps,
    midpoint_pit,
    weighted_mean,
    weighted_quantile,
    weighted_std,
)
from openamundsen_da.util.ts import hydro_year_index


_COVERAGE_LEVELS = (50, 80, 90)
_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _rank_bin(values: np.ndarray, observation: float) -> tuple[int, int] | None:
    if values.size == 0:
        return None
    ranks = np.sort(values)
    left = int(np.sum(ranks < observation))
    equal = int(np.sum(ranks == observation))
    if equal > 0:
        bin_index = left + int(equal // 2)
    else:
        bin_index = left
    n_bins = int(values.size + 1)
    bin_index = min(max(bin_index, 0), n_bins - 1)
    return bin_index, n_bins


def _coverage_flags(values: np.ndarray, observation: float, *, weights: np.ndarray | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for nominal in _COVERAGE_LEVELS:
        alpha = (1.0 - nominal / 100.0) / 2.0
        lower = weighted_quantile(values, alpha, weights=weights)
        upper = weighted_quantile(values, 1.0 - alpha, weights=weights)
        out[f"coverage_{nominal}"] = int(lower <= observation <= upper)
    return out


def _quantile_summary(values: np.ndarray, *, weights: np.ndarray | None) -> dict[str, float]:
    return {
        f"q{int(q * 100):02d}": weighted_quantile(values, q, weights=weights)
        for q in _QUANTILES
    }


def _representation_case_row(
    *,
    raw_case: RawBenchmarkCase,
    representation: str,
    values: Sequence[float],
    weights: Sequence[float] | None = None,
    ensemble_kind: str,
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Representation {representation!r} received no values")
    w_arr = None if weights is None else np.asarray(weights, dtype=float)
    obs_value = float(raw_case.obs_value)
    timestamp = pd.Timestamp(raw_case.timestamp)
    pred_mean = weighted_mean(arr, weights=w_arr)
    spread = weighted_std(arr, weights=w_arr)
    error = pred_mean - obs_value
    crps = ensemble_crps(arr, obs_value, weights=w_arr)
    pit = midpoint_pit(arr, obs_value, weights=w_arr)
    rank = _rank_bin(arr, obs_value) if weights is None and arr.size > 1 else None
    row: dict[str, Any] = {
        "score_set": raw_case.score_set,
        "variable": raw_case.variable,
        "stream": raw_case.stream,
        "representation": representation,
        "ensemble_kind": ensemble_kind,
        "timestamp": timestamp,
        "date": timestamp.date().isoformat(),
        "month": int(timestamp.month),
        "water_year": int(hydro_year_index(pd.DatetimeIndex([timestamp]), 10, 1)[0]),
        "obs_id": raw_case.obs_id,
        "step_name": raw_case.step_name,
        "obs_value": obs_value,
        "pred_mean": pred_mean,
        "spread": spread,
        "error": error,
        "abs_error": abs(error),
        "sq_error": error ** 2,
        "crps": crps,
        "pit": pit,
        "n_members": int(arr.size),
    }
    row.update(_quantile_summary(arr, weights=w_arr))
    row.update(_coverage_flags(arr, obs_value, weights=w_arr))
    if rank is None:
        row["rank_bin"] = np.nan
        row["rank_bin_count"] = np.nan
    else:
        row["rank_bin"] = int(rank[0])
        row["rank_bin_count"] = int(rank[1])
    return row


def build_case_scores(raw_cases: Sequence[RawBenchmarkCase]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        rows.append(
            _representation_case_row(
                raw_case=raw_case,
                representation="open_loop",
                values=(raw_case.open_loop_value,),
                ensemble_kind="deterministic",
            )
        )
        if raw_case.score_set == "continuous":
            if raw_case.da_informed_values is None:
                raise ValueError("Continuous benchmark case missing da_informed_values")
            rows.append(
                _representation_case_row(
                    raw_case=raw_case,
                    representation="da_informed_ensemble",
                    values=raw_case.da_informed_values,
                    ensemble_kind="unweighted_ensemble",
                )
            )
            continue
        if raw_case.prior_values is None:
            raise ValueError("Analysis benchmark case missing prior_values")
        if raw_case.posterior_values is None or raw_case.posterior_weights is None:
            raise ValueError("Analysis benchmark case missing posterior values or weights")
        rows.append(
            _representation_case_row(
                raw_case=raw_case,
                representation="prior",
                values=raw_case.prior_values,
                ensemble_kind="unweighted_ensemble",
            )
        )
        rows.append(
            _representation_case_row(
                raw_case=raw_case,
                representation="posterior",
                values=raw_case.posterior_values,
                weights=raw_case.posterior_weights,
                ensemble_kind="weighted_ensemble",
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "score_set",
                "variable",
                "stream",
                "representation",
                "ensemble_kind",
                "timestamp",
                "date",
                "month",
                "water_year",
                "obs_id",
                "step_name",
                "obs_value",
                "pred_mean",
                "spread",
                "error",
                "abs_error",
                "sq_error",
                "crps",
                "pit",
                "n_members",
                "q05",
                "q25",
                "q50",
                "q75",
                "q95",
                "coverage_50",
                "coverage_80",
                "coverage_90",
                "rank_bin",
                "rank_bin_count",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["score_set", "variable", "stream", "timestamp", "obs_id", "representation"]
    ).reset_index(drop=True)


__all__ = ["build_case_scores"]
