from __future__ import annotations
"""
openamundsen_da.core.prior_forcing

Build a prior ensemble of meteorological forcings (perturbed CSVs) for
openAMUNDSEN. Creates an open-loop copy and N member_XXX sets under the step
directory, matching the discovery used by the ensemble launcher.

Design
- Inputs: explicit input meteo dir, project dir, and step dir
- Dates: inclusive [start_date..end_date] read from the step YAML
- Params: read from project.yml under data_assimilation.prior_forcing
- Perturbations: temperature additive ΔT ~ N(0, σ_T²), precipitation factor
  f_p ~ LogNormal(μ_P, σ_P²), constant per member across stations and time
- Schema: requires 'date'; 'temp' and 'precip' are optional per station file
- Precip negatives: if 'precip' exists and contains negatives, abort
- Output: <step_dir>/ensembles/prior/{open_loop,member_XXX}/{meteo,results}
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen.util import read_yaml_file

from openamundsen_da.core.constants import (
    DA_BLOCK,
    DA_PRIOR_BLOCK,
    DA_ENSEMBLE_SIZE,
    DA_RANDOM_SEED,
    DA_SIGMA_T,
    DA_MU_P,
    DA_SIGMA_P,
    DEFAULT_TIME_COL,
    DEFAULT_TEMP_COL,
    DEFAULT_PRECIP_COL,
    ENSEMBLE_PRIOR,
    START_DATE,
    END_DATE,
)
from openamundsen_da.core.stats import sample_delta_t, sample_precip_factor
from openamundsen_da.io.paths import (
    find_project_yaml,
    find_step_yaml,
    meteo_dir_for_member,
    default_results_dir,
    prior_root as prior_root_dir,
    open_loop_dir as open_loop_dir_for_step,
    member_dir_for_index,
)


@dataclass
class PriorParams:
    """Prior configuration read from project.yml."""
    ensemble_size: int
    random_seed: int
    sigma_t: float
    mu_p: float
    sigma_p: float


def _read_prior_params(project_dir: Path) -> PriorParams:
    """Load prior configuration from project.yml > data_assimilation.prior_forcing."""
    proj_yaml = find_project_yaml(project_dir)
    cfg = read_yaml_file(proj_yaml) or {}
    da = (cfg.get(DA_BLOCK) or {}).get(DA_PRIOR_BLOCK) or {}
    try:
        return PriorParams(
            ensemble_size=int(da[DA_ENSEMBLE_SIZE]),
            random_seed=int(da[DA_RANDOM_SEED]),
            sigma_t=float(da[DA_SIGMA_T]),
            mu_p=float(da[DA_MU_P]),
            sigma_p=float(da[DA_SIGMA_P]),
        )
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(
            f"Missing prior parameter in project.yml:{' ' + missing} under "
            f"{DA_BLOCK}.{DA_PRIOR_BLOCK}"
        ) from e


def _read_step_dates(step_dir: Path) -> Tuple[pd.Timestamp, pd.Timestamp, Path]:
    """Read inclusive [start_date..end_date] from the step YAML in step_dir."""
    step_yaml = find_step_yaml(step_dir)
    step_cfg = read_yaml_file(step_yaml) or {}
    try:
        start = pd.to_datetime(step_cfg[START_DATE])
        end = pd.to_datetime(step_cfg[END_DATE])
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"Missing required key '{missing}' in {step_yaml}") from e
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"Invalid start/end date in {step_yaml}")
    return start, end, step_yaml


def _list_station_csvs(meteo_dir: Path) -> Tuple[Path, List[Path]]:
    """Return path to stations.csv and a sorted list of per-station CSV files."""
    stations = meteo_dir / "stations.csv"
    if not stations.is_file():
        raise FileNotFoundError(f"Missing stations.csv in {meteo_dir}")
    csvs = sorted(p for p in meteo_dir.glob("*.csv") if p.name != "stations.csv")
    if not csvs:
        raise FileNotFoundError(f"No station CSV files found in {meteo_dir}")
    return stations, csvs


def _ensure_schema_and_precip_positive(df: pd.DataFrame, src: Path) -> None:
    """Validate required columns and ensure no negative precipitation values exist.

    - 'date' must be present in all files
    - 'temp' and 'precip' are optional; if 'precip' exists it must be non-negative
    """
    if DEFAULT_TIME_COL not in df.columns:
        raise ValueError(f"{src.name}: missing required column: {DEFAULT_TIME_COL}")
    if DEFAULT_PRECIP_COL in df.columns:
        p = pd.to_numeric(df[DEFAULT_PRECIP_COL], errors="coerce")
        if (p.dropna() < 0).any():
            raise ValueError(f"{src.name}: precipitation contains negative values")


def _inclusive_filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Inclusive date filtering on 'date' column."""
    t = pd.to_datetime(df[DEFAULT_TIME_COL], errors="coerce")
    mask = (t >= start) & (t <= end)
    return df.loc[mask].copy()


def _make_member_dirs(root: Path) -> Tuple[Path, Path]:
    """Create meteo and results subdirs under the given member/open_loop root."""
    meteo = meteo_dir_for_member(root)
    results = default_results_dir(root)
    meteo.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return meteo, results


def _write_info(member_root: Path, name: str, seed: int, delta_t: float, f_p: float,
                start: pd.Timestamp, end: pd.Timestamp, input_dir: Path) -> None:
    """Write a compact INFO.txt summarizing the member perturbations and context."""
    info = member_root / "INFO.txt"
    lines = [
        f"Member: {name}",
        f"Random seed: {seed}",
        "",
        "Perturbations (constant per member):",
        f"  delta_T (additive): {delta_t:+.3f}",
        f"  precip factor f_p:  {f_p:.3f}",
        "",
        "Date filter (inclusive):",
        f"  start_date: {start}",
        f"  end_date:   {end}",
        "",
        "Schema (required):",
        f"  {DEFAULT_TIME_COL}, {DEFAULT_TEMP_COL}, {DEFAULT_PRECIP_COL}",
        "",
        "Input:",
        f"  meteo dir: {input_dir}",
        "",
        "Layout:",
        f"  {member_root / 'meteo'}",
        f"  {member_root / 'results'}",
    ]
    info.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(df: pd.DataFrame, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


def build_prior_ensemble(input_meteo_dir: Path | str, project_dir: Path | str, step_dir: Path | str) -> None:
    """Build open-loop and prior ensemble member meteo directories for a step.

    Parameters
    - input_meteo_dir: Path to original, long-span station CSV directory
      (contains stations.csv and <station_id>.csv files)
    - project_dir: Path to project root containing project.yml
    - step_dir: Path to step directory containing step_XX.yml with dates
    """
    input_meteo_dir = Path(input_meteo_dir)
    project_dir = Path(project_dir)
    step_dir = Path(step_dir)

    # Read configuration
    params = _read_prior_params(project_dir)
    start, end, step_yaml = _read_step_dates(step_dir)

    logger.info(
        "Building prior ensemble -> ensemble={ens}  N={n}  seed={seed}",
        ens=ENSEMBLE_PRIOR, n=params.ensemble_size, seed=params.random_seed,
    )
    logger.info("Dates (inclusive): {s} .. {e}", s=str(start), e=str(end))

    stations_csv, station_files = _list_station_csvs(input_meteo_dir)

    # Set RNG deterministically
    rng = np.random.default_rng(params.random_seed)

    # Prepare open_loop
    prior_root = prior_root_dir(step_dir)
    open_loop_root = open_loop_dir_for_step(step_dir)
    meteo_ol, results_ol = _make_member_dirs(open_loop_root)

    # Process open_loop (unperturbed, filtered)
    for src in station_files:
        df = pd.read_csv(src)
        _ensure_schema_and_precip_positive(df, src)
        df = _inclusive_filter(df, start, end)
        _write_csv(df, meteo_ol / src.name)
    (meteo_ol / "stations.csv").write_text(Path(stations_csv).read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("Open-loop written: {p}", p=str(open_loop_root))

    # Create members
    for i in range(1, params.ensemble_size + 1):
        member_name = f"member_{i:03d}"
        member_root = member_dir_for_index(step_dir, i)
        meteo_dir, _ = _make_member_dirs(member_root)

        # Sample perturbations (stationary per member)
        delta_t = sample_delta_t(rng, params.sigma_t)
        f_p = sample_precip_factor(rng, params.mu_p, params.sigma_p)
        logger.info("[{m}] delta_T={dt:+.3f}  f_p={fp:.3f}", m=member_name, dt=delta_t, fp=f_p)

        for src in station_files:
            df = pd.read_csv(src)
            _ensure_schema_and_precip_positive(df, src)
            df = _inclusive_filter(df, start, end)
            # Apply perturbations only where columns exist
            if DEFAULT_TEMP_COL in df.columns:
                df[DEFAULT_TEMP_COL] = pd.to_numeric(df[DEFAULT_TEMP_COL], errors="coerce") + delta_t
            if DEFAULT_PRECIP_COL in df.columns:
                df[DEFAULT_PRECIP_COL] = pd.to_numeric(df[DEFAULT_PRECIP_COL], errors="coerce") * f_p
            _write_csv(df, meteo_dir / src.name)

        # Copy stations.csv unchanged and write member INFO
        (meteo_dir / "stations.csv").write_text(Path(stations_csv).read_text(encoding="utf-8"), encoding="utf-8")
        _write_info(member_root, member_name, params.random_seed, delta_t, f_p, start, end, input_meteo_dir)

    logger.info("Prior ensemble completed under: {root}", root=str(prior_root))


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prior meteo ensemble (CSV)")
    p.add_argument("--input-meteo-dir", required=True, type=Path)
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--step-dir", required=True, type=Path)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        build_prior_ensemble(args.input_meteo_dir, args.project_dir, args.step_dir)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
