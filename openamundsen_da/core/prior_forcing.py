from __future__ import annotations
"""
openamundsen_da.core.prior_forcing

Build a prior ensemble of meteorological forcings (perturbed CSVs) for
openAMUNDSEN. Creates an open-loop copy and N member_XXX sets under the step
directory, matching the discovery used by the ensemble launcher.

Design
- Inputs: explicit input meteo dir, project dir, and step dir
- Dates: inclusive [start_date..end_date] read from the step YAML
- Params: read from project YAML under data_assimilation.prior_forcing
- Perturbations: additive temperature and dew-point temperature offsets plus
  multiplicative precipitation and shortwave factors, constant per member
  across stations and time
- Schema: first column must be datetime (name is flexible); 'temp', 'precip',
  'rel_hum', and 'sw_in' are optional per station file
- Precip negatives: if 'precip' exists and contains negatives, abort
- Output: <step_dir>/ensembles/prior/{open_loop,member_XXX}/{meteo,results}
"""

import argparse
import concurrent.futures as cf
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.core.env import (
    apply_env_from_project,
    ensure_gdal_proj_from_conda,
    apply_numeric_thread_defaults,
    _read_yaml_file,
)

from openamundsen_da.core.constants import (
    DA_BLOCK,
    DA_PRIOR_BLOCK,
    DA_ENSEMBLE_SIZE,
    DA_RANDOM_SEED,
    DA_SIGMA_T,
    DA_MU_P,
    DA_SIGMA_P,
    DA_SIGMA_RH,
    DA_SIGMA_SW,
    ENSEMBLE_PRIOR,
    START_DATE,
    END_DATE,
    LOGURU_FORMAT,
)
from openamundsen_da.util.stats import (
    sample_delta_rh,
    sample_delta_t,
    sample_precip_factor,
    sample_shortwave_factor,
)
from openamundsen_da.util.parallel import pick_max_workers, resolve_base_seed
from openamundsen_da.util.meteo import filter_and_write_meteo
from openamundsen_da.io.paths import (
    find_setup_yaml,
    find_step_yaml,
    find_project_yaml,
    infer_project_dir,
    infer_setup_dir_from_project,
    meteo_dir_for_member,
    default_results_dir,
    prior_root as prior_root_dir,
    open_loop_dir as open_loop_dir_for_step,
    member_dir_for_index,
)


@dataclass
class PriorParams:
    """Prior configuration read from project YAML."""
    ensemble_size: int
    random_seed: int
    sigma_t: float
    mu_p: float
    sigma_p: float
    sigma_rh: float
    sigma_sw: float


def _read_prior_params(project_dir: Path) -> PriorParams:
    """Load prior configuration from project YAML > data_assimilation.prior_forcing."""
    project_yaml = find_project_yaml(project_dir)
    cfg = _read_yaml_file(project_yaml) or {}
    da = (cfg.get(DA_BLOCK) or {}).get(DA_PRIOR_BLOCK) or {}
    try:
        cfg_seed = int(da[DA_RANDOM_SEED])
        seed = resolve_base_seed(cfg_seed)
        if seed != cfg_seed:
            logger.info("OA_BASE_SEED override in effect -> seed={}", seed)
        return PriorParams(
            ensemble_size=int(da[DA_ENSEMBLE_SIZE]),
            random_seed=seed,
            sigma_t=float(da[DA_SIGMA_T]),
            mu_p=float(da[DA_MU_P]),
            sigma_p=float(da[DA_SIGMA_P]),
            sigma_rh=float(da.get(DA_SIGMA_RH, 0.0)),
            sigma_sw=float(da.get(DA_SIGMA_SW, 0.0)),
        )
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(
            f"Missing prior parameter in project YAML:{' ' + missing} under "
            f"{DA_BLOCK}.{DA_PRIOR_BLOCK}"
        ) from e


def _read_step_start_and_project_end(step_dir: Path) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Read step start_date and project end_date (inclusive).

    Robustness:
    - step start_date is mandatory (from step YAML)
    - project end_date is mandatory (from project YAML)
    """
    # Read the required step start.
    step_yaml = find_step_yaml(step_dir)
    step_cfg = _read_yaml_file(step_yaml) or {}
    try:
        start = pd.to_datetime(step_cfg[START_DATE])
    except KeyError as e:
        raise ValueError(f"Missing required key '{START_DATE}' in {step_yaml}") from e
    if pd.isna(start):
        raise ValueError(f"Invalid {START_DATE} in {step_yaml}")

    # Read the required project end.
    project_dir = infer_project_dir(step_dir)
    project_yaml = find_project_yaml(project_dir)
    project_cfg = _read_yaml_file(project_yaml) or {}
    try:
        end = pd.to_datetime(project_cfg[END_DATE])
    except KeyError as e:
        raise ValueError(f"Missing required key '{END_DATE}' in {project_yaml}") from e
    if pd.isna(end):
        raise ValueError(f"Invalid {END_DATE} in {project_yaml}")
    return start, end


def _make_member_dirs(root: Path) -> Tuple[Path, Path]:
    """Create meteo and results subdirs under the given member/open_loop root."""
    meteo = meteo_dir_for_member(root)
    results = default_results_dir(root)
    meteo.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return meteo, results


def _write_info(
    member_root: Path,
    name: str,
    seed: int,
    delta_t: float,
    f_p: float,
    delta_rh: float,
    f_sw: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    input_dir: Path,
) -> None:
    """Write a compact INFO.txt summarizing the member perturbations and context."""
    info = member_root / "INFO.txt"
    lines = [
        f"Member: {name}",
        f"Random seed: {seed}",
        "",
        "Perturbations (constant per member):",
        f"  delta_T (additive): {delta_t:+.3f}",
        f"  precip factor f_p:  {f_p:.3f}",
        f"  delta_Td (dew point additive): {delta_rh:+.3f}",
        f"  shortwave factor f_sw: {f_sw:.3f}",
        "",
        "Date filter (inclusive):",
        f"  start_date: {start}",
        f"  end_date:   {end}",
        "",
        "Schema:",
        "  required: first column = datetime (name flexible)",
        "  optional: temp, precip, rel_hum, sw_in",
        "",
        "Input:",
        f"  meteo dir: {input_dir}",
        "",
        "Layout:",
        f"  {member_root / 'meteo'}",
        f"  {member_root / 'results'}",
    ]
    info.write_text("\n".join(lines), encoding="utf-8")


def _build_member(
    member_idx: int,
    member_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    delta_t: float,
    f_p: float,
    delta_rh: float,
    f_sw: float,
    random_seed: int,
    input_meteo_dir: Path,
) -> None:
    """Worker: write perturbed meteo for one member."""
    meteo_dir, _ = _make_member_dirs(member_root)
    filter_and_write_meteo(
        src_dir=input_meteo_dir,
        dst_dir=meteo_dir,
        start=start,
        end=end,
        delta_t=delta_t,
        f_p=f_p,
        delta_rh=delta_rh,
        f_sw=f_sw,
    )
    _write_info(
        member_root,
        f"member_{member_idx:03d}",
        seed=random_seed,
        delta_t=delta_t,
        f_p=f_p,
        delta_rh=delta_rh,
        f_sw=f_sw,
        start=start,
        end=end,
        input_dir=input_meteo_dir,
    )


def build_prior_ensemble(
    input_meteo_dir: Path | str,
    project_dir: Path | str,
    step_dir: Path | str,
    *,
    max_workers: int | None = None,
    overwrite: bool = False,
) -> None:
    """Build open-loop and prior ensemble member meteo directories for a step.

    Parameters
    - input_meteo_dir: Path to original, long-span station CSV directory
      (contains stations.csv and <station_id>.csv files)
    - project_dir: Path to project directory under setup/projects
    - step_dir: Path to step directory containing step_XX.yml with dates
    - max_workers: Optional worker cap for member creation (defaults to
      min(CPU, ensemble_size, MAX_WORKERS env))
    - overwrite: If False, skip existing open_loop/member directories; if True,
      (re)write outputs
    """
    input_meteo_dir = Path(input_meteo_dir)
    project_dir = Path(project_dir)
    step_dir = Path(step_dir)

    # Read configuration
    inferred_project_dir = infer_project_dir(step_dir)
    if inferred_project_dir != project_dir:
        raise ValueError(
            f"step_dir belongs to {inferred_project_dir}, but --project-dir is {project_dir}"
        )
    params = _read_prior_params(project_dir)
    start, end = _read_step_start_and_project_end(step_dir)

    logger.info(
        "Building prior ensemble -> ensemble={ens}  N={n}  seed={seed}",
        ens=ENSEMBLE_PRIOR, n=params.ensemble_size, seed=params.random_seed,
    )
    logger.info("Dates (inclusive): {s} .. {e}", s=str(start), e=str(end))

    # Set RNG deterministically
    rng = np.random.default_rng(params.random_seed)

    # Prepare open_loop
    prior_root = prior_root_dir(step_dir)
    open_loop_root = open_loop_dir_for_step(step_dir)
    if open_loop_root.exists() and not overwrite:
        logger.info("Open-loop exists -> skipping (use --overwrite to rebuild)")
    else:
        meteo_ol, _ = _make_member_dirs(open_loop_root)
        filter_and_write_meteo(
            src_dir=input_meteo_dir,
            dst_dir=meteo_ol,
            start=start,
            end=end,
            delta_t=0.0,
            f_p=1.0,
            delta_rh=0.0,
            f_sw=1.0,
        )
        logger.info("Open-loop written: {p}", p=str(open_loop_root))

    # Prepare member tasks (skip existing unless overwrite)
    tasks = []
    for i in range(1, params.ensemble_size + 1):
        member_name = f"member_{i:03d}"
        member_root = member_dir_for_index(step_dir, i)
        if member_root.exists() and not overwrite:
            logger.info(f"[{member_name}] exists -> skipping (use --overwrite)")
            continue
        delta_t = sample_delta_t(rng, params.sigma_t)
        f_p = sample_precip_factor(rng, params.mu_p, params.sigma_p)
        delta_rh = sample_delta_rh(rng, params.sigma_rh)
        f_sw = sample_shortwave_factor(rng, params.sigma_sw)
        logger.info(
            "[{m}] delta_T={dt:+.3f}  f_p={fp:.3f}  delta_Td={drh:+.3f}  f_sw={fsw:.3f}",
            m=member_name,
            dt=delta_t,
            fp=f_p,
            drh=delta_rh,
            fsw=f_sw,
        )
        tasks.append((i, member_root, delta_t, f_p, delta_rh, f_sw, input_meteo_dir))

    if not tasks:
        logger.info("No members to build (all exist and overwrite is False).")
        logger.info("Prior ensemble completed under: {root}", root=str(prior_root))
        return

    workers = pick_max_workers(max_workers, fallback=params.ensemble_size, limit=len(tasks))
    logger.info("Building {} member(s) with max_workers={}", len(tasks), workers)

    if workers <= 1:
        for i, member_root, delta_t, f_p, delta_rh, f_sw, src_dir in tasks:
            _build_member(
                member_idx=i,
                member_root=member_root,
                start=start,
                end=end,
                delta_t=delta_t,
                f_p=f_p,
                delta_rh=delta_rh,
                f_sw=f_sw,
                random_seed=params.random_seed,
                input_meteo_dir=src_dir,
            )
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _build_member,
                    i,
                    member_root,
                    start,
                    end,
                    delta_t,
                    f_p,
                    delta_rh,
                    f_sw,
                    params.random_seed,
                    src_dir,
                ): f"member_{i:03d}"
                for i, member_root, delta_t, f_p, delta_rh, f_sw, src_dir in tasks
            }
            for fut in cf.as_completed(futs):
                name = futs[fut]
                try:
                    fut.result()
                    logger.info("[{}] written", name)
                except Exception as e:
                    logger.error("[{}] failed: {}", name, e)
                    raise

    logger.info("Prior ensemble completed under: {root}", root=str(prior_root))


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prior meteo ensemble (CSV)")
    p.add_argument("--input-meteo-dir", required=True, type=Path)
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--max-workers", type=int, default=None, help="Max workers for member build (overrides MAX_WORKERS env)")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        # Configure logger to match launcher formatting (green timestamp | level | message)
        logger.remove()
        logger.add(
            sys.stderr,
            level=args.log_level,
            enqueue=True,
            colorize=True,
            format=LOGURU_FORMAT,
        )

        # Apply environment (suppress GDAL warnings by setting GDAL_DATA/PROJ_LIB where possible)
        try:
            setup_dir = infer_setup_dir_from_project(args.project_dir)
            setup_yaml = find_setup_yaml(setup_dir)
            apply_env_from_project(setup_yaml)
        except Exception:
            # Best-effort: continue even if project env section is missing
            pass
        ensure_gdal_proj_from_conda()
        apply_numeric_thread_defaults()
        # Quiet GDAL debug output; avoid importing osgeo to prevent early warnings
        os.environ.setdefault("CPL_DEBUG", "OFF")

        build_prior_ensemble(
            args.input_meteo_dir,
            args.project_dir,
            args.step_dir,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
        )
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
