"""openamundsen_da.methods.h_of_x.model_scf

Model-based Snow Cover Fraction (SCF) operator H(x).

This module derives an areal SCF in a given AOI from openAMUNDSEN
distributed outputs (snow depth/HS or SWE) for a single date and member.

Two supported per-pixel operators (both averaged over the AOI):

- Depth threshold (deterministic)
  I_snow = 1 if X > h0 else 0, SCF = mean(I_snow)

- Logistic (probabilistic)
  p_snow = 1 / (1 + exp(-k * (X - h0))), SCF = mean(p_snow)

Where X is either snow depth (HS, meters) or SWE (e.g., mm). The parameters
"h0" and "k" are interpreted in the same units as X and its inverse.

Notes
- This operator intentionally mirrors simple satellite SCF processing
  (thresholding) but stays flexible via the logistic variant for stability
  in data assimilation.
- AOI geometry is reprojected to the raster CRS if needed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Tuple
import concurrent.futures as cf

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from rasterio.mask import mask as rio_mask

from openamundsen_da.core.env import ensure_gdal_proj_from_conda, _read_yaml_file
from openamundsen_da.core.constants import (
    LOGURU_FORMAT,
    VAR_HS,
    VAR_SWE,
    HOFX_BLOCK,
    HOFX_METHOD,
    HOFX_VARIABLE,
    HOFX_PARAMS,
    HOFX_PARAM_H0,
    HOFX_PARAM_K,
    DA_BLOCK,
)
from openamundsen_da.io.paths import (
    find_member_daily_raster,
    member_id_from_results_dir,
    find_project_yaml,
    read_step_config,
    list_member_dirs,
    open_loop_dir,
)
from openamundsen_da.util.aoi import read_single_aoi
from openamundsen_da.util.stats import sigmoid


Variable = Literal["hs", "swe"]
Method = Literal["depth_threshold", "logistic"]


@dataclass
class SCFParams:
    """SCF operator parameters.

    Attributes
    ----------
    h0 : float
        Midpoint/threshold in the same units as the input variable (m for HS,
        mm for SWE if your SWE raster uses mm). For the threshold method, this
        is the cut-off; for logistic, it's the 50% point.
    k : float
        Slope in 1/units of X. Larger k means a sharper transition around h0.
        Ignored by the depth_threshold method.
    """

    h0: float = 0.05
    k: float = 80.0


def _parse_hofx_block(hofx: dict[str, Any]) -> tuple[str, str, SCFParams]:
    """Return H(x) settings from a config block, applying defaults where keys are missing."""
    method = str(hofx.get(HOFX_METHOD, "depth_threshold"))
    variable = str(hofx.get(HOFX_VARIABLE, "hs"))
    params = SCFParams()
    plist = hofx.get(HOFX_PARAMS) or {}
    if isinstance(plist, dict):
        if HOFX_PARAM_H0 in plist:
            params.h0 = float(plist[HOFX_PARAM_H0])
        if HOFX_PARAM_K in plist:
            params.k = float(plist[HOFX_PARAM_K])
    return method, variable, params


def load_hofx_from_project(project_dir: Path) -> tuple[str, str, SCFParams]:
    """Read required H(x) configuration from project.yml."""
    proj = find_project_yaml(project_dir)
    cfg = _read_yaml_file(proj) or {}
    da = cfg.get(DA_BLOCK, {}) if isinstance(cfg, dict) else {}
    hofx = da.get(HOFX_BLOCK) or cfg.get(HOFX_BLOCK)
    if not isinstance(hofx, dict):
        hofx = {}
    if not hofx:
        raise ValueError(f"Missing '{DA_BLOCK}.{HOFX_BLOCK}' section in {proj}")
    return _parse_hofx_block(hofx)


def _read_masked_array(raster_path: Path, aoi_path: Path) -> np.ma.MaskedArray:
    """Read raster and mask by AOI geometry; return masked array."""
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS; unable to align with AOI")
        gdf, _ = read_single_aoi(aoi_path, required_field="region_id", to_crs=src.crs)
        shapes: Iterable = gdf.geometry
        data, _ = rio_mask(src, shapes, crop=True, nodata=src.nodata, filled=False)
        arr = np.ma.array(data[0], copy=False)
    return arr


def _valid_mask(x: np.ma.MaskedArray) -> np.ndarray:
    """Return boolean mask of valid (non-masked, finite) pixels."""
    data = np.ma.array(x, copy=False)
    return (~data.mask) & np.isfinite(data)


def _scf_depth_threshold(x: np.ma.MaskedArray, h0: float) -> Tuple[int, int, float]:
    """Compute SCF using deterministic threshold (I = 1 if x > h0)."""
    valid = _valid_mask(x)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("AOI contains no valid pixels for SCF computation")
    snow = valid & (x > h0)
    n_snow = int(snow.sum())
    scf = float(n_snow) / float(n_valid)
    return n_valid, n_snow, scf


def _scf_logistic(x: np.ma.MaskedArray, h0: float, k: float) -> Tuple[int, float]:
    """Compute SCF using logistic probability: mean(sigmoid(k * (x - h0)))."""
    valid = _valid_mask(x)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("AOI contains no valid pixels for SCF computation")
    dx = np.clip((x - h0), a_min=-1e6, a_max=1e6)
    p = sigmoid(k * dx)
    scf = float(p[valid].mean())
    return n_valid, scf


def compute_model_scf(
    *,
    results_dir: Path,
    aoi_path: Path,
    date: datetime,
    variable: Variable = "hs",
    method: Method = "depth_threshold",
    params: SCFParams | None = None,
) -> dict:
    """Compute model SCF for one member/date within an AOI.

    Parameters
    ----------
    results_dir : Path
        Path to member results directory (e.g., .../member_001/results).
    aoi_path : Path
        Vector file with single AOI polygon; reprojected to raster CRS if needed.
    date : datetime
        Date for which to read the raster (daily outputs expected).
    variable : {"hs","swe"}
        Use snow depth (HS) or SWE raster for computation.
    method : {"depth_threshold","logistic"}
        SCF operator: deterministic threshold or logistic probability.
    params : SCFParams, optional
        Parameters for the operator; h0 in units of the variable, k in 1/units.

    Returns
    -------
    dict
        Dict with keys: date, member_id, variable, method, h0, k, n_valid, scf.
    """
    ensure_gdal_proj_from_conda()
    params = params or SCFParams()

    var = variable if variable in (VAR_HS, VAR_SWE) else VAR_HS
    raster_path = find_member_daily_raster(Path(results_dir), var, date.strftime("%Y-%m-%d"))
    arr = _read_masked_array(raster_path, Path(aoi_path))

    if method == "depth_threshold":
        n_valid, n_snow, scf = _scf_depth_threshold(arr, float(params.h0))
    elif method == "logistic":
        n_valid, scf = _scf_logistic(arr, float(params.h0), float(params.k))
        n_snow = int(round(scf * n_valid))  # pseudo-count for reporting only
    else:
        raise ValueError(f"Unknown method: {method}")

    member_id = member_id_from_results_dir(Path(results_dir))
    return {
        "date": date.strftime("%Y-%m-%d"),
        "member_id": member_id,
        "variable": var,
        "method": method,
        "h0": float(params.h0),
        "k": float(params.k),
        "n_valid": int(n_valid),
        "n_snow": int(n_snow),
        "scf": float(scf),
        "raster": Path(raster_path).name,
    }


def compute_member_scf_daily(
    *,
    project_dir: Path,
    results_dir: Path,
    aoi_path: Path,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Compute daily AOI-mean SCF for one member over a date range.

    Uses the same H(x) configuration as assimilation (data_assimilation.h_of_x)
    and reuses :func:`compute_model_scf` for each day where a daily raster is
    available. Missing rasters for particular days are skipped.

    Returns a DataFrame with columns ``time`` (datetime) and ``scf`` sorted by
    time.
    """
    method, variable, params = load_hofx_from_project(Path(project_dir))

    # Normalize variable name for internal use
    var = variable if variable in (VAR_HS, VAR_SWE) else VAR_HS

    # Build daily date range (inclusive, based on calendar days)
    start_day = datetime(start.year, start.month, start.day)
    end_day = datetime(end.year, end.month, end.day)
    if end_day < start_day:
        return pd.DataFrame(columns=["time", "scf"])
    dates = pd.date_range(start_day, end_day, freq="D").to_pydatetime()

    times: list[datetime] = []
    scfs: list[float] = []
    for dt in dates:
        try:
            out = compute_model_scf(
                results_dir=Path(results_dir),
                aoi_path=Path(aoi_path),
                date=dt,
                variable=var,  # type: ignore[arg-type]
                method=method,  # type: ignore[arg-type]
                params=params,
            )
        except FileNotFoundError:
            # No daily raster for this date -> skip
            continue
        except Exception as exc:
            logger.warning("SCF daily computation failed for {} at {}: {}", results_dir, dt.date(), exc)
            continue
        times.append(dt)
        scfs.append(float(out["scf"]))

    if not times:
        return pd.DataFrame(columns=["time", "scf"])
    df = pd.DataFrame({"time": times, "scf": scfs})
    return df.sort_values("time")


def _compute_member_scf_for_step_worker(
    project_dir: str,
    aoi_path: str,
    results_dir: str,
    start_iso: str,
    end_iso: str,
    overwrite: bool,
) -> tuple[str, bool]:
    """Worker: compute SCF daily series for a single member results dir.

    Returns (member_name, created) where created indicates whether a CSV was
    written (False when skipped due to overwrite=False and existing output).
    """
    res_dir = Path(results_dir)
    out_csv = res_dir / "point_scf_aoi.csv"
    if out_csv.exists() and not overwrite:
        return res_dir.parent.name, False

    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    df = compute_member_scf_daily(
        project_dir=Path(project_dir),
        results_dir=res_dir,
        aoi_path=Path(aoi_path),
        start=start,
        end=end,
    )
    if df.empty:
        return res_dir.parent.name, False

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return res_dir.parent.name, True


def compute_step_scf_daily_for_all_members(
    *,
    project_dir: Path,
    season_dir: Path,
    step_dir: Path,
    aoi_path: Path,
    max_workers: int = 4,
    overwrite: bool = False,
) -> None:
    """Compute daily model SCF for all prior members in a step.

    For the given ``step_dir``, this function:
    - Reads the step start/end dates from its YAML.
    - Discovers prior ensemble members (including open_loop when present).
    - In parallel across members, computes daily AOI-mean SCF time series
      using :func:`compute_member_scf_daily`.
    - Writes the result to ``<member>/results/point_scf_aoi.csv`` for each
      member, which can then be used by the season plotting utilities via
      ``var_col='scf'`` and ``station='point_scf_aoi.csv'``.

    Existing CSVs are skipped unless ``overwrite=True``.
    """
    step_dir = Path(step_dir)
    project_dir = Path(project_dir)
    season_dir = Path(season_dir)
    aoi_path = Path(aoi_path)

    cfg = read_step_config(step_dir) or {}
    try:
        s_val = cfg.get("start_date")
        e_val = cfg.get("end_date")
        start = datetime.fromisoformat(str(s_val))
        end = datetime.fromisoformat(str(e_val))
    except Exception as exc:
        raise ValueError(f"Missing or invalid start_date/end_date in step config for {step_dir}") from exc

    prior_root = step_dir / "ensembles" / "prior"
    if not prior_root.is_dir():
        raise FileNotFoundError(f"No prior ensemble found under {prior_root}")

    members = list_member_dirs(step_dir / "ensembles", "prior")
    if not members:
        logger.warning("No prior members found under {}; skipping SCF computation.", step_dir)
        return

    # Include open_loop if present
    try:
        ol_dir = open_loop_dir(step_dir)
        if ol_dir.is_dir():
            members = [ol_dir] + members
    except Exception:
        pass

    # Use member results directories
    jobs: list[tuple[str, str, str, str, str, bool]] = []
    all_exist = True
    for mdir in members:
        res_dir = Path(mdir) / "results"
        if not res_dir.is_dir():
            all_exist = False
            continue
        out_csv = res_dir / "point_scf_aoi.csv"
        if not out_csv.is_file():
            all_exist = False
        jobs.append(
            (
                str(project_dir),
                str(aoi_path),
                str(res_dir),
                start.isoformat(),
                end.isoformat(),
                bool(overwrite),
            )
        )

    # If every member already has SCF output and overwrite=False, skip work.
    if jobs and all_exist and not overwrite:
        logger.info(
            "SCF daily series already present for all members in {}; overwrite=False -> skipping SCF computation.",
            step_dir.name,
        )
        return

    if not jobs:
        logger.warning("No member results directories found for {}; skipping SCF computation.", step_dir)
        return

    n_workers = max(1, min(int(max_workers or 1), len(jobs)))
    logger.info(
        "Computing daily model SCF for {} member(s) in {} using {} worker(s) ...",
        len(jobs),
        step_dir.name,
        n_workers,
    )

    created = 0
    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_compute_member_scf_for_step_worker, *args) for args in jobs]
        for fut in cf.as_completed(futures):
            try:
                name, did_create = fut.result()
                if did_create:
                    created += 1
            except Exception as exc:
                logger.warning("SCF computation failed for a member in {}: {}", step_dir, exc)

    logger.info(
        "Model SCF daily series written for {} / {} member(s) in {}",
        created,
        len(jobs),
        step_dir.name,
    )


def cli_main(argv: list[str] | None = None) -> int:
    """CLI for computing model SCF per member/date.

    Examples
    --------
    oa-da-model-scf \
      --project-dir C:/.../examples/test-project \
      --member-results C:/.../member_001/results \
      --aoi examples/test-project/env/GMBA_Inventory_L8_15422.gpkg \
      --date 2017-12-10
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-model-scf",
        description=(
            "Compute model-derived Snow Cover Fraction (SCF) from openAMUNDSEN "
            "daily outputs (HS/SWE) within an AOI for one member/date using the "
            "project-level H(x) configuration."
        ),
    )
    parser.add_argument("--project-dir", type=Path, required=True, help="Project root containing project.yml with data_assimilation.h_of_x")
    parser.add_argument("--member-results", type=Path, required=True, help="Path to member results directory")
    parser.add_argument("--aoi", type=Path, required=True, help="Path to single-feature AOI vector file")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path")
    parser.add_argument("--region-id-field", type=str, default="region_id", help="AOI field name for region_id")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level (INFO, DEBUG, ...)")

    args = parser.parse_args(argv)

    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    # Parse date
    try:
        dt = datetime.strptime(args.date, "%Y-%m-%d")
    except Exception as e:
        logger.error(f"Invalid --date format (expected YYYY-MM-DD): {args.date}")
        return 2

    try:
        method, variable, prm = load_hofx_from_project(Path(args.project_dir))
    except Exception as e:
        logger.error("Failed to read H(x) configuration: {}", e)
        return 2

    # Compute SCF
    try:
        out = compute_model_scf(
            results_dir=Path(args.member_results),
            aoi_path=Path(args.aoi),
            date=dt,
            variable=variable,  # type: ignore[arg-type]
            method=("logistic" if method == "logistic" else "depth_threshold"),  # type: ignore[arg-type]
            params=prm,
        )
    except Exception as e:
        logger.error(f"Model SCF computation failed: {e}")
        return 1

    # Attempt to extract region_id for reporting (optional)
    region_id = None
    try:
        gdf = gpd.read_file(args.aoi)
        if len(gdf) == 1 and args.region_id_field in gdf.columns:
            region_id = str(gdf.iloc[0][args.region_id_field])
    except Exception:
        pass

    # Prepare CSV
    df = pd.DataFrame({
        "date": [out["date"]],
        "member_id": [out["member_id"]],
        "region_id": [region_id if region_id is not None else ""],
        "variable": [out["variable"]],
        "method": [out["method"]],
        "h0": [out["h0"]],
        "k": [out["k"]],
        "n_valid": [out["n_valid"]],
        "scf_model": [round(out["scf"], 4)],
        "raster": [out["raster"]],
    })

    # Output path default: <member_results>/model_scf_YYYYMMDD.csv
    out_csv = args.output
    if out_csv is None:
        out_csv = Path(args.member_results) / f"model_scf_{dt.strftime('%Y%m%d')}.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    logger.info(
        "MODEL_SCF | raster={} member={} region={} var={} method={} h0={} k={} valid={} scf={:.3f} -> {}".format(
            out["raster"], out["member_id"], region_id if region_id else "", out["variable"], out["method"], out["h0"], out["k"], out["n_valid"], out["scf"], out_csv.name
        )
    )
    return 0


def cli_season_daily(argv: list[str] | None = None) -> int:
    """CLI: compute daily model SCF for all members and steps in a season.

    Example
    -------
    oa-da-model-scf-season-daily \\
      --project-dir C:/.../examples/test-project \\
      --season-dir C:/.../propagation/season_2017-2018 \\
      --aoi C:/.../env/GMBA_Inventory_L8_15422.gpkg \\
      --max-workers 8
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-model-scf-season-daily",
        description=(
            "Compute daily model SCF (AOI-mean) for all prior members in each "
            "step of a season, writing point_scf_aoi.csv per member."
        ),
    )
    parser.add_argument("--project-dir", type=Path, required=True, help="Project root containing project.yml")
    parser.add_argument("--season-dir", type=Path, required=True, help="Season root containing step_* directories")
    parser.add_argument("--aoi", type=Path, required=True, help="Single-feature AOI vector (same as used by assimilation)")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers per step")
    parser.add_argument("--overwrite", action="store_true", help="Recompute SCF even if point_scf_aoi.csv exists")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level (INFO, DEBUG, ...)")

    args = parser.parse_args(argv)

    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    season_dir = Path(args.season_dir)
    steps = sorted(p for p in season_dir.glob("step_*") if p.is_dir())
    if not steps:
        logger.error("No step_* directories found under {}", season_dir)
        return 1

    logger.info("Computing daily model SCF for season: {} ({} step(s))", season_dir.name, len(steps))
    for step in steps:
        try:
            compute_step_scf_daily_for_all_members(
                project_dir=Path(args.project_dir),
                season_dir=season_dir,
                step_dir=step,
                aoi_path=Path(args.aoi),
                max_workers=int(args.max_workers or 4),
                overwrite=bool(args.overwrite),
            )
        except Exception as exc:
            logger.error("SCF daily computation failed for step {}: {}", step.name, exc)
            return 2
    logger.info("Season-wide model SCF daily computation complete for {}", season_dir)
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
