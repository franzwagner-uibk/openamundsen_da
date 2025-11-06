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

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from rasterio.mask import mask as rio_mask

from openamundsen_da.core.env import ensure_gdal_proj_from_conda
from openamundsen_da.core.constants import LOGURU_FORMAT


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


def _parse_date_from_filename(p: Path) -> datetime:
    """Extract date from file name formats like snowdepth_daily_YYYY-MM-DDT....tif.

    Tries ISO 'YYYY-MM-DD' first; falls back to first contiguous 8 digits.
    """
    stem = p.stem
    m = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d")
    m2 = re.search(r"(\d{8})", stem)
    if m2:
        return datetime.strptime(m2.group(1), "%Y%m%d")
    raise ValueError(f"Could not parse date from raster name: {p.name}")


def _member_id_from_results_dir(results_dir: Path) -> str:
    """Return member ID like 'member_001' given results dir .../member_001/results"""
    parent = Path(results_dir).parent
    return parent.name


def _find_member_raster(results_dir: Path, variable: Variable, date: datetime) -> Path:
    """Locate the raster file for the given date and variable in a member results dir.

    Expected patterns:
    - snowdepth_daily_YYYY-MM-DDT*.tif for variable='hs'
    - swe_daily_YYYY-MM-DDT*.tif      for variable='swe'
    """
    prefix = "snowdepth_daily_" if variable == "hs" else "swe_daily_"
    patt = f"{prefix}{date.strftime('%Y-%m-%d')}T*.tif"
    matches = sorted(Path(results_dir).glob(patt))
    if not matches:
        raise FileNotFoundError(f"No raster found matching {patt} in {results_dir}")
    return matches[0]


def _read_masked_array(raster_path: Path, aoi_path: Path) -> Tuple[np.ma.MaskedArray, dict]:
    """Read raster and mask by AOI geometry; return masked array and raster profile."""
    gdf = gpd.read_file(aoi_path)
    if len(gdf) != 1:
        raise ValueError(f"AOI must contain exactly one feature (got {len(gdf)})")
    with rasterio.open(raster_path) as src:
        # Reproject AOI if needed
        if gdf.crs is None:
            raise ValueError("AOI has no CRS; unable to align with raster")
        if src.crs is None:
            raise ValueError("Raster has no CRS; unable to align with AOI")
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        shapes: Iterable = gdf.geometry
        data, _ = rio_mask(src, shapes, crop=True, nodata=src.nodata, filled=False)
        arr = np.ma.array(data[0], copy=False)
        profile = src.profile
    return arr, profile


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
    p = 1.0 / (1.0 + np.exp(-k * dx))
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

    raster_path = _find_member_raster(Path(results_dir), variable, date)
    arr, _profile = _read_masked_array(raster_path, Path(aoi_path))

    if method == "depth_threshold":
        n_valid, n_snow, scf = _scf_depth_threshold(arr, float(params.h0))
    elif method == "logistic":
        n_valid, scf = _scf_logistic(arr, float(params.h0), float(params.k))
        n_snow = int(round(scf * n_valid))  # pseudo-count for reporting only
    else:
        raise ValueError(f"Unknown method: {method}")

    member_id = _member_id_from_results_dir(Path(results_dir))
    return {
        "date": date.strftime("%Y-%m-%d"),
        "member_id": member_id,
        "variable": variable,
        "method": method,
        "h0": float(params.h0),
        "k": float(params.k),
        "n_valid": int(n_valid),
        "n_snow": int(n_snow),
        "scf": float(scf),
        "raster": Path(raster_path).name,
    }


def cli_main(argv: list[str] | None = None) -> int:
    """CLI for computing model SCF per member/date.

    Examples
    --------
    oa-da-model-scf \
      --member-results C:/.../member_001/results \
      --aoi examples/test-project/env/GMBA_Inventory_L8_15422.gpkg \
      --date 2017-12-10 \
      --variable hs \
      --method logistic \
      --h0 0.05 --k 80
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-model-scf",
        description=(
            "Compute model-derived Snow Cover Fraction (SCF) from openAMUNDSEN "
            "daily outputs (HS/SWE) within an AOI for one member/date."
        ),
    )
    parser.add_argument("--member-results", type=Path, required=True, help="Path to member results directory")
    parser.add_argument("--aoi", type=Path, required=True, help="Path to single-feature AOI vector file")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD")
    parser.add_argument("--variable", type=str, default="hs", choices=["hs", "swe"], help="Use HS or SWE raster")
    parser.add_argument("--method", type=str, default="depth_threshold", choices=["depth_threshold", "logistic"], help="SCF operator")
    parser.add_argument("--h0", type=float, default=None, help="Threshold/midpoint h0 (units of variable)")
    parser.add_argument("--k", type=float, default=None, help="Logistic slope k (1/units of variable)")
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

    # Parameters
    prm = SCFParams()
    if args.h0 is not None:
        prm.h0 = float(args.h0)
    if args.k is not None:
        prm.k = float(args.k)
    if args.method == "depth_threshold":
        # k is unused; warn if provided
        if args.k is not None:
            logger.warning("--k provided but unused for depth_threshold method")

    # Compute SCF
    try:
        out = compute_model_scf(
            results_dir=Path(args.member_results),
            aoi_path=Path(args.aoi),
            date=dt,
            variable=args.variable,  # type: ignore[arg-type]
            method=("logistic" if args.method == "logistic" else "depth_threshold"),  # type: ignore[arg-type]
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


if __name__ == "__main__":
    sys.exit(cli_main())

