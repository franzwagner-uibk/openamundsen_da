"""
openamundsen_da.observer.satellite_scf

Minimal observation processing for MODIS/Terra Snow Cover Daily (MOD10A1
Collection 6/6.1). Assumes the product was exported as a single-band GeoTIFF
containing the layer `NDSI_Snow_Cover` scaled to 0..100 and with nodata already
applied.

Reads one preprocessed raster, masks it by a single AOI polygon (field
`region_id`), applies a fixed/configurable NDSI threshold, and writes one CSV
row (`date,region_id,scf`).

Notes
- Inputs must be preprocessed (CRS aligned, QA filtering, mosaicking, etc.).
- Only one raster and one region per run; batch support will come later.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from loguru import logger

from openamundsen_da.core.constants import (
    OBS_DIR_NAME,
    SCF_BLOCK,
    SCF_NDSI_THRESHOLD,
    SCF_REGION_ID_FIELD,
)
from openamundsen_da.core.env import ensure_gdal_proj_from_conda
from openamundsen_da.io.paths import find_step_yaml


def _extract_yyyymmdd(p: Path) -> datetime:
    """Extract first YYYYMMDD occurrence from filename and return a date.

    Raises ValueError if no valid date can be found or parsed.
    """
    m = re.search(r"(\d{8})", p.stem)
    if not m:
        raise ValueError(f"No YYYYMMDD found in raster name: {p.name}")
    s = m.group(1)
    try:
        return datetime.strptime(s, "%Y%m%d")
    except Exception as e:
        raise ValueError(f"Invalid YYYYMMDD in raster name: {p.name}") from e


def _read_step_config(step_dir: Path) -> dict:
    """Best-effort read of the step YAML to pick SCF overrides.

    Returns an empty dict on failure; only top-level block `scf` is used.
    """
    try:
        yml = find_step_yaml(step_dir)
    except Exception:
        return {}
    try:
        import ruamel.yaml as _yaml

        y = _yaml.YAML(typ="safe")
        with Path(yml).open("r", encoding="utf-8") as f:
            return y.load(f) or {}
    except Exception:
        return {}


def _compute_scf(arr: np.ma.MaskedArray, threshold: float) -> tuple[int, int, float]:
    """Compute N_valid, N_snow, SCF given a masked NDSI array and threshold.

    Only 0..100 values are considered valid; masked/NaN/out-of-range are invalid.
    Returns (N_valid, N_snow, scf).
    """
    data = np.ma.array(arr, copy=False)
    # valid: not masked, finite, and within expected NDSI range
    valid = (~data.mask) & np.isfinite(data) & (data >= 0) & (data <= 100)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("AOI contains no valid NDSI pixels (N_valid=0)")
    snow = valid & (data > threshold)
    n_snow = int(snow.sum())
    scf = float(n_snow / n_valid)
    return n_valid, n_snow, scf


def run_observation_processing(
    input_raster: Path,
    region_path: Path,
    output_csv: Path,
    *,
    ndsi_threshold: float = 40.0,
    region_id_field: str = "region_id",
) -> None:
    """Compute single-region SCF from one MOD10A1 NDSI raster and write CSV.

    Parameters
    ----------
    input_raster : Path
        Path to a MODIS/Terra MOD10A1 Collection 6/6.1 raster exported to
        GeoTIFF containing band `NDSI_Snow_Cover`, scaled 0..100 with nodata.
    region_path : Path
        Path to a vector file containing a single AOI polygon with field
        `region_id` (or `region_id_field`).
    output_csv : Path
        Path to output CSV file; parent directories are created if needed.
    ndsi_threshold : float, optional
        NDSI threshold for snow classification (default 40.0).
    region_id_field : str, optional
        Name of the region identifier field in AOI (default 'region_id').
    """
    ensure_gdal_proj_from_conda()  # help GDAL/PROJ locate data on Windows/conda

    input_raster = Path(input_raster)
    region_path = Path(region_path)
    output_csv = Path(output_csv)

    # Load AOI, enforce single feature and required field
    gdf = gpd.read_file(region_path)
    if len(gdf) != 1:
        raise ValueError(f"AOI must contain exactly one feature (got {len(gdf)})")
    if region_id_field not in gdf.columns:
        raise KeyError(f"AOI missing field '{region_id_field}'")
    region_id = str(gdf.iloc[0][region_id_field])

    # Open raster and validate CRS alignment
    with rasterio.open(input_raster) as src:
        if gdf.crs is None or src.crs is None or gdf.crs != src.crs:
            raise ValueError("CRS mismatch or missing CRS between raster and AOI")
        # Rasterize-mask by AOI geometry, preserve mask (masked array)
        shapes: Iterable = gdf.geometry
        data, _ = rio_mask(src, shapes, crop=True, nodata=src.nodata, filled=False)
        band = np.ma.array(data[0], copy=False)  # first band

    # Compute SCF and date
    n_valid, n_snow, scf = _compute_scf(band, float(ndsi_threshold))
    dt = _extract_yyyymmdd(input_raster)

    # Prepare output directory and CSV content
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "date": [dt.strftime("%Y-%m-%d")],
        "region_id": [region_id],
        "scf": [scf],
    })
    df.to_csv(output_csv, index=False)

    # Single concise log line
    logger.info(
        f"SCF | raster={input_raster.name} region={region_id} valid={n_valid} snow={n_snow} scf={scf:.3f} -> {output_csv.name}"
    )


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry for single-image, single-region SCF extraction.

    Examples
    --------
    oa-da-scf --raster NDSI_Snow_Cover_20250315.tif \
              --region region.gpkg \
              --step-dir path/to/step
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-scf",
        description="Compute SCF from one MOD10A1 NDSI raster and one AOI",
    )
    parser.add_argument("--raster", required=True, type=Path, help="Path to NDSI GeoTIFF (0..100)")
    parser.add_argument("--region", required=True, type=Path, help="Path to AOI vector with 'region_id'")
    parser.add_argument("--step-dir", type=Path, help="Step directory to place CSV under 'obs' and read config")
    parser.add_argument("--output", type=Path, help="Explicit output CSV path (overrides --step-dir)")
    parser.add_argument("--ndsi-threshold", type=float, help="Override NDSI threshold (default 40)")

    args = parser.parse_args(argv)

    # Resolve config overrides from step YAML if provided
    ndsi_thr = args.ndsi_threshold if args.ndsi_threshold is not None else 40.0
    region_field = "region_id"
    out_csv: Path | None = args.output

    if args.step_dir and out_csv is None:
        # Derive output path under <step_dir>/obs with date from raster name
        dt = _extract_yyyymmdd(Path(args.raster))
        out_dir = Path(args.step_dir) / OBS_DIR_NAME
        out_csv = out_dir / f"obs_scf_MOD10A1_{dt.strftime('%Y%m%d')}.csv"

        # Load SCF overrides from step YAML
        cfg = _read_step_config(Path(args.step_dir))
        scf_cfg = (cfg or {}).get(SCF_BLOCK) or {}
        ndsi_thr = float(scf_cfg.get(SCF_NDSI_THRESHOLD, ndsi_thr))
        region_field = str(scf_cfg.get(SCF_REGION_ID_FIELD, region_field))

    if out_csv is None:
        parser.error("Either --step-dir or --output must be provided")

    try:
        run_observation_processing(
            input_raster=Path(args.raster),
            region_path=Path(args.region),
            output_csv=out_csv,
            ndsi_threshold=ndsi_thr,
            region_id_field=region_field,
        )
        return 0
    except Exception as e:
        logger.error(f"SCF processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())
