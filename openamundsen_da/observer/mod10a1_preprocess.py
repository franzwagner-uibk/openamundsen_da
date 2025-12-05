"""openamundsen_da.observer.mod10a1_preprocess

Purpose
- Convert MODIS/Terra MOD10A1 (Collection 6/6.1) HDF files into observation‑ready
  artifacts for the data‑assimilation workflow.

Per scene outputs
- `NDSI_Snow_Cover_YYYYMMDD.tif` (GeoTIFF) — NDSI_Snow_Cover band reprojected to
  the target CRS, optionally clipped to a single AOI.
- `NDSI_Snow_Cover_YYYYMMDD_class.tif` (Byte GeoTIFF) — simple 3‑class product
  encoded as 0=invalid, 1=no snow, 2=snow (threshold on 0..100 NDSI).
- `scf_summary.csv` (season‑level table) — one row per accepted day with
  date, region_id, SCF (0..1), cloud_fraction (0..1), and source filename.

Notes
- Assumes single‑region AOI (exactly one polygon feature) and preprocessed inputs.
- Cloud screening is optional via `--max-cloud-fraction` using value 200 pixels.
- Designed to feed directly into `openamundsen_da.observer.satellite_scf`.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

try:  # GDAL is required for HDF handling
    from osgeo import gdal  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    gdal = None  # type: ignore
    _GDAL_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - import side effect
    gdal.UseExceptions()
    _GDAL_IMPORT_ERROR = None

from openamundsen_da.core.constants import (
    MOD10A1_PRODUCT,
    MOD10A1_SDS_NAME,
    OBS_DIR_NAME,
)
from openamundsen_da.core.env import ensure_gdal_proj_from_conda



@dataclass(frozen=True)
class Mod10A1Meta:
    """Metadata parsed from a MOD10A1 filename."""

    product: str
    date: datetime
    tile: str | None


_MOD10A1_RE = re.compile(
    r"^(?P<product>[A-Z0-9]{7,})\.A(?P<year>\d{4})(?P<doy>\d{3})"
    r"(?:\.(?P<tile>h\d{2}v\d{2}))?",
    re.IGNORECASE,
)


def _parse_mod10a1_name(hdf_path: Path) -> Mod10A1Meta:
    """Parse product, acquisition date, and tile from a MOD10A1 filename.

    Parameters
    ----------
    hdf_path : Path
        Path to the MOD10A1 HDF file.

    Returns
    -------
    Mod10A1Meta
        Parsed product name, acquisition date, and optional h/v tile string.
    """

    match = _MOD10A1_RE.match(hdf_path.name)
    if not match:
        raise ValueError(f"Filename does not look like MOD10A1: {hdf_path.name}")

    product = match.group("product").upper()
    if product != MOD10A1_PRODUCT:
        raise ValueError(f"Unsupported product '{product}' (expected {MOD10A1_PRODUCT})")

    year = int(match.group("year"))
    doy = int(match.group("doy"))
    date = datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")
    return Mod10A1Meta(product=product, date=date, tile=match.group("tile"))


def _list_hdf_files(root: Path, recursive: bool) -> list[Path]:
    """Return sorted list of MOD10A1 HDF files under a root directory.

    Parameters
    ----------
    root : Path
        Input directory to scan.
    recursive : bool
        If True, search subdirectories; otherwise only the top level.
    """

    if recursive:
        files = root.rglob("*.hdf")
    else:
        files = root.glob("*.hdf")
    return sorted(p for p in files if p.is_file())


def _find_ndsi_subdataset(hdf_path: Path) -> str:
    """Locate the `NDSI_Snow_Cover` subdataset within the HDF container."""

    _ensure_gdal()
    ds = gdal.Open(str(hdf_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open HDF file: {hdf_path}")
    try:
        for name, _desc in ds.GetSubDatasets() or []:
            if name.endswith(f":{MOD10A1_SDS_NAME}"):
                return name
    finally:
        ds = None
    raise RuntimeError(f"Subdataset '{MOD10A1_SDS_NAME}' not found in {hdf_path.name}")


def _build_output_path(output_root: Path, season_label: str, when: datetime) -> Path:
    """Derive output path for the converted GeoTIFF (flat per season)."""

    season_dir = output_root / season_label
    season_dir.mkdir(parents=True, exist_ok=True)
    return season_dir / f"{MOD10A1_SDS_NAME}_{when.strftime('%Y%m%d')}.tif"


def _warp_ndsi(
    subdataset: str,
    destination: Path,
    *,
    target_epsg: int,
    resolution: float | None,
    bounds: tuple[float, float, float, float] | None,
    aoi_path: Path | None,
    overwrite: bool,
) -> None:
    """Reproject and optionally crop the NDSI subdataset into *destination*.

    Uses nearest neighbor resampling and preserves or sets nodata=255.
    Envelope cropping uses AOI bounds in target CRS; cutline performs an
    exact polygon clip.
    """

    if destination.exists() and not overwrite:
        logger.info("Skipping existing {} (use --overwrite to replace)", destination)
        return

    _ensure_gdal()
    src = gdal.Open(subdataset, gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"Could not open subdataset: {subdataset}")
    try:
        band = src.GetRasterBand(1)
        nodata = band.GetNoDataValue()
    finally:
        src = None

    warp_kwargs: dict[str, object] = {
        "dstSRS": f"EPSG:{target_epsg}",
        "format": "GTiff",
        "resampleAlg": "near",
        "multithread": True,
        "creationOptions": ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2"],
        "dstNodata": nodata if nodata is not None else 255,
    }

    if nodata is not None:
        warp_kwargs["srcNodata"] = nodata

    if resolution is not None:
        warp_kwargs.update({"xRes": float(resolution), "yRes": float(resolution), "targetAlignedPixels": True})

    if bounds is not None:
        # Fast rectangle crop using AOI bounds in target CRS.
        warp_kwargs.update({
            "outputBounds": bounds,
            "outputBoundsSRS": f"EPSG:{target_epsg}",
        })
    elif aoi_path is not None:
        # Exact polygon clip.
        warp_kwargs.update({
            "cutlineDSName": str(aoi_path),
            "cropToCutline": True,
        })

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and overwrite:
        destination.unlink()

    options = gdal.WarpOptions(**warp_kwargs)
    result = gdal.Warp(str(destination), subdataset, options=options)
    if result is None:
        raise RuntimeError(f"gdal.Warp failed for {destination.name}")
    result = None


def convert_mod10a1_directory(
    input_dir: Path,
    output_root: Path,
    season_label: str,
    *,
    aoi_path: Path | None,
    target_epsg: int,
    resolution: float | None,
    use_envelope: bool,
    recursive: bool,
    overwrite: bool,
    max_cloud_fraction: float | None,
    ndsi_threshold: float,
    region_field: str,
) -> list[Path]:
    """Convert all MOD10A1 HDF files under `input_dir` to GeoTIFFs.

    Parameters
    ----------
    input_dir : Path
        Root directory containing MOD10A1 HDF files.
    output_root : Path
        Output root directory (season folder is created below this path).
    season_label : str
        Season folder name below `output_root`.
    aoi_path : Path | None
        AOI vector path (single feature) used for envelope or cutline clipping.
    target_epsg : int
        EPSG code of the output CRS.
    resolution : float | None
        Output pixel size in target CRS units (meters for UTM). If None, auto.
    use_envelope : bool
        If True, crop by AOI bounds; otherwise use exact polygon cutline.
    recursive : bool
        If True, search input directory recursively for HDF files.
    overwrite : bool
        If True, overwrite existing outputs.
    max_cloud_fraction : float | None
        Reject scenes with cloud fraction above this threshold (0..1).
    ndsi_threshold : float
        Threshold (0..100) to classify snow (NDSI > threshold).
    region_field : str
        AOI attribute name with the region identifier written to the summary.
    """

    _ensure_gdal()
    ensure_gdal_proj_from_conda()

    # Step 1: Discover input files
    logger.debug("Scanning {} for MOD10A1 HDF files (recursive={})", input_dir, recursive)
    hdf_files = _list_hdf_files(Path(input_dir), recursive)
    if not hdf_files:
        logger.warning("No MOD10A1 HDF files found in {}", input_dir)
        return []

    # Step 2: Prepare season directory and (optional) AOI context
    season_dir = output_root / season_label
    season_dir.mkdir(parents=True, exist_ok=True)
    summary_path = season_dir / "scf_summary.csv"

    bounds = None
    region_id = region_field
    aoi = Path(aoi_path) if aoi_path is not None else None
    if aoi:
        gdf = gpd.read_file(aoi)
        if gdf.empty:
            raise ValueError("AOI file does not contain geometries")
        if len(gdf) != 1:
            raise ValueError("AOI must contain exactly one feature")
        if region_field not in gdf.columns:
            raise KeyError(f"AOI missing field '{region_field}'")
        region_id = str(gdf.iloc[0][region_field])
        if use_envelope:
            if gdf.crs is None:
                raise ValueError("AOI CRS is undefined")
            gdf_target = gdf if (target_epsg is None or gdf.crs.to_epsg() == target_epsg) else gdf.to_crs(target_epsg)
            tb = gdf_target.total_bounds
            bounds = (float(tb[0]), float(tb[1]), float(tb[2]), float(tb[3]))
            logger.debug("AOI bounds in EPSG:{} -> {}", target_epsg, bounds)

    outputs: list[Path] = []
    # Step 3: Process each HDF -> NDSI, classify, summarize
    for hdf in hdf_files:
        try:
            meta = _parse_mod10a1_name(hdf)
        except ValueError as exc:
            logger.error("Skipping {}: {}", hdf.name, exc)
            continue

        try:
            subdataset = _find_ndsi_subdataset(hdf)
        except RuntimeError as exc:
            logger.error("Skipping {}: {}", hdf.name, exc)
            continue

        destination = _build_output_path(output_root, season_label, meta.date)
        try:  # Warp the NDSI subdataset to GeoTIFF (CRS/resolution/clip)
            _warp_ndsi(
                subdataset,
                destination,
                target_epsg=target_epsg,
                resolution=resolution,
                bounds=bounds,
                aoi_path=aoi if (aoi and not use_envelope) else None,
                overwrite=overwrite,
            )
        except Exception as exc:
            logger.error("Conversion failed for {}: {}", hdf.name, exc)
            continue

        try:  # Read array + metadata for cloud/SCF classification
            data, nodata, transform, projection = _read_ndsi_raster(destination)
        except Exception as exc:
            logger.error("Failed reading {}: {}", destination.name, exc)
            if destination.exists():
                destination.unlink(missing_ok=True)
            continue

        # Step 4: Compute cloud fraction and apply threshold if requested
        cloud_fraction = _cloud_fraction_from_data(data, nodata)
        if cloud_fraction is None:
            logger.warning(
                "Discarded {} because AOI contained no usable NDSI pixels (only nodata or QA codes)",
                destination.name,
            )
            if destination.exists():
                destination.unlink(missing_ok=True)
            continue

        if max_cloud_fraction is not None and cloud_fraction > max_cloud_fraction:
            logger.info(
                "Skipping {} due to cloud_fraction {:.3f} > {:.3f}",
                hdf.name,
                cloud_fraction,
                max_cloud_fraction,
            )
            if destination.exists():
                destination.unlink(missing_ok=True)
            continue

        # Step 5: Classify and compute SCF
        class_arr, scf, n_valid, n_snow = _classify_ndsi_data(data, nodata, ndsi_threshold)
        if scf is None:
            logger.warning(
                "Discarded {} because no valid NDSI pixels remained after filtering",
                destination.name,
            )
            if destination.exists():
                destination.unlink(missing_ok=True)
            continue

        class_path = destination.with_name(destination.stem + "_class.tif")
        try:  # Step 6: Write classification raster
            _write_classification_raster(
                class_path,
                class_arr,
                transform,
                projection,
                overwrite,
            )
        except Exception as exc:
            logger.error("Failed writing classification {}: {}", class_path.name, exc)
            if destination.exists():
                destination.unlink(missing_ok=True)
            if class_path.exists():
                class_path.unlink(missing_ok=True)
            continue

        try:  # Step 7: Update season summary table
            _update_scf_summary(
                summary_path,
                meta.date,
                region_id,
                scf,
                cloud_fraction,
                destination.name,
                n_valid,
                n_snow,
            )
        except Exception as exc:
            logger.error("Failed updating SCF summary for {}: {}", destination.name, exc)

        # Step 8: Log success and add to outputs
        logger.info(
            "Converted {} -> {} (scf={:.2f}, cloud_fraction={:.3f})",
            hdf.name,
            destination.relative_to(output_root),
            scf,
            cloud_fraction,
        )
        outputs.append(destination)

    return outputs


def cli_main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point for MOD10A1 preprocessing."""

    parser = argparse.ArgumentParser(
        prog="oa-da-mod10a1",
        description=(
            "Convert MODIS/Terra MOD10A1 (Collection 6/6.1) HDF files into "
            "GeoTIFFs containing the NDSI_Snow_Cover layer."
        ),
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing MOD10A1 HDF files")
    parser.add_argument("--season-label", required=True, help="Season folder name under the obs root")
    parser.add_argument("--project-dir", type=Path, help="Project directory (default: current working dir)")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override output root directory (default: <project-dir>/obs)",
    )
    parser.add_argument("--aoi", "--roi", dest="aoi", type=Path, help="ROI vector for clipping (GeoPackage, Shapefile, etc.)")
    parser.add_argument("--aoi-field", "--roi-field", dest="aoi_field", default="region_id", help="Field name in ROI with the region identifier (default: region_id)")
    parser.add_argument("--target-epsg", type=int, default=25832, help="Output EPSG code (default: 25832)")
    parser.add_argument("--resolution", type=float, help="Output pixel size in target units (e.g., 500)")
    parser.add_argument("--ndsi-threshold", type=float, default=40.0, help="NDSI threshold for snow classification (default: 40)")
    parser.add_argument("--no-envelope", action="store_true", help="Use exact AOI cutline instead of envelope bounds")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subdirectories for HDF files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing GeoTIFFs")
    parser.add_argument(
        "--max-cloud-fraction",
        type=float,
        help="Skip scenes with cloud coverage higher than this fraction (e.g., 0.1 for 10%%)",
    )
    parser.add_argument("--log-level", default="INFO", help="Loguru log level (default: INFO)")

    args = parser.parse_args(list(argv) if argv is not None else None)

    from openamundsen_da.core.constants import LOGURU_FORMAT
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    project_dir = Path(args.project_dir) if args.project_dir else Path.cwd()
    output_root = Path(args.output_root) if args.output_root else project_dir / OBS_DIR_NAME

    outputs = convert_mod10a1_directory(
        input_dir=Path(args.input_dir),
        output_root=output_root,
        season_label=args.season_label,
        aoi_path=Path(args.aoi) if args.aoi else None,
        target_epsg=int(args.target_epsg),
        resolution=float(args.resolution) if args.resolution is not None else None,
        use_envelope=not args.no_envelope,
        recursive=not args.no_recursive,
        overwrite=bool(args.overwrite),
        max_cloud_fraction=float(args.max_cloud_fraction) if args.max_cloud_fraction is not None else None,
        ndsi_threshold=float(args.ndsi_threshold),
        region_field=args.aoi_field,
    )

    logger.info("Finished - generated {} GeoTIFF(s)", len(outputs))
    return 0


def _ensure_gdal() -> None:
    """Raise a helpful error if GDAL bindings are missing."""

    if gdal is None:  # type: ignore[truth-value]
        raise RuntimeError(
            "osgeo.gdal is required but not available. Install GDAL with Python bindings in the "
            "active environment."
        ) from _GDAL_IMPORT_ERROR


## NOTE: Keep CLI guard at the very end of the file so that all
## helper functions are defined before cli_main() is executed.


def _read_ndsi_raster(raster_path: Path) -> tuple[np.ndarray, float | None, tuple | None, str | None]:
    """Read raster values, nodata, transform, and projection from GeoTIFF."""

    ds = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open GeoTIFF: {raster_path}")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    if arr is None:
        raise RuntimeError(f"Could not read data array from {raster_path}")
    nodata = band.GetNoDataValue()
    transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None
    return np.asarray(arr), nodata, transform, projection


def _cloud_fraction_from_data(data: np.ndarray, nodata: float | None) -> float | None:
    """Fraction of cloud pixels (value 200) among usable pixels (0-100 or 200)."""

    mask = np.zeros(data.shape, dtype=bool)
    if nodata is not None:
        mask |= data == nodata

    usable = (~mask) & ((data == 200) | ((data >= 0) & (data <= 100)))
    if not np.any(usable):
        return None

    cloud = (~mask) & (data == 200)
    cloud_count = int(np.count_nonzero(cloud))
    usable_count = int(np.count_nonzero(usable))
    return cloud_count / usable_count


def _classify_ndsi_data(
    data: np.ndarray,
    nodata: float | None,
    threshold: float,
) -> tuple[np.ndarray, float | None, int, int]:
    """Return (classification array, SCF, n_valid, n_snow) derived from NDSI data."""

    mask_invalid = np.zeros(data.shape, dtype=bool)
    if nodata is not None:
        mask_invalid |= data == nodata

    valid = (~mask_invalid) & (data >= 0) & (data <= 100)
    snow = valid & (data > threshold)
    no_snow = valid & (data <= threshold)

    class_arr = np.zeros(data.shape, dtype=np.uint8)
    class_arr[no_snow] = 1
    class_arr[snow] = 2

    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return class_arr, None, 0, 0

    snow_count = int(np.count_nonzero(snow))
    scf = snow_count / valid_count
    return class_arr, scf, valid_count, snow_count


def _write_classification_raster(
    output_path: Path,
    data: np.ndarray,
    transform: tuple | None,
    projection: str | None,
    overwrite: bool,
) -> None:
    """Write classification raster as Byte GeoTIFF with nodata=0."""

    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            return

    height, width = data.shape
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(
        str(output_path),
        width,
        height,
        1,
        gdal.GDT_Byte,
        ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2"],
    )
    if dst is None:
        raise RuntimeError(f"Could not create classification raster {output_path}")
    if transform:
        dst.SetGeoTransform(transform)
    if projection:
        dst.SetProjection(projection)

    band = dst.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(0)
    band.FlushCache()
    dst = None


def _update_scf_summary(
    summary_path: Path,
    date: datetime,
    region_id: str,
    scf: float,
    cloud_fraction: float,
    source_name: str,
    n_valid: int,
    n_snow: int,
) -> None:
    """Append or update SCF summary CSV in the season directory."""

    row = pd.DataFrame(
        {
            "date": [date.strftime("%Y-%m-%d")],
            "region_id": [region_id],
            "n_valid": [int(n_valid)],
            "n_snow": [int(n_snow)],
            "scf": [round(scf, 3)],
            "cloud_fraction": [round(cloud_fraction, 3)],
            "source": [source_name],
        }
    )

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[existing["date"] != row.loc[0, "date"]]
        row = pd.concat([existing, row], ignore_index=True)
        row = row.sort_values("date")

    row.to_csv(summary_path, index=False)

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(cli_main())

