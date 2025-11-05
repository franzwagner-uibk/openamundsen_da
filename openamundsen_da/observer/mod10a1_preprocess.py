"""openamundsen_da.observer.mod10a1_preprocess

Helper utilities to convert MODIS/Terra MOD10A1 (Collection 6/6.1) HDF files
into GeoTIFFs containing the `NDSI_Snow_Cover` layer, optionally clipped to an
AOI and reprojected to a target CRS.

Outputs target GeoTIFFs named `NDSI_Snow_Cover_YYYYMMDD.tif` under
`<output_root>/<season_label>/<YYYY>/<YYYY-MM-DD>/`, matching the expectations
of :mod:`openamundsen_da.observer.satellite_scf`.
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
    """Parse product, acquisition date, and tile from a MOD10A1 filename."""

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
    """Return sorted list of MOD10A1 HDF files under *root*."""

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


def _target_bounds(aoi_path: Path, target_epsg: int) -> tuple[float, float, float, float]:
    """Compute AOI bounds in *target_epsg* for fast cropping."""

    gdf = gpd.read_file(aoi_path)
    if gdf.empty:
        raise ValueError(f"AOI file is empty: {aoi_path}")
    if target_epsg and gdf.crs is not None and gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(target_epsg)
    minx, miny, maxx, maxy = gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _build_output_path(output_root: Path, season_label: str, when: datetime) -> Path:
    """Derive output path for the converted GeoTIFF."""

    season_dir = output_root / season_label
    datedir = season_dir / when.strftime("%Y") / when.strftime("%Y-%m-%d")
    datedir.mkdir(parents=True, exist_ok=True)
    return datedir / f"{MOD10A1_SDS_NAME}_{when.strftime('%Y%m%d')}.tif"


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
    """Reproject and optionally crop the NDSI subdataset into *destination*."""

    if destination.exists() and not overwrite:
        logger.info("Skipping existing %s (use --overwrite to replace)", destination)
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
) -> list[Path]:
    """Convert all MOD10A1 HDF files under *input_dir* to GeoTIFFs."""

    _ensure_gdal()
    ensure_gdal_proj_from_conda()

    logger.debug("Scanning %s for MOD10A1 HDF files (recursive=%s)", input_dir, recursive)
    hdf_files = _list_hdf_files(Path(input_dir), recursive)
    if not hdf_files:
        logger.warning("No MOD10A1 HDF files found in %s", input_dir)
        return []

    bounds = None
    aoi = Path(aoi_path) if aoi_path is not None else None
    if aoi and use_envelope:
        bounds = _target_bounds(aoi, target_epsg)
        logger.debug("AOI bounds in EPSG:%s -> %s", target_epsg, bounds)

    outputs: list[Path] = []
    for hdf in hdf_files:
        try:
            meta = _parse_mod10a1_name(hdf)
        except ValueError as exc:
            logger.error("Skipping %s: %s", hdf.name, exc)
            continue

        try:
            subdataset = _find_ndsi_subdataset(hdf)
        except RuntimeError as exc:
            logger.error("Skipping %s: %s", hdf.name, exc)
            continue

        destination = _build_output_path(output_root, season_label, meta.date)
        try:
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
            logger.error("Conversion failed for %s: %s", hdf.name, exc)
            continue

        logger.info(
            "Converted %s -> %s", hdf.name, destination.relative_to(output_root)
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
    parser.add_argument("--aoi", type=Path, help="AOI vector for clipping (GeoPackage, Shapefile, etc.)")
    parser.add_argument("--target-epsg", type=int, default=25832, help="Output EPSG code (default: 25832)")
    parser.add_argument("--resolution", type=float, help="Output pixel size in target units (e.g., 500)")
    parser.add_argument("--no-envelope", action="store_true", help="Use exact AOI cutline instead of envelope bounds")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subdirectories for HDF files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing GeoTIFFs")
    parser.add_argument("--log-level", default="INFO", help="Loguru log level (default: INFO)")

    args = parser.parse_args(list(argv) if argv is not None else None)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())

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
    )

    logger.info("Finished - generated %d GeoTIFF(s)", len(outputs))
    return 0


def _ensure_gdal() -> None:
    """Raise a helpful error if GDAL bindings are missing."""

    if gdal is None:  # type: ignore[truth-value]
        raise RuntimeError(
            "osgeo.gdal is required but not available. Install GDAL with Python bindings in the "
            "active environment."
        ) from _GDAL_IMPORT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
