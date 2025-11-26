from __future__ import annotations

"""AOI helpers for single-feature region handling.

Functions here centralize common AOI logic used by satellite SCF and H(x):
- Ensure exactly one feature
- Ensure required attribute is present (default: 'region_id')
- Optionally reproject to a target CRS
"""

from pathlib import Path
from typing import Tuple

import geopandas as gpd


def read_single_aoi(
    aoi_path: Path,
    *,
    required_field: str | None = "region_id",
    to_crs: "object | None" = None,
) -> Tuple[gpd.GeoDataFrame, str]:
    """Read a single-feature AOI and return (GeoDataFrame, region_id).

    Parameters
    ----------
    aoi_path : Path
        Vector file (e.g., GPKG/GeoJSON) containing exactly one polygon
        feature that represents the AOI.
    required_field : str or None, optional
        Name of the attribute field that contains the region identifier.
        Defaults to 'region_id'. When set to ``None``, no attribute is
        required and an empty string is returned as ``region_id``.
    to_crs : Any, optional
        If provided, the AOI is reprojected to this CRS.
    """
    # Prefer the default engine (pyogrio in newer GeoPandas) but fall back
    # to Fiona if the GDAL / pyogrio stack is misconfigured. This makes AOI
    # handling robust across environments and Docker images.
    try:
        gdf = gpd.read_file(aoi_path)
    except Exception as e:
        msg = str(e)
        if "GDAL data directory" in msg or "pyogrio" in msg:
            gdf = gpd.read_file(aoi_path, engine="fiona")
        else:
            raise
    if len(gdf) != 1:
        raise ValueError(f"AOI must contain exactly one feature (got {len(gdf)})")
    if required_field is not None and required_field not in gdf.columns:
        raise KeyError(f"AOI missing field '{required_field}'")
    if to_crs is not None:
        if gdf.crs is None:
            raise ValueError("AOI has no CRS; unable to align with target CRS")
        gdf = gdf.to_crs(to_crs)
    if required_field is None:
        region_id = ""
    else:
        region_id = str(gdf.iloc[0][required_field])
    return gdf, region_id
