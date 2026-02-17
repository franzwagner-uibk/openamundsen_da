from __future__ import annotations

"""AOI helpers for polygon region handling.

Functions here centralize common AOI logic used by satellite SCF and H(x):
- Merge one or more features into one AOI geometry
- Ensure required attribute is present (default: 'region_id')
- Optionally reproject to a target CRS
"""

from pathlib import Path
from typing import Tuple

import geopandas as gpd
from shapely.ops import unary_union


def read_single_aoi(
    aoi_path: Path,
    *,
    required_field: str | None = "region_id",
    to_crs: "object | None" = None,
) -> Tuple[gpd.GeoDataFrame, str]:
    """Read one-or-more AOI polygons and return a single merged geometry.

    Parameters
    ----------
    aoi_path : Path
        Vector file (e.g., GPKG/GeoJSON) containing one or more polygon
        features that represent the AOI.
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
    if gdf.empty:
        raise ValueError(f"AOI must contain at least one feature (got {len(gdf)})")
    effective_field = required_field
    if required_field is not None and required_field not in gdf.columns:
        if required_field == "region_id" and "id" in gdf.columns:
            effective_field = "id"
        else:
            raise KeyError(f"AOI missing field '{required_field}'")

    region_id = ""
    if effective_field is not None:
        region_id = str(gdf.iloc[0][effective_field])

    if len(gdf) > 1:
        merged_geom = unary_union([geom for geom in gdf.geometry if geom is not None and not geom.is_empty])
        if merged_geom is None or merged_geom.is_empty:
            raise ValueError("AOI geometries are empty after union.")
        row = {effective_field: region_id} if effective_field is not None else {}
        gdf = gpd.GeoDataFrame([row], geometry=[merged_geom], crs=gdf.crs)

    if to_crs is not None:
        if gdf.crs is None:
            raise ValueError("AOI has no CRS; unable to align with target CRS")
        gdf = gdf.to_crs(to_crs)
    return gdf, region_id
