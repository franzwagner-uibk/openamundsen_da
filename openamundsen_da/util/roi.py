"""ROI (region of interest) helpers for polygon region handling."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd
from shapely.ops import unary_union


def read_single_roi(
    roi_path: Path,
    *,
    required_field: str | None = None,
    to_crs: "object | None" = None,
) -> Tuple[gpd.GeoDataFrame, str]:
    """Read one-or-more ROI polygons and return a single merged geometry.

    Parameters
    ----------
    roi_path : Path
        Vector file (e.g., GPKG/GeoJSON) containing one or more polygon
        features that represent the ROI.
    required_field : str or None, optional
        Name of the attribute field that contains the region identifier.
        Defaults to 'region_id'. When set to ``None``, no attribute is
        required and an empty string is returned as ``region_id``.
    to_crs : Any, optional
        If provided, the ROI is reprojected to this CRS.
    """
    # Prefer the default engine (pyogrio) but fall back to Fiona for robustness.
    try:
        gdf = gpd.read_file(roi_path)
    except Exception as e:
        msg = str(e)
        if "GDAL data directory" in msg or "pyogrio" in msg:
            gdf = gpd.read_file(roi_path, engine="fiona")
        else:
            raise
    if gdf.empty:
        raise ValueError(f"ROI must contain at least one feature (got {len(gdf)})")
    effective_field = required_field
    if required_field is not None and required_field not in gdf.columns:
        if required_field == "region_id" and "id" in gdf.columns:
            effective_field = "id"
        else:
            raise KeyError(f"ROI missing field '{required_field}'")

    region_id = ""
    if effective_field is not None:
        # Keep the first feature label as region_id when multiple polygons are present.
        region_id = str(gdf.iloc[0][effective_field])

    if len(gdf) > 1:
        merged_geom = unary_union([geom for geom in gdf.geometry if geom is not None and not geom.is_empty])
        if merged_geom is None or merged_geom.is_empty:
            raise ValueError("ROI geometries are empty after union.")
        row = {effective_field: region_id} if effective_field is not None else {}
        gdf = gpd.GeoDataFrame([row], geometry=[merged_geom], crs=gdf.crs)

    if to_crs is not None:
        if gdf.crs is None:
            raise ValueError("ROI has no CRS; unable to align with target CRS")
        gdf = gdf.to_crs(to_crs)
    return gdf, region_id


__all__ = ["read_single_roi"]
