"""ROI (region of interest) helpers for single-feature region handling."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd


def read_single_roi(
    roi_path: Path,
    *,
    required_field: str | None = None,
    to_crs: "object | None" = None,
) -> Tuple[gpd.GeoDataFrame, str]:
    """Read a single-feature ROI and return (GeoDataFrame, region_id).

    Parameters
    ----------
    roi_path : Path
        Vector file (e.g., GPKG/GeoJSON) containing exactly one polygon
        feature that represents the ROI.
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
    if len(gdf) != 1:
        raise ValueError(f"ROI must contain exactly one feature (got {len(gdf)})")
    if required_field is not None and required_field not in gdf.columns:
        raise KeyError(f"ROI missing field '{required_field}'")
    if to_crs is not None:
        if gdf.crs is None:
            raise ValueError("ROI has no CRS; unable to align with target CRS")
        gdf = gdf.to_crs(to_crs)
    if required_field is None:
        region_id = ""
    else:
        region_id = str(gdf.iloc[0][required_field])
    return gdf, region_id


__all__ = ["read_single_roi"]
