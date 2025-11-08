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
    required_field: str = "region_id",
    to_crs: "object | None" = None,
) -> Tuple[gpd.GeoDataFrame, str]:
    """Read a single-feature AOI and return (GeoDataFrame, region_id).

    Parameters
    ----------
    aoi_path : Path
        Vector file (e.g., GPKG/GeoJSON) containing exactly one polygon
        feature that represents the AOI.
    required_field : str, optional
        Name of the attribute field that contains the region identifier.
        Defaults to 'region_id'.
    to_crs : Any, optional
        If provided, the AOI is reprojected to this CRS.
    """
    gdf = gpd.read_file(aoi_path)
    if len(gdf) != 1:
        raise ValueError(f"AOI must contain exactly one feature (got {len(gdf)})")
    if required_field not in gdf.columns:
        raise KeyError(f"AOI missing field '{required_field}'")
    if to_crs is not None:
        if gdf.crs is None:
            raise ValueError("AOI has no CRS; unable to align with target CRS")
        gdf = gdf.to_crs(to_crs)
    region_id = str(gdf.iloc[0][required_field])
    return gdf, region_id

