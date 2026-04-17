from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd


GISCO_GEOJSON_BASE_URL = "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson"
GISCO_BOUNDARIES_GEOJSON_NAME = "CNTR_BN_01M_2020_3857.geojson"
GISCO_REGIONS_GEOJSON_NAME = "CNTR_RG_01M_2020_3857.geojson"
GISCO_LABELS_GEOJSON_NAME = "CNTR_LB_2020_3857.geojson"
GISCO_GEOJSON_NAME = GISCO_BOUNDARIES_GEOJSON_NAME


def overview_cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "openamundsen_da" / "project_maps" / "overview"


def _gisco_geojson_url(filename: str) -> str:
    return f"{GISCO_GEOJSON_BASE_URL}/{filename}"


def ensure_overview_countries_geojson(*, cache_dir: Path | None = None, filename: str = GISCO_BOUNDARIES_GEOJSON_NAME) -> Path:
    target_dir = Path(cache_dir) if cache_dir is not None else overview_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = target_dir / filename

    if not geojson_path.is_file():
        with urlopen(_gisco_geojson_url(filename)) as response:
            geojson_path.write_bytes(response.read())

    return geojson_path


@lru_cache(maxsize=8)
def _load_overview_geojson_cached(filename: str, cache_dir: str | None) -> gpd.GeoDataFrame:
    geojson_path = ensure_overview_countries_geojson(
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        filename=filename,
    )
    data = gpd.read_file(geojson_path)
    data = data[data.geometry.notna()].copy()
    data = data.loc[~data.geometry.is_empty].copy()
    return data


def _cached_overview_copy(filename: str, *, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    # Return a copy so callers can filter/clip without mutating the shared in-process cache.
    return _load_overview_geojson_cached(filename, str(cache_dir) if cache_dir is not None else None).copy()


def load_overview_boundaries(*, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    return _cached_overview_copy(GISCO_BOUNDARIES_GEOJSON_NAME, cache_dir=cache_dir)


def load_overview_labels(*, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    return _cached_overview_copy(GISCO_LABELS_GEOJSON_NAME, cache_dir=cache_dir)


def load_overview_regions(*, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    return _cached_overview_copy(GISCO_REGIONS_GEOJSON_NAME, cache_dir=cache_dir)


__all__ = [
    "GISCO_BOUNDARIES_GEOJSON_NAME",
    "GISCO_GEOJSON_BASE_URL",
    "GISCO_REGIONS_GEOJSON_NAME",
    "ensure_overview_countries_geojson",
    "load_overview_boundaries",
    "load_overview_labels",
    "load_overview_regions",
    "overview_cache_dir",
]
