from __future__ import annotations

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


def load_overview_boundaries(*, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    geojson_path = ensure_overview_countries_geojson(
        cache_dir=cache_dir,
        filename=GISCO_BOUNDARIES_GEOJSON_NAME,
    )
    countries = gpd.read_file(geojson_path)
    countries = countries[countries.geometry.notna()].copy()
    countries = countries.loc[~countries.geometry.is_empty].copy()
    return countries


def load_overview_labels(*, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    geojson_path = ensure_overview_countries_geojson(
        cache_dir=cache_dir,
        filename=GISCO_LABELS_GEOJSON_NAME,
    )
    labels = gpd.read_file(geojson_path)
    labels = labels[labels.geometry.notna()].copy()
    labels = labels.loc[~labels.geometry.is_empty].copy()
    return labels


def load_overview_regions(*, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    geojson_path = ensure_overview_countries_geojson(
        cache_dir=cache_dir,
        filename=GISCO_REGIONS_GEOJSON_NAME,
    )
    regions = gpd.read_file(geojson_path)
    regions = regions[regions.geometry.notna()].copy()
    regions = regions.loc[~regions.geometry.is_empty].copy()
    return regions


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
