from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlopen
import zipfile

import geopandas as gpd


GISCO_COUNTRIES_URL = "https://gisco-services.ec.europa.eu/distribution/v2/countries/download/ref-countries-2020-01m.geojson.zip"
GISCO_ZIP_NAME = "ref-countries-2020-01m.geojson.zip"
GISCO_GEOJSON_NAME = "CNTR_BN_01M_2020_3857.geojson"


def overview_cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "openamundsen_da" / "project_maps" / "overview"


def ensure_overview_countries_geojson(*, cache_dir: Path | None = None) -> Path:
    target_dir = Path(cache_dir) if cache_dir is not None else overview_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / GISCO_ZIP_NAME
    geojson_path = target_dir / GISCO_GEOJSON_NAME

    if not geojson_path.is_file():
        if not zip_path.is_file():
            with urlopen(GISCO_COUNTRIES_URL) as response:
                zip_path.write_bytes(response.read())
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(GISCO_GEOJSON_NAME, path=target_dir)

    return geojson_path


def load_overview_boundaries(*, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    geojson_path = ensure_overview_countries_geojson(cache_dir=cache_dir)
    countries = gpd.read_file(geojson_path)
    countries = countries[countries.geometry.notna()].copy()
    countries = countries.loc[~countries.geometry.is_empty].copy()
    return countries


__all__ = [
    "GISCO_COUNTRIES_URL",
    "ensure_overview_countries_geojson",
    "load_overview_boundaries",
    "overview_cache_dir",
]
