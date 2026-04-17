from __future__ import annotations

import argparse
from functools import lru_cache
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

import geopandas as gpd
from loguru import logger

from openamundsen_da.io.paths import infer_setup_dir_from_project


GISCO_GEOJSON_BASE_URL = "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson"
GISCO_BOUNDARIES_GEOJSON_NAME = "CNTR_BN_01M_2020_3857.geojson"
GISCO_REGIONS_GEOJSON_NAME = "CNTR_RG_01M_2020_3857.geojson"
GISCO_LABELS_GEOJSON_NAME = "CNTR_LB_2020_3857.geojson"
GISCO_GEOJSON_NAME = GISCO_BOUNDARIES_GEOJSON_NAME
_ALL_OVERVIEW_GEOJSONS = (
    GISCO_BOUNDARIES_GEOJSON_NAME,
    GISCO_REGIONS_GEOJSON_NAME,
    GISCO_LABELS_GEOJSON_NAME,
)


def overview_cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "openamundsen_da" / "maps" / "overview"


def overview_setup_env_dir(setup_dir: Path) -> Path:
    return Path(setup_dir).resolve() / "env"


def overview_geojson_path(*, setup_dir: Path, filename: str) -> Path:
    return overview_setup_env_dir(setup_dir) / filename


def _gisco_geojson_url(filename: str) -> str:
    return f"{GISCO_GEOJSON_BASE_URL}/{filename}"


def ensure_overview_countries_geojson(
    *,
    setup_dir: Path | None = None,
    cache_dir: Path | None = None,
    filename: str = GISCO_BOUNDARIES_GEOJSON_NAME,
) -> Path:
    target_dir = (
        overview_setup_env_dir(Path(setup_dir))
        if setup_dir is not None
        else (Path(cache_dir) if cache_dir is not None else overview_cache_dir())
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = target_dir / filename

    if not geojson_path.is_file():
        with urlopen(_gisco_geojson_url(filename)) as response:
            payload = response.read()
        with tempfile.NamedTemporaryFile(dir=target_dir, prefix=f".{filename}.", suffix=".tmp", delete=False) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, geojson_path)

    return geojson_path


def ensure_overview_geojsons(
    *,
    setup_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    return {
        filename: ensure_overview_countries_geojson(
            setup_dir=setup_dir,
            cache_dir=cache_dir,
            filename=filename,
        )
        for filename in _ALL_OVERVIEW_GEOJSONS
    }


@lru_cache(maxsize=8)
def _load_overview_geojson_cached(filename: str, setup_dir: str | None, cache_dir: str | None) -> gpd.GeoDataFrame:
    geojson_path = ensure_overview_countries_geojson(
        setup_dir=Path(setup_dir) if setup_dir is not None else None,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        filename=filename,
    )
    data = gpd.read_file(geojson_path)
    data = data[data.geometry.notna()].copy()
    data = data.loc[~data.geometry.is_empty].copy()
    return data


def _cached_overview_copy(filename: str, *, setup_dir: Path | None = None, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    # Return a copy so callers can filter/clip without mutating the shared in-process cache.
    return _load_overview_geojson_cached(
        filename,
        str(Path(setup_dir).resolve()) if setup_dir is not None else None,
        str(cache_dir) if cache_dir is not None else None,
    ).copy()


def load_overview_boundaries(*, setup_dir: Path | None = None, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    return _cached_overview_copy(GISCO_BOUNDARIES_GEOJSON_NAME, setup_dir=setup_dir, cache_dir=cache_dir)


def load_overview_labels(*, setup_dir: Path | None = None, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    return _cached_overview_copy(GISCO_LABELS_GEOJSON_NAME, setup_dir=setup_dir, cache_dir=cache_dir)


def load_overview_regions(*, setup_dir: Path | None = None, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    return _cached_overview_copy(GISCO_REGIONS_GEOJSON_NAME, setup_dir=setup_dir, cache_dir=cache_dir)


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oa-da-fetch-overview-geojson",
        description="Download overview GISCO GeoJSON assets into <setup>/env for project-map overview panels.",
    )
    parser.add_argument("--setup-dir", type=Path, help="Setup directory that owns env/")
    parser.add_argument("--project-dir", type=Path, help="Project directory used to infer the setup directory")
    args = parser.parse_args(argv)

    if bool(args.setup_dir) == bool(args.project_dir):
        parser.error("provide exactly one of --setup-dir or --project-dir")

    setup_dir = Path(args.setup_dir).resolve() if args.setup_dir else infer_setup_dir_from_project(Path(args.project_dir))
    outputs = ensure_overview_geojsons(setup_dir=setup_dir)
    for filename, path in outputs.items():
        logger.info("Prepared overview GeoJSON {} -> {}", filename, path)
    return 0


__all__ = [
    "GISCO_BOUNDARIES_GEOJSON_NAME",
    "GISCO_GEOJSON_BASE_URL",
    "GISCO_REGIONS_GEOJSON_NAME",
    "cli_main",
    "ensure_overview_countries_geojson",
    "ensure_overview_geojsons",
    "load_overview_boundaries",
    "load_overview_labels",
    "load_overview_regions",
    "overview_cache_dir",
    "overview_geojson_path",
    "overview_setup_env_dir",
]
