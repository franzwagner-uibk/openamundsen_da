"""Station metadata loaders shared across visualization modules."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import abspath_relative_to, find_setup_yaml, list_member_dirs
from openamundsen_da.util.station_da import STATION_SNOW_DEPTH_METADATA_FILENAME


def _transform_coords(x, y, src_crs: str | None, dst_crs: str | None):
    if src_crs is None or dst_crs is None or str(src_crs).lower() == str(dst_crs).lower():
        return np.asarray(x), np.asarray(y)
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(np.asarray(x), np.asarray(y))


def _find_setup_yaml_lenient(setup_dir: Path) -> Path | None:
    try:
        return find_setup_yaml(setup_dir)
    except FileNotFoundError as exc:
        if "missing projects/" not in str(exc):
            raise
        candidates = [Path(setup_dir) / f"{Path(setup_dir).name}.yml", Path(setup_dir) / "setup.yml"]
        candidates.extend(sorted(Path(setup_dir).glob("*.yml")))
        return next((path for path in candidates if path.is_file()), None)


def load_setup_station_table(setup_dir: Path) -> pd.DataFrame | None:
    """Load setup-level station metadata from the configured meteo source."""
    setup_yaml = _find_setup_yaml_lenient(Path(setup_dir))
    if setup_yaml is None:
        return None
    setup_cfg = _read_yaml_file(setup_yaml) or {}
    if not isinstance(setup_cfg, dict):
        return None
    input_data = (setup_cfg.get("input_data") or {}) if isinstance(setup_cfg.get("input_data"), dict) else {}
    meteo_cfg = (input_data.get("meteo") or {}) if isinstance(input_data.get("meteo"), dict) else {}
    meteo_dir_raw = meteo_cfg.get("dir") or "meteo"
    meteo_dir = Path(abspath_relative_to(setup_dir, Path(str(meteo_dir_raw))))
    meteo_format = str(meteo_cfg.get("format") or "csv").strip().lower()
    grid_crs = setup_cfg.get("crs")

    if meteo_format == "csv":
        stations_path = meteo_dir / "stations.csv"
        if not stations_path.is_file():
            return None
        stations = pd.read_csv(stations_path)
        if {"id", "x", "y"}.issubset(stations.columns):
            meteo_crs = meteo_cfg.get("crs")
            if meteo_crs and grid_crs and str(meteo_crs).lower() != str(grid_crs).lower():
                xs, ys = _transform_coords(stations["x"], stations["y"], meteo_crs, grid_crs)
                stations = stations.copy()
                stations["x"] = xs
                stations["y"] = ys
            return stations
        return None

    if meteo_format == "netcdf":
        rows: list[dict[str, object]] = []
        for nc_path in sorted(meteo_dir.glob("*.nc")):
            with xr.open_dataset(nc_path) as ds:
                station_id = str(nc_path.stem)
                lon = float(ds["lon"].values)
                lat = float(ds["lat"].values)
                alt = float(ds["alt"].values)
                station_name = ds.attrs.get("station_name")
                if not station_name and "station_name" in ds:
                    station_name = str(np.asarray(ds["station_name"]).reshape(-1)[0])
                x, y = _transform_coords(lon, lat, "epsg:4326", grid_crs)
                rows.append(
                    {
                        "id": station_id,
                        "name": str(station_name or station_id),
                        "x": float(x),
                        "y": float(y),
                        "alt": alt,
                    }
                )
        return pd.DataFrame(rows) if rows else None

    return None


def load_setup_snow_depth_station_table(
    setup_dir: Path,
    *,
    obs_stations_dirs: Sequence[Path] = (),
    grid_crs: str | None = None,
) -> pd.DataFrame | None:
    """Load setup-level snow-observation station metadata."""
    setup_dir = Path(setup_dir)
    if grid_crs is None:
        setup_yaml = _find_setup_yaml_lenient(setup_dir)
        if setup_yaml is not None:
            setup_cfg = _read_yaml_file(setup_yaml) or {}
            if isinstance(setup_cfg, dict):
                grid_crs = setup_cfg.get("crs")

    candidate_dirs: list[Path] = []
    for obs_dir in obs_stations_dirs:
        path = Path(obs_dir)
        if path not in candidate_dirs:
            candidate_dirs.append(path)
    default_dir = setup_dir / "obs" / "stations"
    if default_dir not in candidate_dirs:
        candidate_dirs.append(default_dir)

    for obs_dir in candidate_dirs:
        metadata_path = obs_dir / STATION_SNOW_DEPTH_METADATA_FILENAME
        if not metadata_path.is_file():
            continue
        stations = pd.read_csv(metadata_path)
        if "id" not in stations.columns:
            raise ValueError(f"Snow observation station metadata missing required column 'id': {metadata_path}")
        out = stations.copy()
        out["id"] = out["id"].astype(str)
        if "name" not in out.columns:
            out["name"] = out["id"]
        if {"x", "y"}.issubset(out.columns):
            out["x"] = pd.to_numeric(out["x"], errors="coerce")
            out["y"] = pd.to_numeric(out["y"], errors="coerce")
        elif {"lon", "lat"}.issubset(out.columns):
            if grid_crs is None:
                raise ValueError(
                    f"Snow observation station metadata has lon/lat but setup CRS is unavailable: {metadata_path}"
                )
            xs, ys = _transform_coords(out["lon"], out["lat"], "epsg:4326", grid_crs)
            out["x"] = xs
            out["y"] = ys
        else:
            raise ValueError(f"Snow observation station metadata must contain x/y or lon/lat: {metadata_path}")
        if "alt" in out.columns:
            out["alt"] = pd.to_numeric(out["alt"], errors="coerce")
        finite_xy = np.isfinite(out["x"].astype(float)) & np.isfinite(out["y"].astype(float))
        out = out.loc[finite_xy].copy()
        return out if not out.empty else None
    return None


def load_ensemble_station_table(step_dir: Path, ensemble: str) -> Optional[pd.DataFrame]:
    """Load per-step station metadata from open_loop or first member meteo dir."""
    base = Path(step_dir) / "ensembles" / str(ensemble)
    candidates = [base / "open_loop" / "meteo" / "stations.csv"]
    members = list_member_dirs(Path(step_dir) / "ensembles", ensemble)
    if members:
        candidates.append(members[0] / "meteo" / "stations.csv")
    for path in candidates:
        if path.is_file():
            try:
                return pd.read_csv(path)
            except Exception:
                return None
    return None


def load_ensemble_station_table_from_steps(step_dirs: Sequence[Path], ensemble: str = "prior") -> Optional[pd.DataFrame]:
    """Load the first available per-step station table across a list of steps."""
    for step_dir in step_dirs:
        stations = load_ensemble_station_table(Path(step_dir), ensemble)
        if stations is not None:
            return stations
    return None


__all__ = [
    "load_ensemble_station_table",
    "load_ensemble_station_table_from_steps",
    "load_setup_snow_depth_station_table",
    "load_setup_station_table",
]
