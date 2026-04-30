from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.util.config_validators import require_mapping
from openamundsen_da.util.roi_grid import resolve_setup_grid_spec


@dataclass(frozen=True)
class WetSnowLineConfig:
    elevation_band_size_m: float
    smoothing_window_bands: int
    crossing_fraction: float
    wet_elevation_percentile: float
    aspect_diagnostics: str
    sector_relative_threshold: float


@dataclass(frozen=True)
class WetSnowLineEvaluation:
    wet_snow_line: float | None
    wet_elevation_percentile: float | None
    n_valid: int
    n_wet: int
    wet_bands: int
    method: str
    gate_reason: str | None
    profile: pd.DataFrame
    sector_relative_lines: dict[str, float | None]
    sector_relative_profiles: dict[str, pd.DataFrame]


_ALLOWED_ASPECT_DIAGNOSTICS = {"off", "north_south", "four_sectors"}
_SECTOR_LABELS = {
    "north_south": ("N", "S"),
    "four_sectors": ("N", "E", "S", "W"),
}


def _coerce_float(raw: object, *, path: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value at {path}: {raw!r}") from exc


def _coerce_int(raw: object, *, path: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value at {path}: {raw!r}") from exc
    return value


def load_wet_snow_line_config(project_dir: Path) -> WetSnowLineConfig:
    project_yaml = find_project_yaml(project_dir)
    cfg = require_mapping(_read_yaml_file(project_yaml) or {}, path="project")
    da_cfg = require_mapping(cfg.get("data_assimilation"), path="project.data_assimilation")
    raw = require_mapping(
        da_cfg.get("wet_snow_line"),
        path="project.data_assimilation.wet_snow_line",
    )
    cfg_path = "project.data_assimilation.wet_snow_line"
    aspect_diagnostics = str(raw.get("aspect_diagnostics", "four_sectors")).strip().lower()
    if aspect_diagnostics not in _ALLOWED_ASPECT_DIAGNOSTICS:
        raise ValueError(
            f"{cfg_path}.aspect_diagnostics must be one of "
            f"{sorted(_ALLOWED_ASPECT_DIAGNOSTICS)}, got {aspect_diagnostics!r}"
        )
    out = WetSnowLineConfig(
        elevation_band_size_m=_coerce_float(
            raw.get("elevation_band_size_m", 100.0),
            path=f"{cfg_path}.elevation_band_size_m",
        ),
        smoothing_window_bands=_coerce_int(
            raw.get("smoothing_window_bands", 3),
            path=f"{cfg_path}.smoothing_window_bands",
        ),
        crossing_fraction=_coerce_float(
            raw.get("crossing_fraction", 0.5),
            path=f"{cfg_path}.crossing_fraction",
        ),
        wet_elevation_percentile=_coerce_float(
            raw.get("wet_elevation_percentile", 95.0),
            path=f"{cfg_path}.wet_elevation_percentile",
        ),
        aspect_diagnostics=aspect_diagnostics,
        sector_relative_threshold=_coerce_float(
            raw.get("sector_relative_threshold", 0.8),
            path=f"{cfg_path}.sector_relative_threshold",
        ),
    )
    if out.elevation_band_size_m <= 0.0:
        raise ValueError(f"{cfg_path}.elevation_band_size_m must be > 0")
    if out.smoothing_window_bands < 1:
        raise ValueError(f"{cfg_path}.smoothing_window_bands must be >= 1")
    if not (0.0 <= out.crossing_fraction <= 1.0):
        raise ValueError(f"{cfg_path}.crossing_fraction must be within [0, 1]")
    if not (0.0 <= out.wet_elevation_percentile <= 100.0):
        raise ValueError(f"{cfg_path}.wet_elevation_percentile must be within [0, 100]")
    if not (0.0 <= out.sector_relative_threshold <= 1.0):
        raise ValueError(f"{cfg_path}.sector_relative_threshold must be within [0, 1]")
    return out


@lru_cache(maxsize=16)
def _load_dem_and_aspect(setup_dir_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    setup_dir = Path(setup_dir_str)
    spec = resolve_setup_grid_spec(setup_dir)
    with rasterio.open(spec.dem_path) as src:
        dem = src.read(1).astype(float)
        nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan

    dem_filled = dem.copy()
    if np.isnan(dem_filled).any():
        fill_value = float(np.nanmedian(dem_filled))
        dem_filled = np.where(np.isnan(dem_filled), fill_value, dem_filled)

    res_x = abs(float(spec.transform.a))
    res_y = abs(float(spec.transform.e))
    grad_y, grad_x = np.gradient(dem_filled, res_y, res_x)
    slope = np.hypot(grad_x, grad_y)
    aspect = 90.0 - np.degrees(np.arctan2(grad_y, -grad_x))
    aspect = np.mod(aspect + 360.0, 360.0)
    aspect[slope <= 1e-9] = np.nan
    return dem, aspect, slope


def _build_profile(
    *,
    dem: np.ndarray,
    valid_mask: np.ndarray,
    wet_mask: np.ndarray,
    cfg: WetSnowLineConfig,
) -> pd.DataFrame:
    valid_elev = dem[valid_mask]
    if valid_elev.size == 0:
        return pd.DataFrame(
            columns=[
                "band_low_m",
                "band_high_m",
                "band_mid_m",
                "n_valid",
                "n_wet",
                "f_wet",
                "f_wet_smooth",
                "above_crossing_fraction",
                "crossing_fraction",
            ]
        )

    band = float(cfg.elevation_band_size_m)
    low = np.floor(np.nanmin(valid_elev) / band) * band
    high = np.ceil(np.nanmax(valid_elev) / band) * band
    if np.isclose(low, high):
        high = low + band
    edges = np.arange(low, high + band, band, dtype=float)
    if edges.size < 2:
        edges = np.array([low, low + band], dtype=float)

    rows: list[dict[str, float | int]] = []
    for idx in range(len(edges) - 1):
        band_low = float(edges[idx])
        band_high = float(edges[idx + 1])
        band_mask = valid_mask & (dem >= band_low) & (dem < band_high)
        n_valid = int(np.count_nonzero(band_mask))
        n_wet = int(np.count_nonzero(wet_mask & band_mask))
        f_wet = float(n_wet / n_valid) if n_valid > 0 else np.nan
        rows.append(
            {
                "band_low_m": band_low,
                "band_high_m": band_high,
                "band_mid_m": band_low + (band / 2.0),
                "n_valid": n_valid,
                "n_wet": n_wet,
                "f_wet": f_wet,
            }
        )
    profile = pd.DataFrame(rows)
    if profile.empty:
        profile["f_wet_smooth"] = pd.Series(dtype=float)
        profile["above_crossing_fraction"] = pd.Series(dtype=bool)
        profile["crossing_fraction"] = pd.Series(dtype=float)
        return profile
    profile["f_wet_smooth"] = (
        profile["f_wet"]
        .rolling(window=int(cfg.smoothing_window_bands), center=True, min_periods=1)
        .median()
    )
    profile["above_crossing_fraction"] = profile["f_wet_smooth"] >= float(cfg.crossing_fraction)
    profile["crossing_fraction"] = float(cfg.crossing_fraction)
    return profile


def _interpolate_crossing(
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    threshold: float,
) -> float:
    if np.isclose(y1, y2):
        return float(x1)
    frac = (threshold - y1) / (y2 - y1)
    return float(x1 + frac * (x2 - x1))


def _find_first_downward_crossing(profile: pd.DataFrame, threshold: float) -> float | None:
    series = profile[["band_mid_m", "f_wet_smooth"]].dropna().sort_values("band_mid_m").reset_index(drop=True)
    if len(series) < 2:
        return None
    for idx in range(len(series) - 1):
        y1 = float(series.loc[idx, "f_wet_smooth"])
        y2 = float(series.loc[idx + 1, "f_wet_smooth"])
        if y1 >= threshold and y2 < threshold:
            return _interpolate_crossing(
                x1=float(series.loc[idx, "band_mid_m"]),
                y1=y1,
                x2=float(series.loc[idx + 1, "band_mid_m"]),
                y2=y2,
                threshold=threshold,
            )
    return None


def _find_sector_relative_crossing(profile: pd.DataFrame, relative_threshold: float) -> float | None:
    series = profile[["band_mid_m", "f_wet_smooth"]].dropna().sort_values("band_mid_m").reset_index(drop=True)
    if len(series) < 2:
        return None
    wsf_max = float(series["f_wet_smooth"].max())
    if not np.isfinite(wsf_max) or wsf_max <= 0.0:
        return None
    threshold = float(relative_threshold) * wsf_max
    max_mask = np.isclose(series["f_wet_smooth"].to_numpy(dtype=float), wsf_max)
    start_idx = int(np.where(max_mask)[0][-1])
    for idx in range(start_idx, len(series) - 1):
        y1 = float(series.loc[idx, "f_wet_smooth"])
        y2 = float(series.loc[idx + 1, "f_wet_smooth"])
        if y1 >= threshold and y2 < threshold:
            return _interpolate_crossing(
                x1=float(series.loc[idx, "band_mid_m"]),
                y1=y1,
                x2=float(series.loc[idx + 1, "band_mid_m"]),
                y2=y2,
                threshold=threshold,
            )
    return None


def _build_fraction_profile(
    *,
    dem: np.ndarray,
    valid_mask: np.ndarray,
    wet_fraction: np.ndarray,
    cfg: WetSnowLineConfig,
) -> pd.DataFrame:
    valid_elev = dem[valid_mask]
    if valid_elev.size == 0:
        return pd.DataFrame(columns=["band_mid_m", "f_wet", "f_wet_smooth"])

    band = float(cfg.elevation_band_size_m)
    low = np.floor(np.nanmin(valid_elev) / band) * band
    high = np.ceil(np.nanmax(valid_elev) / band) * band
    if np.isclose(low, high):
        high = low + band
    edges = np.arange(low, high + band, band, dtype=float)
    if edges.size < 2:
        edges = np.array([low, low + band], dtype=float)

    rows: list[dict[str, float]] = []
    for idx in range(len(edges) - 1):
        band_low = float(edges[idx])
        band_high = float(edges[idx + 1])
        band_mask = valid_mask & (dem >= band_low) & (dem < band_high)
        rows.append(
            {
                "band_mid_m": band_low + (band / 2.0),
                "f_wet": float(np.nanmean(wet_fraction[band_mask])) if np.any(band_mask) else np.nan,
            }
        )

    profile = pd.DataFrame(rows)
    profile["f_wet_smooth"] = (
        profile["f_wet"]
        .rolling(window=int(cfg.smoothing_window_bands), center=True, min_periods=1)
        .median()
    )
    return profile


def compute_wet_snow_line_from_fraction_grid(
    *,
    project_dir: Path,
    dem: np.ndarray,
    roi_mask: np.ndarray,
    wet_fraction: np.ndarray,
    threshold: float | None = None,
) -> float | None:
    """Compute WSLA from a gridded wet-snow fraction field on the model grid."""

    cfg = load_wet_snow_line_config(project_dir)
    dem = np.asarray(dem, dtype=float)
    wet_fraction = np.asarray(wet_fraction, dtype=float)
    valid = np.isfinite(wet_fraction) & np.asarray(roi_mask, dtype=bool) & np.isfinite(dem)
    if not np.any(valid):
        return None

    profile = _build_fraction_profile(dem=dem, valid_mask=valid, wet_fraction=wet_fraction, cfg=cfg)
    crossing_threshold = float(cfg.crossing_fraction if threshold is None else threshold)
    return _find_first_downward_crossing(profile, crossing_threshold)


def _evaluate_from_masks(
    *,
    dem: np.ndarray,
    valid_mask: np.ndarray,
    wet_mask: np.ndarray,
    cfg: WetSnowLineConfig,
) -> WetSnowLineEvaluation:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(dem)
    wet = np.asarray(wet_mask, dtype=bool) & valid
    n_valid = int(np.count_nonzero(valid))
    n_wet = int(np.count_nonzero(wet))
    profile = _build_profile(dem=dem, valid_mask=valid, wet_mask=wet, cfg=cfg)
    wet_snow_line = _find_first_downward_crossing(profile, float(cfg.crossing_fraction))
    wet_percentile = None
    if n_wet > 0:
        wet_percentile = float(np.nanpercentile(dem[wet], cfg.wet_elevation_percentile))
    if n_valid <= 0:
        gate_reason = "no_valid_pixels"
    elif n_wet <= 0:
        gate_reason = "no_wet_pixels"
    elif wet_snow_line is None:
        gate_reason = "no_crossing_fraction"
    else:
        gate_reason = None
    return WetSnowLineEvaluation(
        wet_snow_line=wet_snow_line,
        wet_elevation_percentile=wet_percentile,
        n_valid=n_valid,
        n_wet=n_wet,
        wet_bands=int(np.count_nonzero(profile["n_wet"] > 0)) if not profile.empty else 0,
        method="crossing_fraction",
        gate_reason=gate_reason,
        profile=profile,
        sector_relative_lines={},
        sector_relative_profiles={},
    )


def _aspect_sector_mask(
    *,
    aspect: np.ndarray,
    slope: np.ndarray,
    mode: str,
    sector: str,
) -> np.ndarray:
    valid = np.isfinite(aspect) & np.isfinite(slope) & (slope > 1e-9)
    if mode == "north_south":
        if sector == "N":
            return valid & ((aspect >= 315.0) | (aspect < 45.0))
        if sector == "S":
            return valid & (aspect >= 135.0) & (aspect < 225.0)
        raise ValueError(f"Unsupported north_south sector: {sector}")
    if sector == "N":
        return valid & ((aspect >= 315.0) | (aspect < 45.0))
    if sector == "E":
        return valid & (aspect >= 45.0) & (aspect < 135.0)
    if sector == "S":
        return valid & (aspect >= 135.0) & (aspect < 225.0)
    if sector == "W":
        return valid & (aspect >= 225.0) & (aspect < 315.0)
    raise ValueError(f"Unsupported four_sectors sector: {sector}")


def compute_wet_snow_line_from_masks(
    *,
    setup_dir: Path,
    project_dir: Path,
    valid_mask: np.ndarray,
    wet_mask: np.ndarray,
) -> WetSnowLineEvaluation:
    cfg = load_wet_snow_line_config(project_dir)
    dem, aspect, slope = _load_dem_and_aspect(str(Path(setup_dir).resolve()))
    result = _evaluate_from_masks(
        dem=dem,
        valid_mask=valid_mask,
        wet_mask=wet_mask,
        cfg=cfg,
    )
    if cfg.aspect_diagnostics == "off":
        return result

    sector_relative_lines: dict[str, float | None] = {}
    sector_relative_profiles: dict[str, pd.DataFrame] = {}
    for sector in _SECTOR_LABELS[cfg.aspect_diagnostics]:
        sector_mask = _aspect_sector_mask(
            aspect=aspect,
            slope=slope,
            mode=cfg.aspect_diagnostics,
            sector=sector,
        )
        sector_result = _evaluate_from_masks(
            dem=dem,
            valid_mask=np.asarray(valid_mask, dtype=bool) & sector_mask,
            wet_mask=np.asarray(wet_mask, dtype=bool) & sector_mask,
            cfg=cfg,
        )
        sector_relative_lines[sector] = _find_sector_relative_crossing(
            sector_result.profile,
            float(cfg.sector_relative_threshold),
        )
        sector_profile = sector_result.profile.copy()
        if sector_profile.empty:
            sector_profile["sector_relative_threshold_fraction"] = pd.Series(dtype=float)
            sector_profile["below_sector_relative_threshold"] = pd.Series(dtype=bool)
            sector_relative_profiles[sector] = sector_profile
            continue
        max_wet = float(sector_profile["f_wet_smooth"].max()) if sector_profile["f_wet_smooth"].notna().any() else np.nan
        relative_threshold = (
            float(cfg.sector_relative_threshold) * max_wet
            if np.isfinite(max_wet) and max_wet > 0.0
            else np.nan
        )
        sector_profile["sector_relative_threshold_fraction"] = relative_threshold
        sector_profile["below_sector_relative_threshold"] = (
            sector_profile["f_wet_smooth"] < relative_threshold if np.isfinite(relative_threshold) else False
        )
        sector_relative_profiles[sector] = sector_profile
    return WetSnowLineEvaluation(
        wet_snow_line=result.wet_snow_line,
        wet_elevation_percentile=result.wet_elevation_percentile,
        n_valid=result.n_valid,
        n_wet=result.n_wet,
        wet_bands=result.wet_bands,
        method=result.method,
        gate_reason=result.gate_reason,
        profile=result.profile,
        sector_relative_lines=sector_relative_lines,
        sector_relative_profiles=sector_relative_profiles,
    )


__all__ = [
    "WetSnowLineConfig",
    "WetSnowLineEvaluation",
    "compute_wet_snow_line_from_fraction_grid",
    "compute_wet_snow_line_from_masks",
    "load_wet_snow_line_config",
]
