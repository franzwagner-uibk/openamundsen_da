from __future__ import annotations

import numpy as np
from matplotlib.colors import LightSource
from rasterio import features as rio_features
from rasterio.transform import array_bounds

from openamundsen_da.methods.viz.project_maps.data import StaticContext


def grid_extent(context: StaticContext) -> tuple[float, float, float, float]:
    left, bottom, right, top = array_bounds(
        int(context.roi_mask.shape[0]),
        int(context.roi_mask.shape[1]),
        context.spec.transform,
    )
    return (float(left), float(right), float(bottom), float(top))


def hillshade_extent(context: StaticContext) -> tuple[float, float, float, float]:
    if context.hillshade_dem is not None and context.hillshade_transform is not None:
        left, bottom, right, top = array_bounds(
            int(context.hillshade_dem.shape[0]),
            int(context.hillshade_dem.shape[1]),
            context.hillshade_transform,
        )
        return (float(left), float(right), float(bottom), float(top))
    return grid_extent(context)


def hillshade(context: StaticContext, *, derived_cache: dict[str, np.ndarray] | None = None) -> np.ndarray:
    if derived_cache is not None and "hillshade" in derived_cache:
        return derived_cache["hillshade"]
    dem_source = context.hillshade_dem if context.hillshade_dem is not None else context.dem
    dem = np.asarray(dem_source, dtype=float)
    filled = dem.copy()
    if np.isfinite(filled).any():
        filled[~np.isfinite(filled)] = float(np.nanmedian(filled))
    else:
        filled[:] = 0.0
    transform = context.hillshade_transform if context.hillshade_transform is not None else context.spec.transform
    dx = float(transform.a)
    dy = float(transform.e)
    shades = []
    weights = np.array([0.40, 0.27, 0.20, 0.13], dtype=float)
    for azdeg in (315, 45, 270, 135):
        light = LightSource(azdeg=azdeg, altdeg=45)
        shades.append(
            light.hillshade(
                filled,
                vert_exag=1.3,
                dx=dx,
                dy=dy,
            )
        )
    shade = np.average(np.stack(shades, axis=0), axis=0, weights=weights)
    shade = np.clip(shade, 0.0, 1.0)
    if derived_cache is not None:
        derived_cache["hillshade"] = shade
    return shade


def hillshade_underlay(
    context: StaticContext,
    *,
    derived_cache: dict[str, np.ndarray] | None = None,
) -> np.ma.MaskedArray:
    cache_key = "hillshade_underlay"
    if derived_cache is not None and cache_key in derived_cache:
        return np.ma.masked_invalid(derived_cache[cache_key])

    shade = hillshade(context, derived_cache=derived_cache)
    transform = context.hillshade_transform if context.hillshade_transform is not None else context.spec.transform
    mask = rio_features.rasterize(
        [(geom, 1) for geom in context.roi_gdf.geometry if geom is not None and not geom.is_empty],
        out_shape=shade.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    underlay = np.where(mask, shade, np.nan)
    if derived_cache is not None:
        derived_cache[cache_key] = underlay
    return np.ma.masked_invalid(underlay)
