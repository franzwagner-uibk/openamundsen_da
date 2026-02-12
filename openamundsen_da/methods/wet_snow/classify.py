"""
wet_snow/classify.py
Author: Franz Wagner
Date: 2025-11-25
Description:
    Batch classification of wet snow masks from openAMUNDSEN raster outputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
from loguru import logger

from openamundsen_da.core.constants import ENSEMBLE_PRIOR
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    list_member_dirs,
    infer_project_dir,
    infer_setup_dir_from_project,
    find_setup_yaml,
)
from openamundsen_da.util.parallel import pick_max_workers, run_tasks_with_pool

_RHO_WATER_DEFAULT = 1000.0  # kg m-3
_MASK_NODATA = np.uint8(255)
_FRACTION_NODATA = -9999.0

_DEPTH_RE = re.compile(r"^snowdepth_daily_(?P<stamp>[^.]+)\.tif$")
_LWC_RE = re.compile(
    r"^liquid_water_content_(?P<layer>\d+)_(?P<start>\d{4}-\d{2}-\d{2}T\d{4})_"
    r"(?P<end>\d{4}-\d{2}-\d{2}T\d{4})\.tif$"
)


def _grid_format_for_step(step_dir: Path) -> str | None:
    """
    Read output_data.grids.format from setup YAML.

    Returns lower-case format or None if not found/unsupported.
    """
    step_dir = Path(step_dir)
    try:
        project_dir = infer_project_dir(step_dir)
        setup_dir = infer_setup_dir_from_project(project_dir)
        setup_yaml = find_setup_yaml(setup_dir)
    except Exception:
        return None
    try:
        cfg = _read_yaml_file(setup_yaml) or {}
        fmt = (
            cfg.get("output_data", {})
            .get("grids", {})
            .get("format")
        )
        if fmt:
            fmt = str(fmt).lower().strip()
            if fmt in {"geotiff", "netcdf"}:
                return fmt
            if fmt == "ascii":
                logger.warning("output_data.grids.format=ascii not supported for wet-snow classification; falling back to autodetect.")
    except Exception:
        return None
    return None


@dataclass
class DepthEntry:
    stamp: str
    data: np.ndarray
    profile: dict


def _load_depth_entries(results_dir: Path, preferred_format: str | None = None) -> List[DepthEntry]:
    """
    Load daily snow depth grids (GeoTIFF or NetCDF) for a member.

    Returns a list of depth slices with data and profile. GeoTIFFs are
    preferred; if none are found, falls back to NetCDF output_grids.nc.
    """
    entries: List[DepthEntry] = []
    fmt = (preferred_format or "").lower().strip() or None
    if fmt not in {None, "geotiff", "netcdf"}:
        fmt = None

    # GeoTIFF first
    if fmt in {None, "geotiff"}:
        for path in sorted(results_dir.glob("snowdepth_daily_*.tif")):
            m = _DEPTH_RE.match(path.name)
            if not m:
                continue
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                profile = src.profile
            entries.append(DepthEntry(stamp=m.group("stamp"), data=data, profile=profile))
        if entries or fmt == "geotiff":
            return entries

    # NetCDF fallback
    if fmt in {None, "netcdf"}:
        nc_candidates = sorted(results_dir.glob("*.nc"))
        for nc_path in nc_candidates:
            try:
                import xarray as xr  # lazy
            except Exception:
                break
            try:
                with xr.open_dataset(nc_path) as ds:
                    if "snowdepth_daily" not in ds:
                        continue
                    da = ds["snowdepth_daily"]
                    time_dims = [d for d in da.dims if d.startswith("time")]
                    if not time_dims:
                        continue
                    time_dim = time_dims[0]
                    times = pd.to_datetime(ds[time_dim].values)
                    url = f"NETCDF:{nc_path}:snowdepth_daily"
                    with rasterio.open(url) as src:
                        for idx, t in enumerate(times):
                            stamp = t.strftime("%Y-%m-%dT%H%M")
                            data = src.read(idx + 1).astype(np.float32)
                            entries.append(
                                DepthEntry(
                                    stamp=stamp,
                                    data=data,
                                    profile=src.profile,
                                )
                            )
                    break
            except Exception:
                continue

    return entries


def _collect_lwc_files(results_dir: Path, preferred_format: str | None = None) -> Dict[str, List[Path]]:
    """
    Group liquid water rasters by their start timestamp.

    Parameters
    ----------
    results_dir : Path
        Member results directory containing liquid water rasters.
    preferred_format : {"geotiff","netcdf",None}

    Returns
    -------
    dict
        Mapping YYYY-MM-DDTHHMM strings to a list of layer rasters.
    """
    grouped: Dict[str, List[Path]] = {}
    fmt = (preferred_format or "").lower().strip() or None
    if fmt not in {None, "geotiff", "netcdf"}:
        fmt = None

    # GeoTIFFs (if present)
    if fmt in {None, "geotiff"}:
        for path in sorted(results_dir.glob("liquid_water_content_*.tif")):
            m = _LWC_RE.match(path.name)
            if not m:
                continue
            grouped.setdefault(m.group("start"), []).append(path)
        if grouped or fmt == "geotiff":
            return grouped

    # NetCDF fallback
    if fmt in {None, "netcdf"}:
        nc_candidates = sorted(results_dir.glob("*.nc"))
        for nc_path in nc_candidates:
            try:
                import xarray as xr  # lazy
            except Exception:
                break
            try:
                with xr.open_dataset(nc_path) as ds:
                    if "liquid_water_content" not in ds:
                        continue
                    da = ds["liquid_water_content"]
                    if "snow_layer" not in da.dims:
                        continue
                    time_dims = [d for d in da.dims if d.startswith("time")]
                    if not time_dims:
                        continue
                    time_dim = time_dims[0]
                    times = pd.to_datetime(ds[time_dim].values)
                    bounds = ds.get(f"{time_dim}_bounds")
                    n_layers = da.sizes.get("snow_layer", 0)
                    url = f"NETCDF:{nc_path}:liquid_water_content"
                    with rasterio.open(url) as src:
                        for i, t in enumerate(times):
                            stamp_date = pd.to_datetime(t).date()
                            stamp = f"{stamp_date:%Y-%m-%d}T0000"
                            base = i * n_layers
                            indexes = list(range(base + 1, base + n_layers + 1))
                            data = src.read(indexes)
                            grouped[stamp] = [layer.astype(np.float32) for layer in data]
                    break
            except Exception:
                continue

    return grouped


def _read_sum_lwc(lw_paths: Sequence[Path]) -> np.ndarray:
    """
    Sum liquid water layers while honoring nodata masks.

    Parameters
    ----------
    lw_paths : sequence of Path or arrays
        Raster paths or arrays representing liquid water per snow layer.

    Returns
    -------
    ndarray
        Array with the summed liquid water content per pixel.
    """
    total: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    for item in lw_paths:
        if isinstance(item, Path):
            with rasterio.open(item) as src:
                data = src.read(1).astype(np.float32)
                nodata = src.nodata
        else:
            data = np.asarray(item, dtype=np.float32)
            nodata = None
        invalid = ~np.isfinite(data)
        if nodata is not None:
            invalid |= data == nodata
        data = np.where(invalid, 0.0, data)
        if total is None:
            total = data
            valid_mask = ~invalid
        else:
            total += data
            valid_mask &= ~invalid
    if total is None or valid_mask is None:
        raise RuntimeError("No valid liquid water rasters were provided.")
    total = np.where(valid_mask, total, np.nan)
    return total


def _compute_fraction(
    depth_entry: DepthEntry,
    lw_arrays: Sequence[np.ndarray],
    threshold_frac: float,
    out_dir: Path,
    mask_prefix: str,
    fraction_prefix: str,
    write_fraction: bool,
    overwrite: bool,
    rho_water: float,
    min_depth_m: float,
) -> None:
    """
    Compute volumetric LWC fraction and write classification rasters.

    Parameters
    ----------
    depth_entry : DepthEntry
        Daily snow depth grid slice (data + profile + timestamp).
    lw_arrays : sequence of arrays or paths
        Liquid water layers matching the same timestamp (arrays in meters).
    threshold_frac : float
        Wet classification threshold in fraction (not percent).
    out_dir : Path
        Output directory under the member results.
    mask_prefix : str
        Prefix for the mask filename.
    fraction_prefix : str
        Prefix for the fraction (percent) filename.
    write_fraction : bool
        Whether to write the LWC percent raster.
    overwrite : bool
        Whether to overwrite existing rasters.
    rho_water : float
        Density of water (kg m-3).
    min_depth_m : float
        Minimum depth threshold for evaluation (meters).
    """
    stamp = depth_entry.stamp
    mask_path = out_dir / f"{mask_prefix}_{stamp}.tif"
    frac_path = out_dir / f"{fraction_prefix}_{stamp}.tif"

    if mask_path.exists() and not overwrite:
        logger.info("Wet snow mask exists -> skipping {}", mask_path)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    depth = depth_entry.data.astype(np.float32)
    profile = depth_entry.profile.copy()
    depth_nodata = profile.get("nodata")

    # True nodata: invalid depth values (outside grid or flagged nodata).
    depth_invalid = ~np.isfinite(depth)
    if depth_nodata is not None:
        depth_invalid |= depth == depth_nodata

    # Within the model domain, we distinguish shallow/no-snow from deeper snow.
    depth_valid = ~depth_invalid
    shallow = depth_valid & (depth <= min_depth_m)
    deep = depth_valid & (depth > min_depth_m)

    # For theta computation we only use "deep" pixels. Shallow/no-snow pixels
    # are treated as non-wet AOI (dry) but are excluded from the ratio.
    depth_theta = np.where(deep, depth, np.nan)

    # Sum LWC layers. Missing LWC is interpreted as zero (dry) where depth is
    # valid; only true outside-of-domain remains nodata.
    lw_total = _read_sum_lwc(lw_arrays)
    lw_total = np.where(np.isfinite(lw_total), lw_total, 0.0)

    denom = rho_water * depth_theta
    theta = np.full(depth.shape, np.nan, dtype=np.float32)
    valid_theta = np.isfinite(lw_total) & np.isfinite(denom)
    np.divide(lw_total, denom, out=theta, where=valid_theta)

    # Classification: nodata=255 (true missing), dry/land=0, wet=1.
    wet_mask = np.full(depth.shape, _MASK_NODATA, dtype=np.uint8)
    # All depth-valid pixels (shallow or deep) are considered land AOI by
    # default and classified as non-wet (0).
    wet_mask = np.where(depth_valid, 0, wet_mask)
    # Among deep pixels with a valid theta, mark wet where threshold exceeded.
    wet_mask = np.where(deep & (theta >= threshold_frac), 1, wet_mask)

    mask_profile = profile.copy()
    mask_profile.update(driver="GTiff", dtype="uint8", count=1, nodata=int(_MASK_NODATA), compress="lzw")

    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(wet_mask, 1)
    logger.info("Wrote wet snow mask {}", mask_path)

    if write_fraction:
        theta_percent = theta * 100.0
        frac_array = np.where(np.isfinite(theta_percent), theta_percent, _FRACTION_NODATA)
        frac_profile = profile.copy()
        frac_profile.update(driver="GTiff", dtype="float32", count=1, nodata=_FRACTION_NODATA, compress="lzw")
        with rasterio.open(frac_path, "w", **frac_profile) as dst:
            dst.write(frac_array.astype(np.float32), 1)
        logger.info("Wrote LWC fraction {}", frac_path)


def _iter_steps(setup_dir: Optional[Path], step_dir: Optional[Path]) -> List[Path]:
    """
    Determine which step directories to process.

    Parameters
    ----------
    setup_dir : Path or None
        Setup directory containing step subfolders.
    step_dir : Path or None
        Specific step directory if only one should be processed.

    Returns
    -------
    list of Path
        Single step when ``step_dir`` is provided, otherwise all setup steps.
    """
    if step_dir is not None:
        return [step_dir]
    if setup_dir is None:
        raise ValueError("Either --setup-dir or --step-dir must be provided.")
    from openamundsen_da.io.paths import list_step_dirs

    return list_step_dirs(setup_dir)


def _iter_members(
    step_dir: Path,
    member_whitelist: Optional[Sequence[str]],
) -> Iterable[Path]:
    """
    Yield open_loop plus all (or whitelisted) prior members for a step.

    Parameters
    ----------
    step_dir : Path
        Step directory containing ensembles/prior.
    member_whitelist : sequence of str, optional
        Member folder names to keep.

    Returns
    -------
    iterable of Path
        Ordered list where ``open_loop`` precedes member directories.
    """
    base = step_dir / "ensembles" / ENSEMBLE_PRIOR
    if not base.exists():
        logger.warning("Prior ensemble directory missing: {}", base)
        return []
    members = list_member_dirs(step_dir / "ensembles", ENSEMBLE_PRIOR)
    open_loop = base / "open_loop"
    ordered: List[Path] = []
    if open_loop.exists():
        ordered.append(open_loop)
    if member_whitelist:
        whitelist = set(member_whitelist)
        members = [m for m in members if m.name in whitelist]
    ordered.extend(members)
    return ordered


def _process_member(
    member_dir: Path,
    threshold_frac: float,
    args: SimpleNamespace,
    grid_format: str | None,
) -> None:
    """
    Run the wet-snow classification for a single member directory.

    Parameters
    ----------
    member_dir : Path
        Member or open_loop directory holding a ``results`` subfolder.
    threshold_frac : float
        Wet-snow threshold expressed as a fraction (not percent).
    args : argparse.Namespace
        Parsed CLI arguments (shared options).
    """
    results_dir = member_dir / "results"
    if not results_dir.is_dir():
        logger.warning("Results directory missing for {}", member_dir)
        return

    depth_entries = _load_depth_entries(results_dir, preferred_format=grid_format)
    lwc_files = _collect_lwc_files(results_dir, preferred_format=grid_format)
    if not depth_entries:
        logger.warning("No snow depth grids in {}", results_dir)
        return
    if not lwc_files:
        logger.warning("No liquid water grids in {}", results_dir)
        return

    out_dir = results_dir / args.output_subdir
    for depth in depth_entries:
        lw_paths = lwc_files.get(depth.stamp)
        if not lw_paths:
            logger.warning("Missing liquid water grids for {} in {}", depth.stamp, member_dir)
            continue
        # Convert GeoTIFF LWC paths to arrays on the fly
        if lw_paths and isinstance(lw_paths[0], Path):
            arrays = []
            for p in lw_paths:
                with rasterio.open(p) as src:
                    arrays.append(src.read(1).astype(np.float32))
            lw_arrays = arrays
        else:
            lw_arrays = [np.asarray(a, dtype=np.float32) for a in lw_paths]
        try:
            _compute_fraction(
                depth_entry=depth,
                lw_arrays=lw_arrays,
                threshold_frac=threshold_frac,
                out_dir=out_dir,
                mask_prefix=args.mask_prefix,
                fraction_prefix=args.fraction_prefix,
                write_fraction=args.write_fraction,
                overwrite=args.overwrite,
                rho_water=args.water_density,
                min_depth_m=args.min_depth_mm / 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to classify {} {}: {}", member_dir.name, depth.stamp, exc)


def _classify_members(
    members: Sequence[Path],
    threshold_frac: float,
    args: SimpleNamespace,
    grid_format: str | None,
    max_workers: int | None,
) -> None:
    """Classify wet snow for all members, optionally in parallel."""
    tasks = [(m, threshold_frac, args, grid_format) for m in members]
    workers = pick_max_workers(max_workers, fallback=len(members), limit=len(members))
    logger.info("Classifying {} member(s) with max_workers={}", len(tasks), workers)
    try:
        run_tasks_with_pool(_process_member, tasks, max_workers=workers, fallback_workers=len(tasks), label="wet_snow")
    except Exception as exc:  # noqa: BLE001
        logger.error("Wet-snow classification failed: {}", exc)
        raise


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for the wet-snow classification CLI.
    """
    parser = argparse.ArgumentParser(
        prog="oa-da-wet-snow",
        description="Classify volumetric wet snow masks from openAMUNDSEN outputs.",
    )
    parser.add_argument("--setup-dir", type=Path, help="Setup root (contains steps/step_*).")
    parser.add_argument("--step-dir", type=Path, help="Single step directory to process.")
    parser.add_argument("--members", nargs="+", help="Only process listed member directories.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Wet-snow threshold [%] (Rottler et al. 2024 default: 0.1).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Worker cap (overrides MAX_WORKERS env). Defaults to min(CPU, #members).",
    )
    parser.add_argument("--output-subdir", default="wet_snow", help="Subdirectory under results/.")
    parser.add_argument("--mask-prefix", default="wet_snow_mask", help="Filename prefix for masks.")
    parser.add_argument("--fraction-prefix", default="lwc_fraction", help="Prefix for fraction outputs.")
    parser.add_argument("--write-fraction", action="store_true", help="Write fraction rasters (percent).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--water-density", type=float, default=_RHO_WATER_DEFAULT, help="Water density (kg m-3).")
    parser.add_argument("--min-depth-mm", type=float, default=5.0, help="Minimum snow depth (mm) to evaluate.")
    return parser


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point used by `python -m ...wet_snow.classify`.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument list for testing; defaults to ``sys.argv`` when omitted.

    Returns
    -------
    int
        Zero on success, non-zero when parsing or processing fails.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        step_dirs = _iter_steps(args.setup_dir, args.step_dir)
    except ValueError as exc:
        parser.error(str(exc))
        return 1

    worker_args = SimpleNamespace(
        output_subdir=args.output_subdir,
        mask_prefix=args.mask_prefix,
        fraction_prefix=args.fraction_prefix,
        write_fraction=bool(args.write_fraction),
        overwrite=bool(args.overwrite),
        water_density=float(args.water_density),
        min_depth_mm=float(args.min_depth_mm),
    )

    threshold_frac = args.threshold / 100.0
    for step_dir in step_dirs:
        logger.info("Processing step {}", step_dir)
        grid_format = _grid_format_for_step(step_dir)
        members = list(_iter_members(step_dir, args.members))
        if not members:
            logger.warning("No members found under {}", step_dir)
            continue
        _classify_members(members, threshold_frac, worker_args, grid_format, args.max_workers)

    return 0


def classify_step_wet_snow(
    step_dir: Path,
    *,
    members: Optional[Sequence[str]] = None,
    threshold_percent: float = 0.1,
    output_subdir: str = "wet_snow",
    mask_prefix: str = "wet_snow_mask",
    fraction_prefix: str = "lwc_fraction",
    write_fraction: bool = False,
    overwrite: bool = False,
    water_density: float = _RHO_WATER_DEFAULT,
    min_depth_mm: float = 5.0,
    max_workers: int | None = None,
) -> None:
    """Classify wet-snow masks for a single step directory.

    This programmatic helper mirrors the CLI behavior for one step. It is
    used by the setup pipeline to ensure wet-snow masks are available
    for assimilation when required.
    """
    step_dir = Path(step_dir)
    threshold_frac = float(threshold_percent) / 100.0

    args = SimpleNamespace(
        output_subdir=output_subdir,
        mask_prefix=mask_prefix,
        fraction_prefix=fraction_prefix,
        write_fraction=bool(write_fraction),
        overwrite=bool(overwrite),
        water_density=float(water_density),
        min_depth_mm=float(min_depth_mm),
    )

    logger.info("Classifying wet snow for step {}", step_dir)
    members_iter = list(_iter_members(step_dir, members))
    if not members_iter:
        logger.warning("No members found under {}; skipping wet-snow classification.", step_dir)
        return
    grid_format = _grid_format_for_step(step_dir)

    _classify_members(members_iter, threshold_frac, args, grid_format, max_workers)


if __name__ == "__main__":
    raise SystemExit(cli_main())
