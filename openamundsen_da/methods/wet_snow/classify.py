"""
wet_snow/classify.py
Author: Franz Wagner
Date: 2025-11-25
Description:
    Batch classification of wet snow masks from openAMUNDSEN raster outputs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import rasterio
from loguru import logger

from openamundsen_da.core.constants import ENSEMBLE_PRIOR
from openamundsen_da.io.paths import list_member_dirs

_RHO_WATER_DEFAULT = 1000.0  # kg m-3
_MASK_NODATA = np.uint8(255)
_FRACTION_NODATA = -9999.0

_DEPTH_RE = re.compile(r"^snowdepth_daily_(?P<stamp>[^.]+)\.tif$")
_LWC_RE = re.compile(
    r"^liquid_water_content_(?P<layer>\d+)_(?P<start>\d{4}-\d{2}-\d{2}T\d{4})_"
    r"(?P<end>\d{4}-\d{2}-\d{2}T\d{4})\.tif$"
)


def _collect_depth_files(results_dir: Path) -> Dict[str, Path]:
    """
    Collect snow depth rasters indexed by timestamp.

    Parameters
    ----------
    results_dir : Path
        Member results directory containing snowdepth rasters.

    Returns
    -------
    dict
        Mapping timestamp strings to raster paths.
    """
    depth_files: Dict[str, Path] = {}
    for path in sorted(results_dir.glob("snowdepth_daily_*.tif")):
        m = _DEPTH_RE.match(path.name)
        if not m:
            continue
        depth_files[m.group("stamp")] = path
    return depth_files


def _collect_lwc_files(results_dir: Path) -> Dict[str, List[Path]]:
    """
    Group liquid water rasters by their start timestamp.

    Parameters
    ----------
    results_dir : Path
        Member results directory containing liquid water rasters.

    Returns
    -------
    dict
        Mapping YYYY-MM-DDTHHMM strings to a list of layer rasters.
    """
    grouped: Dict[str, List[Path]] = {}
    for path in sorted(results_dir.glob("liquid_water_content_*.tif")):
        m = _LWC_RE.match(path.name)
        if not m:
            continue
        grouped.setdefault(m.group("start"), []).append(path)
    return grouped


def _read_sum_lwc(lw_paths: Sequence[Path]) -> np.ndarray:
    """
    Sum liquid water layers while honoring nodata masks.

    Parameters
    ----------
    lw_paths : sequence of Path
        Paths to raster layers representing liquid water per snow layer.

    Returns
    -------
    ndarray
        Array with the summed liquid water content per pixel.
    """
    total: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    for path in lw_paths:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
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
    depth_path: Path,
    lw_paths: Sequence[Path],
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
    depth_path : Path
        Daily snow depth raster path.
    lw_paths : sequence of Path
        Layered liquid water rasters matching the same timestamp.
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
    depth_match = _DEPTH_RE.match(depth_path.name)
    if not depth_match:
        logger.warning("Skipping unexpected snow depth file {}", depth_path)
        return
    stamp = depth_match.group("stamp")
    mask_path = out_dir / f"{mask_prefix}_{stamp}.tif"
    frac_path = out_dir / f"{fraction_prefix}_{stamp}.tif"

    if mask_path.exists() and not overwrite:
        logger.info("Wet snow mask exists -> skipping {}", mask_path)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(depth_path) as depth_src:
        depth = depth_src.read(1).astype(np.float32)
        profile = depth_src.profile
        depth_nodata = depth_src.nodata

    depth_mask = ~np.isfinite(depth)
    if depth_nodata is not None:
        depth_mask |= depth == depth_nodata
    depth_mask |= depth <= min_depth_m
    depth = np.where(depth_mask, np.nan, depth)

    lw_total = _read_sum_lwc(lw_paths)

    denom = rho_water * depth
    theta = np.full(depth.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(lw_total) & np.isfinite(depth)
    np.divide(lw_total, denom, out=theta, where=valid)

    # Classification: nodata=255, dry=0, wet=1
    wet_mask = np.full(depth.shape, _MASK_NODATA, dtype=np.uint8)
    wet_mask = np.where(np.isfinite(theta), 0, wet_mask)
    wet_mask = np.where(theta >= threshold_frac, 1, wet_mask)

    mask_profile = profile.copy()
    mask_profile.update(dtype="uint8", count=1, nodata=int(_MASK_NODATA), compress="lzw")

    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(wet_mask, 1)
    logger.info("Wrote wet snow mask {}", mask_path)

    if write_fraction:
        theta_percent = theta * 100.0
        frac_array = np.where(np.isfinite(theta_percent), theta_percent, _FRACTION_NODATA)
        frac_profile = profile.copy()
        frac_profile.update(dtype="float32", count=1, nodata=_FRACTION_NODATA, compress="lzw")
        with rasterio.open(frac_path, "w", **frac_profile) as dst:
            dst.write(frac_array.astype(np.float32), 1)
        logger.info("Wrote LWC fraction {}", frac_path)


def _iter_steps(season_dir: Optional[Path], step_dir: Optional[Path]) -> List[Path]:
    """
    Determine which step directories to process.

    Parameters
    ----------
    season_dir : Path or None
        Season directory containing step subfolders.
    step_dir : Path or None
        Specific step directory if only one should be processed.

    Returns
    -------
    list of Path
        Single step when ``step_dir`` is provided, otherwise all season steps.
    """
    if step_dir is not None:
        return [step_dir]
    if season_dir is None:
        raise ValueError("Either --season-dir or --step-dir must be provided.")
    return sorted(p for p in season_dir.glob("step_*") if p.is_dir())


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
    args: argparse.Namespace,
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

    depth_files = _collect_depth_files(results_dir)
    lwc_files = _collect_lwc_files(results_dir)
    if not depth_files:
        logger.warning("No snow depth rasters in {}", results_dir)
        return
    if not lwc_files:
        logger.warning("No liquid water rasters in {}", results_dir)
        return

    out_dir = results_dir / args.output_subdir
    for stamp, depth_path in depth_files.items():
        lw_paths = lwc_files.get(stamp)
        if not lw_paths:
            logger.warning("Missing liquid water rasters for {} in {}", stamp, member_dir)
            continue
        try:
            _compute_fraction(
                depth_path=depth_path,
                lw_paths=lw_paths,
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
            logger.error("Failed to classify {} {}: {}", member_dir.name, stamp, exc)


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
    parser.add_argument("--season-dir", type=Path, help="Season root (contains step_*).")
    parser.add_argument("--step-dir", type=Path, help="Single step directory to process.")
    parser.add_argument("--members", nargs="+", help="Only process listed member directories.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Wet-snow threshold [%] (Rottler et al. 2024 default: 0.1).",
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
        step_dirs = _iter_steps(args.season_dir, args.step_dir)
    except ValueError as exc:
        parser.error(str(exc))
        return 1

    threshold_frac = args.threshold / 100.0
    for step_dir in step_dirs:
        logger.info("Processing step {}", step_dir)
        members = list(_iter_members(step_dir, args.members))
        if not members:
            logger.warning("No members found under {}", step_dir)
            continue
        for member_dir in members:
            logger.info("Classifying wet snow for {}", member_dir)
            _process_member(member_dir, threshold_frac, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
