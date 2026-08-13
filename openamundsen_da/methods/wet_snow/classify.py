"""
wet_snow/classify.py
Author: Franz Wagner
Date: 2025-11-25
Description:
    Batch classification of wet snow masks from openAMUNDSEN raster outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence

import numpy as np
import rasterio
from loguru import logger

from openamundsen_da.core.constants import ENSEMBLE_PRIOR
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    list_member_dirs,
    infer_project_dir,
    infer_setup_dir_from_project,
    find_project_yaml,
)
from openamundsen_da.io.model_grids import (
    ModelGridFrame as DepthEntry,
    configured_model_grid_format,
    model_grid_reader,
)
from openamundsen_da.util.parallel import pick_max_workers, run_tasks_with_pool
from openamundsen_da.util.storage_policy import PERCENT_UINT8_NODATA, percent_to_uint8_nodata

_RHO_WATER_DEFAULT = 1000.0  # kg m-3
_MASK_NODATA = np.uint8(255)
_FRACTION_NODATA = float(PERCENT_UINT8_NODATA)
CLASSIFICATION_METHOD_FRACTION = "liquid_water_fraction"
CLASSIFICATION_METHOD_AMOUNT = "liquid_water_amount"
CLASSIFICATION_METHODS = (CLASSIFICATION_METHOD_FRACTION, CLASSIFICATION_METHOD_AMOUNT)
DEFAULT_LIQUID_WATER_AMOUNT_THRESHOLD_MM = 5.0

def _grid_format_for_step(step_dir: Path) -> str:
    """Read the required model-grid format from canonical setup YAML."""
    step_dir = Path(step_dir)
    project_dir = infer_project_dir(step_dir)
    setup_dir = infer_setup_dir_from_project(project_dir)
    return configured_model_grid_format(setup_dir).value


@dataclass(frozen=True)
class WetSnowClassificationConfig:
    method: str
    threshold_percent: float
    liquid_water_amount_threshold_mm: float


def _coerce_float(raw: object, *, path: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric, got {raw!r}") from exc


def load_wet_snow_classification_config(project_dir: Path) -> WetSnowClassificationConfig:
    """Read model wet-snow classification settings from project YAML."""
    project_yaml = find_project_yaml(project_dir)
    cfg = _read_yaml_file(project_yaml) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping at project YAML root: {project_yaml}")
    da_cfg = cfg.get("data_assimilation")
    if not isinstance(da_cfg, dict):
        raise ValueError(f"Missing required configuration key: {project_yaml} -> data_assimilation")
    wet_cfg = da_cfg.get("wet_snow")
    if not isinstance(wet_cfg, dict):
        raise ValueError(f"Missing required configuration key: {project_yaml} -> data_assimilation.wet_snow")

    cfg_path = "project.data_assimilation.wet_snow"
    method = str(wet_cfg.get("classification_method", CLASSIFICATION_METHOD_FRACTION)).strip()
    if method not in CLASSIFICATION_METHODS:
        raise ValueError(
            f"{cfg_path}.classification_method must be one of {list(CLASSIFICATION_METHODS)}, got {method!r}"
        )

    threshold_percent = float("nan")
    if method == CLASSIFICATION_METHOD_FRACTION:
        if "classification_threshold_percent" not in wet_cfg:
            raise ValueError(
                f"Missing required configuration key: {project_yaml} -> "
                "data_assimilation.wet_snow.classification_threshold_percent"
            )
        threshold_percent = _coerce_float(
            wet_cfg["classification_threshold_percent"],
            path=f"{cfg_path}.classification_threshold_percent",
        )
        if threshold_percent < 0.0:
            raise ValueError(f"{cfg_path}.classification_threshold_percent must be >= 0")

    amount_threshold = _coerce_float(
        wet_cfg.get("liquid_water_amount_threshold_mm", DEFAULT_LIQUID_WATER_AMOUNT_THRESHOLD_MM),
        path=f"{cfg_path}.liquid_water_amount_threshold_mm",
    )
    if amount_threshold < 0.0:
        raise ValueError(f"{cfg_path}.liquid_water_amount_threshold_mm must be >= 0")

    return WetSnowClassificationConfig(
        method=method,
        threshold_percent=threshold_percent,
        liquid_water_amount_threshold_mm=amount_threshold,
    )


def _read_sum_lwc(layers: Sequence[np.ndarray]) -> np.ndarray:
    """
    Sum liquid water layers while honoring nodata masks.

    Parameters
    ----------
    layers : sequence of arrays
        Arrays representing liquid water per snow layer. Invalid cells are NaN.

    Returns
    -------
    ndarray
        Array with the summed liquid water content per pixel.
    """
    total: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    for item in layers:
        data = np.asarray(item, dtype=np.float32)
        invalid = ~np.isfinite(data)
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
    classification_method: str,
    liquid_water_amount_threshold_mm: float,
    out_dir: Path,
    mask_prefix: str,
    fraction_prefix: str,
    write_fraction: bool,
    overwrite: bool,
    rho_water: float,
    min_depth_m: float,
) -> tuple[Path, ...]:
    """
    Compute volumetric LWC fraction and write classification rasters.

    Parameters
    ----------
    depth_entry : DepthEntry
        Daily snow depth grid slice (data + profile + timestamp).
    lw_arrays : sequence of arrays or paths
        Liquid water layers matching the same timestamp (kg m-2, equivalent to mm water).
    threshold_frac : float
        Wet classification threshold in fraction (not percent).
    classification_method : str
        Wet-snow classification method.
    liquid_water_amount_threshold_mm : float
        Absolute liquid-water threshold in mm water equivalent.
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
        return tuple(
            path
            for path in (mask_path, frac_path if write_fraction else None)
            if path is not None and path.is_file()
        )

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
    # Among deep pixels, mark wet according to the configured model-side
    # observation-operator method.
    if classification_method == CLASSIFICATION_METHOD_FRACTION:
        wet_mask = np.where(deep & (theta >= threshold_frac), 1, wet_mask)
    elif classification_method == CLASSIFICATION_METHOD_AMOUNT:
        wet_mask = np.where(deep & (lw_total >= float(liquid_water_amount_threshold_mm)), 1, wet_mask)
    else:
        raise ValueError(f"Unsupported wet-snow classification method: {classification_method!r}")

    mask_profile = profile.copy()
    mask_profile.update(driver="GTiff", dtype="uint8", count=1, nodata=int(_MASK_NODATA), compress="lzw")

    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(wet_mask, 1)
    logger.info("Wrote wet snow mask {}", mask_path)

    if write_fraction:
        theta_percent = theta * 100.0
        frac_array = np.where(np.isfinite(theta_percent), theta_percent, _FRACTION_NODATA)
        frac_profile = profile.copy()
        frac_profile.update(driver="GTiff", dtype="uint8", count=1, nodata=int(PERCENT_UINT8_NODATA), compress="lzw")
        with rasterio.open(frac_path, "w", **frac_profile) as dst:
            dst.write(percent_to_uint8_nodata(frac_array, nodata_value=_FRACTION_NODATA), 1)
        logger.info("Wrote LWC fraction {}", frac_path)
    return (mask_path, frac_path) if write_fraction else (mask_path,)


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
    grid_format: str,
) -> tuple[str, ...]:
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
        return ()

    reader = model_grid_reader(grid_format)
    depth_entries = reader.depth_series(results_dir)
    lwc_files = reader.liquid_water_series(results_dir)
    if not depth_entries:
        logger.warning("No snow depth grids in {}", results_dir)
        return ()
    if not lwc_files:
        logger.warning("No liquid water grids in {}", results_dir)
        return ()

    out_dir = results_dir / args.output_subdir
    outputs: list[str] = []
    for depth in depth_entries:
        lw_paths = lwc_files.get(depth.stamp)
        if not lw_paths:
            logger.warning("Missing liquid water grids for {} in {}", depth.stamp, member_dir)
            continue
        lw_arrays = [np.asarray(array, dtype=np.float32) for array in lw_paths]
        try:
            outputs.extend(
                str(path)
                for path in _compute_fraction(
                depth_entry=depth,
                lw_arrays=lw_arrays,
                threshold_frac=threshold_frac,
                classification_method=args.classification_method,
                liquid_water_amount_threshold_mm=args.liquid_water_amount_threshold_mm,
                out_dir=out_dir,
                mask_prefix=args.mask_prefix,
                fraction_prefix=args.fraction_prefix,
                write_fraction=args.write_fraction,
                overwrite=args.overwrite,
                rho_water=args.water_density,
                min_depth_m=args.min_depth_mm / 1000.0,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to classify {} {}: {}", member_dir.name, depth.stamp, exc)
    return tuple(outputs)


def _classify_members(
    members: Sequence[Path],
    threshold_frac: float,
    args: SimpleNamespace,
    grid_format: str,
    max_workers: int | None,
) -> tuple[Path, ...]:
    """Classify wet snow for all members, optionally in parallel."""
    tasks = [(m, threshold_frac, args, grid_format) for m in members]
    workers = pick_max_workers(max_workers, fallback=len(members), limit=len(members))
    logger.info("Classifying {} member(s) with max_workers={}", len(tasks), workers)
    try:
        results = run_tasks_with_pool(
            _process_member,
            tasks,
            max_workers=workers,
            fallback_workers=len(tasks),
            label="wet_snow",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Wet-snow classification failed: {}", exc)
        raise
    return tuple(Path(path) for member_paths in results for path in member_paths)


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
        help="Wet-snow LWC fraction threshold [%] for classification-method=liquid_water_fraction.",
    )
    parser.add_argument(
        "--classification-method",
        choices=CLASSIFICATION_METHODS,
        default=CLASSIFICATION_METHOD_FRACTION,
        help="Model wet-snow classification method.",
    )
    parser.add_argument(
        "--liquid-water-amount-threshold-mm",
        type=float,
        default=DEFAULT_LIQUID_WATER_AMOUNT_THRESHOLD_MM,
        help="Absolute liquid-water threshold [mm] for classification-method=liquid_water_amount.",
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
    if float(args.threshold) < 0.0:
        parser.error("--threshold must be >= 0")
    if float(args.liquid_water_amount_threshold_mm) < 0.0:
        parser.error("--liquid-water-amount-threshold-mm must be >= 0")

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
        classification_method=str(args.classification_method),
        liquid_water_amount_threshold_mm=float(args.liquid_water_amount_threshold_mm),
    )

    threshold_frac = args.threshold / 100.0
    if args.classification_method == CLASSIFICATION_METHOD_AMOUNT:
        logger.info(
            "Wet-snow classification method={} threshold={:.3f} mm",
            args.classification_method,
            float(args.liquid_water_amount_threshold_mm),
        )
    else:
        logger.info(
            "Wet-snow classification method={} threshold={:.3f} %",
            args.classification_method,
            float(args.threshold),
        )
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
    classification_method: str = CLASSIFICATION_METHOD_FRACTION,
    liquid_water_amount_threshold_mm: float = DEFAULT_LIQUID_WATER_AMOUNT_THRESHOLD_MM,
    output_subdir: str = "wet_snow",
    mask_prefix: str = "wet_snow_mask",
    fraction_prefix: str = "lwc_fraction",
    write_fraction: bool = False,
    overwrite: bool = False,
    water_density: float = _RHO_WATER_DEFAULT,
    min_depth_mm: float = 5.0,
    max_workers: int | None = None,
) -> tuple[Path, ...]:
    """Classify wet-snow masks for a single step directory.

    This programmatic helper mirrors the CLI behavior for one step. It is
    used by the setup pipeline to ensure wet-snow masks are available
    for assimilation when required.
    """
    step_dir = Path(step_dir)
    if classification_method not in CLASSIFICATION_METHODS:
        raise ValueError(
            f"classification_method must be one of {list(CLASSIFICATION_METHODS)}, got {classification_method!r}"
        )
    if float(threshold_percent) < 0.0:
        raise ValueError("threshold_percent must be >= 0")
    if float(liquid_water_amount_threshold_mm) < 0.0:
        raise ValueError("liquid_water_amount_threshold_mm must be >= 0")
    threshold_frac = float(threshold_percent) / 100.0

    args = SimpleNamespace(
        output_subdir=output_subdir,
        mask_prefix=mask_prefix,
        fraction_prefix=fraction_prefix,
        write_fraction=bool(write_fraction),
        overwrite=bool(overwrite),
        water_density=float(water_density),
        min_depth_mm=float(min_depth_mm),
        classification_method=classification_method,
        liquid_water_amount_threshold_mm=float(liquid_water_amount_threshold_mm),
    )

    if classification_method == CLASSIFICATION_METHOD_AMOUNT:
        logger.info(
            "Classifying wet snow for step {} using {} threshold={:.3f} mm",
            step_dir,
            classification_method,
            float(liquid_water_amount_threshold_mm),
        )
    else:
        logger.info(
            "Classifying wet snow for step {} using {} threshold={:.3f} %",
            step_dir,
            classification_method,
            float(threshold_percent),
        )
    members_iter = list(_iter_members(step_dir, members))
    if not members_iter:
        logger.warning("No members found under {}; skipping wet-snow classification.", step_dir)
        return ()
    grid_format = _grid_format_for_step(step_dir)

    return _classify_members(
        members_iter,
        threshold_frac,
        args,
        grid_format,
        max_workers,
    )


if __name__ == "__main__":
    raise SystemExit(cli_main())
