"""Merge compact sub-domain outputs into global hard mosaics."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from loguru import logger

from openamundsen_da.io.paths import (
    list_steps_sorted,
    project_maps_root,
    project_plots_maps_collection_pdf_path,
)
from openamundsen_da.methods.viz.maps.generated import GENERATED_DA_MAPS_SUBDIR
from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta
from openamundsen_da.subdomain.event_support import resolve_subdomain_event_plan
from openamundsen_da.subdomain.status import save_stage, terminal_status
from openamundsen_da.util.da_output import (
    output_retention_mode,
    write_da_output_grids,
    validate_compact_output_file,
)
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.roi_grid import load_setup_roi_mask
from openamundsen_da.util.run_mode import ensure_run_mode
from openamundsen_da.util.storage_policy import da_summary_netcdf_encoding, preserved_netcdf_encoding
from openamundsen_da.util.yaml_utils import read_yaml_mapping


class CompactCleanupSafetyError(RuntimeError):
    """Raised when compact cleanup would remove files before artifacts are complete."""


@contextmanager
def _atomic_output(path: Path):
    """Yield a same-directory temporary path and replace the target on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
    )
    os.close(fd)
    tmp_path = Path(raw_tmp)
    tmp_path.unlink(missing_ok=True)
    try:
        yield tmp_path
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _tracked_merge(operation):
    """Persist merge stage transitions around a DA or model mosaic operation."""

    @wraps(operation)
    def wrapped(*, manifest_path: Path, **kwargs):
        manifest_path = Path(manifest_path).resolve()
        manifest = SubdomainManifest.load(manifest_path)
        save_stage(manifest, manifest_path, "merge", "running")
        try:
            outputs = operation(manifest_path=manifest_path, **kwargs)
        except BaseException as exc:
            current = SubdomainManifest.load(manifest_path)
            save_stage(
                current,
                manifest_path,
                "merge",
                terminal_status(exc),
                error=str(exc),
            )
            raise
        current = SubdomainManifest.load(manifest_path)
        save_stage(current, manifest_path, "merge", "completed", outputs=outputs)
        return outputs

    return wrapped


def _latest_step_dir(sub: SubdomainMeta) -> Path | None:
    try:
        steps = list_steps_sorted(sub.project_dir)
    except Exception:
        steps = []
    if steps:
        return steps[-1]
    steps_root = sub.project_dir / "steps"
    fallback = sorted(p for p in steps_root.glob("step_*") if p.is_dir()) if steps_root.is_dir() else []
    return fallback[-1] if fallback else None


def _result_sources(sub: SubdomainMeta) -> list[tuple[str, Path]]:
    """Return compact result directories from the latest step.

    We keep open-loop output names unchanged and prefix member outputs with
    the member id, so merged filenames remain unique.
    """
    step_dir = _latest_step_dir(sub)
    if step_dir is None:
        return []
    prior_root = step_dir / "ensembles" / "prior"
    if not prior_root.is_dir():
        return []

    out: list[tuple[str, Path]] = []
    open_loop_results = prior_root / "open_loop" / "results"
    if open_loop_results.is_dir():
        out.append(("open_loop", open_loop_results))
    for member_dir in sorted(prior_root.glob("member_*")):
        if not member_dir.is_dir():
            continue
        res_dir = member_dir / "results"
        if res_dir.is_dir():
            out.append((member_dir.name, res_dir))
    return out


def _compact_da_summary(sub: SubdomainMeta) -> Path | None:
    """Return sub-domain compact DA summary if available."""
    candidate = sub.project_dir / "results" / "grids" / "da_output_grids.nc"
    return candidate if candidate.is_file() else None


def _load_roi(sub: SubdomainMeta) -> np.ndarray:
    with rasterio.open(sub.roi_raster_path) as ds:
        return ds.read(1).astype(bool)


def _window_slices(
    sub: SubdomainMeta,
    data_shape: Tuple[int, int],
    global_shape: Tuple[int, int],
) -> Tuple[slice, slice]:
    if data_shape == global_shape:
        return slice(0, global_shape[0]), slice(0, global_shape[1])
    return (
        slice(sub.window.row_off, sub.window.row_off + data_shape[0]),
        slice(sub.window.col_off, sub.window.col_off + data_shape[1]),
    )


def _expected_coverage_mask(
    manifest: SubdomainManifest,
    selected_ids: Iterable[str],
    global_shape: Tuple[int, int],
) -> np.ndarray:
    selected_ids = list(selected_ids)
    if set(selected_ids) == set(manifest.subdomains.keys()):
        try:
            roi_mask, _, roi_path = load_setup_roi_mask(manifest.setup_dir, ensure_grid=False)
            if roi_mask.shape == global_shape:
                logger.info("Using setup ROI grid coverage mask {}", roi_path)
                return roi_mask
            logger.warning(
                "Setup ROI grid shape mismatch for coverage check: {} vs {} (falling back to sub-domain ROI union)",
                roi_mask.shape,
                global_shape,
            )
        except Exception as exc:
            logger.warning(
                "Could not load setup ROI grid for coverage check (falling back to sub-domain ROI union): {}",
                exc,
            )

    expected = np.zeros(global_shape, dtype=bool)
    for sid in selected_ids:
        sub = manifest.subdomains[sid]
        roi = _load_roi(sub)
        sl_r, sl_c = _window_slices(sub, roi.shape, global_shape)
        if roi.shape == global_shape:
            expected |= roi
            continue
        view = expected[sl_r, sl_c]
        if view.shape != roi.shape:
            logger.warning("Coverage ROI shape mismatch for {}: {} vs {}", sid, roi.shape, view.shape)
            continue
        view |= roi
        expected[sl_r, sl_c] = view
    return expected


def _validate_coverage_or_raise(
    *,
    target_name: str,
    expected_mask: np.ndarray,
    data_mask: np.ndarray,
    sliver_tol_px: int,
) -> None:
    uncovered = expected_mask & (~data_mask)
    uncovered_count = int(np.count_nonzero(uncovered))
    if uncovered_count <= int(sliver_tol_px):
        if uncovered_count > 0:
            logger.warning(
                "Coverage check: {} has {} uncovered expected pixel(s) within tolerance {}",
                target_name,
                uncovered_count,
                sliver_tol_px,
            )
        return
    raise ValueError(
        f"Coverage check failed for {target_name}: {uncovered_count} uncovered expected pixel(s) "
        f"(tolerance {sliver_tol_px})."
    )


@_tracked_merge
def merge_grids(
    *,
    manifest_path: Path,
    subdomains: Optional[Iterable[str]] = None,
    out_dir: Optional[Path] = None,
    coverage_sliver_tol_px: int = 4,
) -> List[Path]:
    """Merge sub-domain grid outputs into global hard mosaics."""
    manifest = SubdomainManifest.load(manifest_path)
    if str(getattr(manifest, "run_mode", "")).lower() != "subdomain":
        raise ValueError(f"Manifest at {manifest_path} is not marked as run_mode='subdomain'.")
    ensure_run_mode(manifest.project_dir, expected="subdomain", write_if_missing=False)
    selected_ids = list(subdomains) if subdomains else list(manifest.subdomains.keys())
    unknown = [sid for sid in selected_ids if sid not in manifest.subdomains]
    if unknown:
        raise ValueError(f"Sub-domains not in manifest: {', '.join(unknown)}")
    if set(selected_ids) == set(manifest.subdomains):
        resolve_subdomain_event_plan(manifest, require_artifacts=True)

    global_shape = (manifest.grid_rows, manifest.grid_cols)
    global_transform = Affine(*manifest.grid_transform)
    expected_mask = _expected_coverage_mask(manifest, selected_ids, global_shape)

    out_base = out_dir or (manifest.project_dir / "results" / "grids")
    out_base.mkdir(parents=True, exist_ok=True)

    compact_entries: list[tuple[SubdomainMeta, Path]] = []
    missing_compact_ids: list[str] = []
    for sid in selected_ids:
        sub = manifest.subdomains[sid]
        compact_da = _compact_da_summary(sub)
        if compact_da is None:
            missing_compact_ids.append(sid)
        else:
            compact_entries.append((sub, compact_da))

    written: List[Path] = []
    if compact_entries:
        if missing_compact_ids:
            raise FileNotFoundError(
                "Compact DA output summary da_output_grids.nc is missing for sub-domain(s): "
                f"{', '.join(missing_compact_ids)}"
            )
        for _subdomain, compact_path in compact_entries:
            validate_compact_output_file(
                project_dir=manifest.project_dir,
                output_nc=compact_path,
            )
        merged_nc = _merge_netcdf(
            output_name="da_output_grids.nc",
            nc_paths=compact_entries,
            global_shape=global_shape,
            manifest=manifest,
            out_dir=out_base,
            expected_mask=expected_mask,
            sliver_tol_px=int(coverage_sliver_tol_px),
        )
        written.append(merged_nc)
        da_summary_written = merged_nc.is_file()
        if da_summary_written:
            validate_compact_output_file(
                project_dir=manifest.project_dir,
                output_nc=merged_nc,
            )
            logger.info("Using merged compact DA output summary {}", merged_nc)

        if output_retention_mode(manifest.project_dir) == "compact":
            logger.info("Compact cleanup is deferred until strict top-level rendering and report validation succeed.")
        return written

    tif_groups: Dict[str, List[Tuple[SubdomainMeta, Path]]] = {}
    nc_groups: Dict[str, List[Tuple[SubdomainMeta, Path]]] = {}
    for sid in selected_ids:
        sub = manifest.subdomains[sid]
        sources = _result_sources(sub)
        if not sources:
            logger.warning("No compact result sources discovered for {} under {}", sid, sub.project_dir)
            continue
        for source_label, res_dir in sources:
            prefix = "" if source_label == "open_loop" else f"{source_label}_"
            for tif in sorted(res_dir.glob("*.tif")):
                tif_groups.setdefault(f"{prefix}{tif.name}", []).append((sub, tif))
            for nc in sorted(res_dir.glob("*.nc")):
                nc_groups.setdefault(f"{prefix}{nc.name}", []).append((sub, nc))

    if not tif_groups and not nc_groups:
        logger.warning("No compact grid outputs found to merge in selected sub-domains.")
        return written

    if tif_groups:
        written.extend(
            _merge_tifs(
                tif_groups=tif_groups,
                global_shape=global_shape,
                transform=global_transform,
                crs=manifest.crs,
                out_dir=out_base,
                expected_mask=expected_mask,
                sliver_tol_px=int(coverage_sliver_tol_px),
            )
        )
    compact_da_summary_merged = False
    if nc_groups:
        for nc_name, entries in sorted(nc_groups.items()):
            merged_nc = _merge_netcdf(
                output_name=nc_name,
                nc_paths=entries,
                global_shape=global_shape,
                manifest=manifest,
                out_dir=out_base,
                expected_mask=expected_mask,
                sliver_tol_px=int(coverage_sliver_tol_px),
            )
            written.append(merged_nc)
            if nc_name == "da_output_grids.nc" and merged_nc.is_file():
                compact_da_summary_merged = True

    da_summary_written = False
    da_summary_path = out_base / "da_output_grids.nc"
    open_loop_nc = out_base / "output_grids.nc"
    member_ncs = sorted(out_base.glob("member_*_output_grids.nc"))
    if compact_da_summary_merged and da_summary_path.is_file():
        da_summary_written = True
        logger.info("Using merged compact DA output summary {}", da_summary_path)
    elif da_summary_path.is_file() and (not open_loop_nc.is_file() or not member_ncs):
        da_summary_written = True
        logger.info("Using existing DA output summary {}", da_summary_path)
    else:
        try:
            da_path = write_da_output_grids(
                open_loop_nc=open_loop_nc,
                member_ncs=member_ncs,
                output_nc=da_summary_path,
            )
            da_summary_written = da_path is not None
            if da_path is not None:
                written.append(da_path)
        except Exception as exc:
            logger.warning("DA output grid summary failed: {}", exc)
        if (not da_summary_written) and da_summary_path.is_file():
            da_summary_written = True
            logger.info("Using existing DA output summary {}", da_summary_path)

    if output_retention_mode(manifest.project_dir) == "compact":
        logger.info("Compact cleanup is deferred until strict top-level rendering and report validation succeed.")
    if da_summary_written:
        validate_compact_output_file(
            project_dir=manifest.project_dir,
            output_nc=da_summary_path,
        )
    return written


def _planned_da_map_paths(project_dir: Path) -> list[Path]:
    maps_root = project_maps_root(project_dir) / GENERATED_DA_MAPS_SUBDIR
    return [maps_root / f"da_{idx}.png" for idx, _event in enumerate(load_assimilation_events(project_dir), start=1)]


def _missing_compact_cleanup_artifacts(project_dir: Path, out_base: Path) -> list[Path | str]:
    missing: list[Path | str] = []
    da_summary_path = out_base / "da_output_grids.nc"
    if not da_summary_path.is_file():
        missing.append(da_summary_path)

    report_path = project_plots_maps_collection_pdf_path(project_dir)
    if not report_path.is_file():
        missing.append(report_path)

    try:
        missing.extend(path for path in _planned_da_map_paths(project_dir) if not path.is_file())
    except Exception as exc:
        missing.append(f"assimilation event map plan could not be checked: {exc}")

    return missing


def validate_compact_cleanup_artifacts_ready(
    *,
    manifest: SubdomainManifest,
    out_dir: Optional[Path] = None,
) -> None:
    """Fail unless merge and render completed and their required artifacts exist."""
    project_dir = Path(manifest.project_dir)
    out_base = Path(out_dir) if out_dir is not None else (project_dir / "results" / "grids")
    missing: list[Path | str] = []

    for stage in ("merge", "render"):
        status = str((manifest.stages.get(stage) or {}).get("status", "pending"))
        if status != "completed":
            missing.append(f"manifest stage {stage!r} is {status!r}, expected 'completed'")

    missing.extend(_missing_compact_cleanup_artifacts(project_dir, out_base))
    if missing:
        formatted = "\n".join(f"- {item}" for item in missing)
        raise CompactCleanupSafetyError(
            "Compact grid cleanup refused because required generated artifacts are missing:\n"
            f"{formatted}\n"
            "Regenerate maps/report first, or keep raw grid support files in place."
        )


def _manifest_owned_compact_artifacts(manifest: SubdomainManifest, out_base: Path) -> list[Path]:
    """Return only transient grid files owned by the selected manifest."""
    artifacts: set[Path] = set()
    resolved_out_base = out_base.resolve()
    merge_outputs = (manifest.stages.get("merge") or {}).get("outputs")
    if not isinstance(merge_outputs, list):
        raise CompactCleanupSafetyError(
            "Manifest merge stage has no output inventory; refusing compact cleanup."
        )
    for raw_path in merge_outputs:
        path = Path(str(raw_path)).resolve()
        if not path.is_relative_to(resolved_out_base):
            raise CompactCleanupSafetyError(
                f"Manifest merge output escapes the selected grids directory: {path}"
            )
        if path.name != "da_output_grids.nc" and path.is_file():
            artifacts.add(path)

    step_patterns = (
        "steps/step_*/ensembles/prior/member_*/results/output_grids*.nc",
        "steps/step_*/ensembles/prior/open_loop/results/output_grids*.nc",
        "steps/step_*/ensembles/prior/member_*/results/**/*.tif",
        "steps/step_*/ensembles/prior/member_*/results/**/*.tiff",
        "steps/step_*/ensembles/prior/open_loop/results/**/*.tif",
        "steps/step_*/ensembles/prior/open_loop/results/**/*.tiff",
        "steps/step_*/ensembles/posterior/member_*/results/output_grids*.nc",
        "steps/step_*/ensembles/posterior/member_*/results/**/*.tif",
        "steps/step_*/ensembles/posterior/member_*/results/**/*.tiff",
    )
    manifest_root = manifest.subdomain_root.resolve()
    for subdomain in manifest.subdomains.values():
        project_root = subdomain.project_dir.resolve()
        if not project_root.is_relative_to(manifest_root):
            raise CompactCleanupSafetyError(
                f"Manifest project escapes its subdomain root: {project_root}"
            )
        compact_summary = project_root / "results" / "grids" / "da_output_grids.nc"
        if compact_summary.is_file():
            artifacts.add(compact_summary.resolve())
        for pattern in step_patterns:
            for path in project_root.glob(pattern):
                resolved = path.resolve()
                if path.is_file() and resolved.is_relative_to(project_root):
                    artifacts.add(resolved)
    return sorted(artifacts)


def cleanup_compact_grid_artifacts(
    *,
    manifest_path: Path,
    out_dir: Optional[Path] = None,
) -> tuple[list[Path], int]:
    """Delete manifest-owned transient grids after strict downstream validation."""
    manifest = SubdomainManifest.load(manifest_path)
    retention_mode = output_retention_mode(manifest.project_dir)
    if retention_mode != "compact":
        save_stage(manifest, manifest_path, "cleanup", "skipped")
        logger.info("Skipping compact grid cleanup because output retention is {}.", retention_mode)
        return [], 0

    out_base = out_dir or (manifest.project_dir / "results" / "grids")
    validate_compact_cleanup_artifacts_ready(manifest=manifest, out_dir=out_base)
    artifacts = _manifest_owned_compact_artifacts(manifest, Path(out_base))
    save_stage(manifest, manifest_path, "cleanup", "running", outputs=artifacts)

    deleted: list[Path] = []
    bytes_freed = 0
    try:
        for path in artifacts:
            size = path.stat().st_size
            path.unlink()
            deleted.append(path)
            bytes_freed += size
    except BaseException as exc:
        current = SubdomainManifest.load(manifest_path)
        save_stage(
            current,
            manifest_path,
            "cleanup",
            terminal_status(exc),
            outputs=deleted,
            error=str(exc),
        )
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise CompactCleanupSafetyError(f"Compact cleanup failed at {path}: {exc}") from exc

    current = SubdomainManifest.load(manifest_path)
    save_stage(current, manifest_path, "cleanup", "completed", outputs=deleted)
    return deleted, bytes_freed


def _configured_model_grid_format(manifest: SubdomainManifest) -> str:
    cfg = read_yaml_mapping(
        manifest.setup_yaml,
        error_cls=ValueError,
        context="Plain-model setup YAML root",
    )
    output_cfg = cfg.get("output_data") or {}
    grids_cfg = output_cfg.get("grids") or {}
    grid_format = str(grids_cfg.get("format", "netcdf")).strip().lower()
    if grid_format not in {"netcdf", "geotiff"}:
        raise ValueError(
            "Plain-model subdomain merge supports output_data.grids.format "
            f"'netcdf' or 'geotiff', got {grid_format!r}."
        )
    return grid_format


@_tracked_merge
def merge_model_grids(
    *,
    manifest_path: Path,
    subdomains: Optional[Iterable[str]] = None,
    out_dir: Optional[Path] = None,
    coverage_sliver_tol_px: int = 4,
) -> List[Path]:
    """Merge plain openAMUNDSEN model grid outputs from each sub-domain."""
    manifest = SubdomainManifest.load(manifest_path)
    if str(getattr(manifest, "run_mode", "")).lower() != "model":
        raise ValueError(f"Manifest at {manifest_path} is not marked as run_mode='model'.")

    selected_ids = list(subdomains) if subdomains is not None else list(manifest.subdomains.keys())
    unknown = [sid for sid in selected_ids if sid not in manifest.subdomains]
    if unknown:
        raise ValueError(f"Sub-domains not in manifest: {', '.join(unknown)}")
    if not selected_ids:
        return []

    global_shape = (manifest.grid_rows, manifest.grid_cols)
    global_transform = Affine(*manifest.grid_transform)
    expected_mask = _expected_coverage_mask(manifest, selected_ids, global_shape)

    out_base = out_dir or (manifest.subdomain_root / "results" / "grids")
    out_base.mkdir(parents=True, exist_ok=True)

    grid_format = _configured_model_grid_format(manifest)
    tif_groups: Dict[str, List[Tuple[SubdomainMeta, Path]]] = {}
    nc_groups: Dict[str, List[Tuple[SubdomainMeta, Path]]] = {}
    owners_by_name: Dict[str, set[str]] = {}
    for sid in selected_ids:
        sub = manifest.subdomains[sid]
        grid_dir = sub.setup_dir / "results" / "grids"
        if not grid_dir.is_dir():
            raise FileNotFoundError(f"Missing model grid output directory for {sid}: {grid_dir}")
        tif_paths = sorted([*grid_dir.glob("*.tif"), *grid_dir.glob("*.tiff")])
        nc_paths = sorted(grid_dir.glob("*.nc"))
        unexpected = tif_paths if grid_format == "netcdf" else nc_paths
        if unexpected:
            names = ", ".join(path.name for path in unexpected[:5])
            raise ValueError(
                f"Mixed model grid artifacts for {sid}: setup selects {grid_format!r}, "
                f"but also found {names}. Remove stale outputs and rerun the model stage."
            )

        if grid_format == "geotiff":
            for tif in tif_paths:
                tif_groups.setdefault(tif.name, []).append((sub, tif))
                owners_by_name.setdefault(tif.name, set()).add(sid)
        else:
            for nc in nc_paths:
                nc_groups.setdefault(nc.name, []).append((sub, nc))
                owners_by_name.setdefault(nc.name, set()).add(sid)

    all_groups = {**tif_groups, **nc_groups}
    if not all_groups:
        raise FileNotFoundError(
            f"No {grid_format} model grid outputs found below selected sub-domain results/grids directories "
            f"in {manifest.subdomain_root}."
        )

    selected_set = set(selected_ids)
    for name, owners in sorted(owners_by_name.items()):
        missing = sorted(selected_set - owners)
        if missing:
            raise FileNotFoundError(
                f"Model grid output {name!r} is missing for sub-domain(s): {', '.join(missing)}"
            )

    expected_names = set(all_groups)
    existing_outputs = sorted(
        path
        for path in [*out_base.glob("*.nc"), *out_base.glob("*.tif"), *out_base.glob("*.tiff")]
        if path.is_file()
    )
    unmanaged = [path for path in existing_outputs if path.name not in expected_names]
    if unmanaged:
        names = ", ".join(path.name for path in unmanaged[:5])
        raise ValueError(
            f"Ambiguous existing merged model outputs in {out_base}: {names}. "
            "Move or remove outputs that are not part of the current manifest before merging."
        )

    written: List[Path] = []
    if tif_groups:
        written.extend(
            _merge_tifs(
                tif_groups=tif_groups,
                global_shape=global_shape,
                transform=global_transform,
                crs=manifest.crs,
                out_dir=out_base,
                expected_mask=expected_mask,
                sliver_tol_px=int(coverage_sliver_tol_px),
            )
        )
    for nc_name, entries in sorted(nc_groups.items()):
        written.append(
            _merge_netcdf(
                output_name=nc_name,
                nc_paths=entries,
                global_shape=global_shape,
                manifest=manifest,
                out_dir=out_base,
                expected_mask=expected_mask,
                sliver_tol_px=int(coverage_sliver_tol_px),
            )
        )
    return written


def _merge_tifs(
    *,
    tif_groups: Dict[str, List[Tuple[SubdomainMeta, Path]]],
    global_shape: Tuple[int, int],
    transform: Affine,
    crs: Optional[str],
    out_dir: Path,
    expected_mask: np.ndarray,
    sliver_tol_px: int,
) -> List[Path]:
    outputs: List[Path] = []
    for fname, entries in sorted(tif_groups.items()):
        with rasterio.open(entries[0][1]) as ds0:
            dtype = ds0.dtypes[0]
            nodata = ds0.nodata
        data_global = np.full(global_shape, np.nan, dtype=np.float64)

        for sub, tif_path in entries:
            roi = _load_roi(sub)
            with rasterio.open(tif_path) as ds:
                arr = ds.read(1).astype(np.float64)
                if ds.nodata is not None:
                    arr[arr == ds.nodata] = np.nan
            sl_r, sl_c = _window_slices(sub, arr.shape, global_shape)
            mask = roi
            if mask.shape != arr.shape:
                if mask.shape == global_shape:
                    mask = mask[sl_r, sl_c]
                else:
                    logger.warning("ROI shape {} does not match data {} for {}", mask.shape, arr.shape, sub.id)
                    mask = np.ones_like(arr, dtype=bool)
            arr = np.where(mask, arr, np.nan)

            dest = data_global[sl_r, sl_c]
            replace = np.isnan(dest) & ~np.isnan(arr)
            dest[replace] = arr[replace]
            data_global[sl_r, sl_c] = dest

        _validate_coverage_or_raise(
            target_name=fname,
            expected_mask=expected_mask,
            data_mask=~np.isnan(data_global),
            sliver_tol_px=sliver_tol_px,
        )

        out_path = out_dir / fname
        nd = nodata if nodata is not None else (-9999.0 if np.issubdtype(np.dtype(dtype), np.floating) else -9999)
        meta = {
            "driver": "GTiff",
            "dtype": dtype,
            "nodata": nd,
            "width": global_shape[1],
            "height": global_shape[0],
            "count": 1,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }
        with _atomic_output(out_path) as tmp_path:
            with rasterio.open(tmp_path, "w", **meta) as dst:
                if np.issubdtype(np.dtype(dtype), np.integer):
                    fill_val = int(nd)
                    write_arr = np.where(np.isnan(data_global), fill_val, np.round(data_global)).astype(dtype)
                    dst.write(write_arr, 1)
                    dst.nodata = fill_val
                else:
                    dst.write(np.where(np.isnan(data_global), nd, data_global).astype(dtype), 1)
        outputs.append(out_path)
        logger.info("Wrote merged grid {}", out_path)
    return outputs


def _merge_netcdf(
    *,
    output_name: str,
    nc_paths: List[Tuple[SubdomainMeta, Path]],
    global_shape: Tuple[int, int],
    manifest: SubdomainManifest,
    out_dir: Path,
    expected_mask: np.ndarray,
    sliver_tol_px: int,
) -> Path:
    _, base_nc = nc_paths[0]
    ds_template = xr.open_dataset(base_nc)

    rows, cols = global_shape
    transform = Affine(*manifest.grid_transform)
    x_range, y_range = rasterio.transform.xy(transform, [0, rows - 1], [0, cols - 1])
    xs = np.linspace(x_range[0], x_range[1], cols)
    ys = np.linspace(y_range[0], y_range[1], rows)

    coords = dict(ds_template.coords)
    if "x" in coords:
        coords["x"] = ("x", xs)
    if "y" in coords:
        coords["y"] = ("y", ys)

    data_vars: Dict[str, Dict[str, object]] = {}
    for name, da in ds_template.data_vars.items():
        if "y" not in da.dims or "x" not in da.dims:
            data_vars[name] = {
                "array": da.values.copy(),
                "fill": da.attrs.get("_FillValue", np.nan),
                "dims": da.dims,
                "attrs": da.attrs,
                "dtype": da.dtype,
                "encoding": preserved_netcdf_encoding(da.encoding),
                "y_idx": None,
                "x_idx": None,
            }
            continue
        y_idx = da.dims.index("y")
        x_idx = da.dims.index("x")
        shape = list(da.shape)
        shape[y_idx] = rows
        shape[x_idx] = cols
        fill = da.attrs.get("_FillValue", np.nan)
        data_vars[name] = {
            "array": np.full(shape, np.nan, dtype=np.float32),
            "fill": fill,
            "dims": da.dims,
            "attrs": da.attrs,
            "dtype": da.dtype,
            "encoding": preserved_netcdf_encoding(da.encoding),
            "y_idx": y_idx,
            "x_idx": x_idx,
        }

    for sub, nc_path in nc_paths:
        roi = _load_roi(sub)
        ds = xr.open_dataset(nc_path)
        for name, da in ds.data_vars.items():
            if name not in data_vars:
                continue
            info = data_vars[name]
            if info["y_idx"] is None or info["x_idx"] is None:
                continue

            arr = da.values.astype(np.float32)
            fill_val = da.attrs.get("_FillValue", np.nan)
            if not (isinstance(fill_val, float) and np.isnan(fill_val)):
                arr = np.where(arr == fill_val, np.nan, arr)

            y_idx = int(info["y_idx"])
            x_idx = int(info["x_idx"])
            sl_r, sl_c = _window_slices(sub, (arr.shape[y_idx], arr.shape[x_idx]), global_shape)

            mask = roi
            if mask.shape != (arr.shape[y_idx], arr.shape[x_idx]):
                if mask.shape == global_shape:
                    mask = mask[sl_r, sl_c]
                else:
                    mask = np.ones((arr.shape[y_idx], arr.shape[x_idx]), dtype=bool)
            mask_nd = mask
            while mask_nd.ndim < arr.ndim:
                mask_nd = np.expand_dims(mask_nd, axis=0)
            arr = np.where(np.broadcast_to(mask_nd, arr.shape), arr, np.nan)

            slice_obj = [slice(None)] * arr.ndim
            slice_obj[y_idx] = slice(sl_r.start, sl_r.stop)
            slice_obj[x_idx] = slice(sl_c.start, sl_c.stop)

            target = info["array"]
            dest = target[tuple(slice_obj)]
            replace = np.isnan(dest) & ~np.isnan(arr)
            dest[replace] = arr[replace]
            target[tuple(slice_obj)] = dest
            info["array"] = target
        ds.close()

    for name, info in data_vars.items():
        if info["y_idx"] is None or info["x_idx"] is None:
            continue
        arr = info["array"]
        valid = ~np.isnan(arr)
        while valid.ndim > 2:
            valid = valid.any(axis=0)
        if valid.shape != expected_mask.shape:
            logger.warning("Coverage check skipped for NetCDF var {} due shape mismatch {}", name, valid.shape)
            continue
        _validate_coverage_or_raise(
            target_name=f"{output_name}:{name}",
            expected_mask=expected_mask,
            data_mask=valid,
            sliver_tol_px=sliver_tol_px,
        )

    out_vars: Dict[str, xr.DataArray] = {}
    for name, info in data_vars.items():
        arr = info["array"]
        fill_val = info["fill"]
        if not (isinstance(fill_val, float) and np.isnan(fill_val)):
            arr = np.where(np.isnan(arr), fill_val, arr)
        out_vars[name] = xr.DataArray(arr.astype(info["dtype"]), dims=info["dims"], attrs=info["attrs"])

    merged = xr.Dataset(data_vars=out_vars, coords=coords, attrs=ds_template.attrs)
    encoding: dict[str, dict] = {}
    for name, info in data_vars.items():
        source_encoding = info.get("encoding")
        if isinstance(source_encoding, dict):
            encoding[name] = source_encoding
        fill_val = info["fill"]
        if not (isinstance(fill_val, float) and np.isnan(fill_val)) and "_FillValue" not in encoding.get(name, {}):
            encoding.setdefault(name, {})["_FillValue"] = fill_val

    if output_name == "da_output_grids.nc":
        encoding.update(da_summary_netcdf_encoding(merged))

    out_path = out_dir / output_name
    try:
        with _atomic_output(out_path) as tmp_path:
            merged.to_netcdf(tmp_path, encoding=encoding)
    finally:
        merged.close()
        ds_template.close()
    logger.info("Wrote merged NetCDF {}", out_path)
    return out_path
