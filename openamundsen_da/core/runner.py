from __future__ import annotations
import json
import os
import sys
import time
import rasterio.transform as rt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from rasterio.transform import guard_transform

from loguru import logger
from openamundsen_da.core.env import apply_numeric_thread_defaults
from openamundsen_da.core.constants import MEMBER_LOG_REL, MEMBER_MANIFEST
from openamundsen.model import OpenAmundsen

from openamundsen_da.core.config import load_merged_config
from openamundsen_da.io.paths import (
    default_results_dir,
    find_project_yaml,
    find_season_yaml,
    find_step_yaml,
    meteo_dir_for_member,
)

@dataclass
class RunResult:
    member_name: str
    status: str            # "success" | "skipped" | "failed"
    results_dir: str
    duration_seconds: float
    error: Optional[str] = None

def _patch_rasterio_transform() -> None:
    import numpy as np
    import rasterio.transform as rt
    from rasterio.transform import guard_transform

    if getattr(rt.xy, "__oa_da_patched__", False):
        return

    orig_xy = rt.xy
    orig_rowcol = rt.rowcol

    def _supports(transform) -> bool:
        transform = guard_transform(transform)
        return transform.b == 0 and transform.d == 0

    def _offset_values(offset):
        if isinstance(offset, str):
            return {
                "center": (0.5, 0.5),
                "ul": (0.0, 0.0),
                "ur": (0.0, 1.0),
                "ll": (1.0, 0.0),
                "lr": (1.0, 1.0),
            }.get(offset)
        try:
            r, c = offset
            return float(r), float(c)
        except Exception:
            return None

    def safe_xy(transform, rows, cols, zs=None, offset="center", **kwargs):
        if kwargs or not _supports(transform):
            return orig_xy(transform, rows, cols, zs=zs, offset=offset, **kwargs)
        transform = guard_transform(transform)
        offs = _offset_values(offset)
        if offs is None:
            return orig_xy(transform, rows, cols, zs=zs, offset=offset)
        dr, dc = offs
        rows_arr = np.asarray(rows, dtype=float)
        cols_arr = np.asarray(cols, dtype=float)
        rows_arr, cols_arr = np.broadcast_arrays(rows_arr, cols_arr)
        xs = transform.c + (cols_arr + dc) * transform.a + (rows_arr + dr) * transform.b
        ys = transform.f + (cols_arr + dc) * transform.d + (rows_arr + dr) * transform.e
        if xs.shape == ():
            return xs.item(), ys.item()
        return xs, ys

    def safe_rowcol(transform, xs, ys, zs=None, op=None, **kwargs):
        if kwargs or not _supports(transform):
            return orig_rowcol(transform, xs, ys, zs=zs, op=op, **kwargs)
        transform = guard_transform(transform)
        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        inv = ~transform
        cols = inv.c + xs_arr * inv.a + ys_arr * inv.b
        rows = inv.f + xs_arr * inv.d + ys_arr * inv.e
        if op is None:
            rounder = np.floor
        elif callable(op):
            rounder = op
        elif isinstance(op, str):
            mapping = {"floor": np.floor, "ceil": np.ceil, "round": np.round}
            if op not in mapping:
                return orig_rowcol(transform, xs, ys, zs=zs, op=op, **kwargs)
            rounder = mapping[op]
        else:
            return orig_rowcol(transform, xs, ys, zs=zs, op=op, **kwargs)
        rows = rounder(rows)
        cols = rounder(cols)
        if rows.shape == ():
            return int(rows.item()), int(cols.item())
        return rows.astype(int), cols.astype(int)

    safe_xy.__oa_da_patched__ = True
    safe_rowcol.__oa_da_patched__ = True
    rt.xy = safe_xy
    rt.rowcol = safe_rowcol


def _patch_linear_fit() -> None:
    import numpy as np
    from openamundsen.meteo import interpolation as oa_interp

    if getattr(oa_interp._linear_fit, "__oa_da_patched__", False):
        return

    def safe_linear_fit(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 2:
            return 0.0, 0.0
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        slope = 0.0 if denom == 0 else np.sum((x - x_mean) * (y - y_mean)) / denom
        intercept = y_mean - slope * x_mean
        if not np.isfinite(slope):
            slope = 0.0
        if not np.isfinite(intercept):
            intercept = y_mean
        return float(slope), float(intercept)

    safe_linear_fit.__oa_da_patched__ = True
    oa_interp._linear_fit = safe_linear_fit

def _write_manifest(results_dir: Path, manifest: Dict[str, Any]) -> None:
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        with (results_dir / MEMBER_MANIFEST).open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Could not write manifest in {results_dir}: {e}")

def run_member(
    project_dir: Path | str,
    season_dir: Path | str,
    step_dir: Path | str,
    member_dir: Path | str,
    *,
    results_dir: Optional[Path | str] = None,
    overwrite: bool = False,
    log_level: Optional[str] = None,
) -> RunResult:
    # keep BLAS threads to 1 per worker
    apply_numeric_thread_defaults()

    _patch_rasterio_transform()
    _patch_linear_fit()

    member_dir = Path(member_dir)
    project_dir = Path(project_dir)
    season_dir  = Path(season_dir)
    step_dir    = Path(step_dir)

    member_name = member_dir.name
    proj_yaml = find_project_yaml(project_dir)
    seas_yaml = find_season_yaml(season_dir)
    step_yaml = find_step_yaml(step_dir)
    meteo_dir = meteo_dir_for_member(member_dir)
    results_dir = Path(results_dir) if results_dir is not None else default_results_dir(member_dir)

    # Prepare logging to a per-member file to avoid interleaved console output
    # Use a dedicated logs/ folder under the member directory to not interfere
    # with results existence checks.
    log_dir = member_dir / MEMBER_LOG_REL[0]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / MEMBER_LOG_REL[1]
    old_stderr = sys.stderr
    log_handle = log_file.open("a", encoding="utf-8")
    sys.stderr = log_handle
    try:
        logger.remove()
        logger.add(sys.stderr, level=(log_level or "INFO"))
    except Exception:
        # If Loguru reconfiguration fails, continue with default stderr
        pass

    os.chdir(project_dir)

    if results_dir.exists() and not overwrite:
        logger.info(f"[{member_name}] Results already exist -> skipping (use --overwrite to rerun)")
        return RunResult(member_name, "skipped", str(results_dir), 0.0, None)

    start = time.time()
    manifest: Dict[str, Any] = {
        "member": member_name,
        "status": "starting",
        "project_yaml": str(proj_yaml),
        "season_yaml": str(seas_yaml),
        "step_yaml": str(step_yaml),
        "meteo_dir": str(meteo_dir),
        "results_dir": str(results_dir),
        "pid": os.getpid(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_manifest(results_dir, manifest)

    try:
        cfg = load_merged_config(
            proj_yaml, seas_yaml, step_yaml,
            member_meteo_dir=meteo_dir,
            results_dir=results_dir,
            log_level=log_level,
        )

        results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{member_name}] Initializing model")
        model = OpenAmundsen(cfg)
        model.initialize()
        logger.info(f"[{member_name}] Running model")
        model.run()

        dur = time.time() - start
        manifest.update({"status": "success", "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "duration_seconds": dur})
        _write_manifest(results_dir, manifest)
        return RunResult(member_name, "success", str(results_dir), dur, None)

    except Exception as e:
        dur = time.time() - start
        manifest.update({"status": "failed", "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "duration_seconds": dur, "error": repr(e)})
        _write_manifest(results_dir, manifest)
        logger.exception(f"[{member_name}] Failed with error: {e}")
        return RunResult(member_name, "failed", str(results_dir), dur, repr(e))
    finally:
        try:
            logger.remove()
        except Exception:
            pass
        sys.stderr = old_stderr
        try:
            log_handle.close()
        except Exception:
            pass
