"""
openamundsen_da.core.runner

Purpose
- Execute a single ensemble member in an isolated worker process.
- Prepare the environment, build the merged openAMUNDSEN configuration,
  run initialize() and run(), and persist a small manifest and logs.

Key Behaviors
- Redirects stderr to a per-member log file so child logs don’t flood the parent console.
- Applies numeric thread defaults (1 thread per process) for BLAS/NumExpr.
- Writes a manifest JSON with status, timing, and errors for post-mortem.
"""

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
from openamundsen_da.core.env import apply_numeric_thread_defaults, _read_yaml_file
from openamundsen_da.core.constants import (
    MEMBER_LOG_REL,
    MEMBER_MANIFEST,
    LOGURU_FORMAT,
    DA_BLOCK,
    RESTART_BLOCK,
    RESTART_USE_STATE,
    RESTART_DUMP_STATE,
    RESTART_STATE_PATTERN,
    STATE_DEFAULT_NAME,
    STATE_POINTER_JSON,
)
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
    """Guard rasterio.transform helpers against non-affine transforms.

    Keeps behavior identical to rasterio for unsupported cases but avoids
    errors for simple, common affine cases used by OA.
    """
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
    """Replace a fragile linear fit with a robust, dependency-free variant."""
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
    """Best-effort write of the per-member run manifest JSON."""
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
    restart_from_state: Optional[bool] = None,
    dump_state: Optional[bool] = None,
    state_pattern: Optional[str] = None,
) -> RunResult:
    # Step 1: Constrain numeric library threads (one per worker)
    apply_numeric_thread_defaults()

    # Step 2: Apply light runtime patches
    _patch_rasterio_transform()
    _patch_linear_fit()

    # Step 3: Normalize inputs to Path objects
    member_dir = Path(member_dir)
    project_dir = Path(project_dir)
    season_dir  = Path(season_dir)
    step_dir    = Path(step_dir)

    member_name = member_dir.name
    # Step 4: Resolve YAMLs and member directories
    proj_yaml = find_project_yaml(project_dir)
    seas_yaml = find_season_yaml(season_dir)
    step_yaml = find_step_yaml(step_dir)
    meteo_dir = meteo_dir_for_member(member_dir)
    results_dir = Path(results_dir) if results_dir is not None else default_results_dir(member_dir)

    # Step 5: Redirect stderr/logging to a per-member log file
    # Use a dedicated logs/ folder under the member directory to not
    # interfere with results existence checks.
    log_dir = member_dir / MEMBER_LOG_REL[0]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / MEMBER_LOG_REL[1]
    old_stderr = sys.stderr
    log_handle = log_file.open("a", encoding="utf-8")
    sys.stderr = log_handle
    try:
        logger.remove()
        logger.add(sys.stderr, level=(log_level or "INFO"), colorize=True, format=LOGURU_FORMAT)
    except Exception:
        # If Loguru reconfiguration fails, continue with default stderr
        pass

    # Step 6: Change to project root (OA expects relative paths from here)
    os.chdir(project_dir)

    # Step 7: Skip if results exist and overwrite is not requested
    if results_dir.exists() and not overwrite:
        logger.info(f"[{member_name}] Results already exist -> skipping (use --overwrite to rerun)")
        return RunResult(member_name, "skipped", str(results_dir), 0.0, None)

    # Step 8: Initialize manifest and timing
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

    # Step 9: Build merged OA config and run the model
    try:
        # Resolve restart behavior from project.yml (overridable via args)
        cfg_yaml = _read_yaml_file(proj_yaml)
        da_cfg = (cfg_yaml.get(DA_BLOCK) or {}) if isinstance(cfg_yaml, dict) else {}
        rs_cfg = (da_cfg.get(RESTART_BLOCK) or {}) if isinstance(da_cfg, dict) else {}
        do_restart = (restart_from_state if restart_from_state is not None else bool(rs_cfg.get(RESTART_USE_STATE, False)))
        do_dump = (dump_state if dump_state is not None else bool(rs_cfg.get(RESTART_DUMP_STATE, False)))
        patt = (state_pattern if state_pattern is not None else (rs_cfg.get(RESTART_STATE_PATTERN) or STATE_DEFAULT_NAME))

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
        # Warm start (optional): copy state from saved file into model
        if do_restart:
            try:
                state_file = _resolve_state_file(results_dir, patt)
                if state_file is None:
                    logger.warning(f"[{member_name}] Warm start enabled but no state file found (pattern='{patt}')")
                else:
                    _copy_state_vars_from_init_file(state_file, model)
                    logger.info(f"[{member_name}] Loaded state from {state_file.name}")
            except Exception as e:
                logger.warning(f"[{member_name}] Warm start failed ({e}); continuing from cold init")
        logger.info(f"[{member_name}] Running model")
        model.run()
        # Dump final state (optional)
        if do_dump:
            try:
                out_name = _state_output_name(patt)
                _dump_init_data(model, results_dir / out_name)
                logger.info(f"[{member_name}] Saved state to {out_name}")
            except Exception as e:
                logger.warning(f"[{member_name}] Could not save state ({e})")

        dur = time.time() - start
        manifest.update({
            "status": "success",
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": dur,
        })
        _write_manifest(results_dir, manifest)
        return RunResult(member_name, "success", str(results_dir), dur, None)

    except Exception as e:
        dur = time.time() - start
        manifest.update({
            "status": "failed",
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": dur,
            "error": repr(e),
        })
        _write_manifest(results_dir, manifest)
        logger.exception(f"[{member_name}] Failed with error: {e}")
        return RunResult(member_name, "failed", str(results_dir), dur, repr(e))
    finally:
        # Step 10: Restore stderr and close log file
        try:
            logger.remove()
        except Exception:
            pass
        sys.stderr = old_stderr
        try:
            log_handle.close()
        except Exception:
            pass


# ---- Warm start helpers -----------------------------------------------------

def _state_output_name(pattern: str) -> str:
    # Use explicit filename if no wildcards; else fall back to default name
    return STATE_DEFAULT_NAME if any(ch in str(pattern) for ch in "*?[]") else str(pattern)


def _resolve_state_file(results_dir: Path, pattern: str) -> Path | None:
    # 1) Direct match or glob within results_dir
    p = results_dir / pattern
    if p.exists() and p.is_file():
        return p
    matches = list(results_dir.glob(pattern))
    if matches:
        matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return matches[0]
    # 2) Pointer-based resolution: read STATE_POINTER_JSON for external path
    try:
        ptr = results_dir / STATE_POINTER_JSON
        if ptr.exists():
            import json
            d = json.loads(ptr.read_text(encoding="utf-8")) or {}
            target = d.get("path") or d.get("state_path")
            if target:
                q = Path(target)
                if not q.is_absolute():
                    # Resolve relative to results_dir
                    q = (results_dir / q).resolve()
                if q.exists() and q.is_file():
                    return q
    except Exception:
        pass
    return None


def _copy_state_vars_from_init_file(filename: Path, dst_model) -> None:
    import gzip
    import pickle

    with gzip.open(filename, "rb") as f:
        d = pickle.load(f)
    for category in dst_model.state.categories:
        for var_name in dst_model.state[category]._meta.keys():
            try:
                var_data = d[category][var_name]
            except KeyError:
                continue
            dst_model.state[category][var_name][:] = var_data


def _dump_init_data(model, filename: Path) -> None:
    import gzip
    import pickle

    init_data = {}
    for category in model.state.categories:
        init_data[category] = {}
        for var_name in model.state[category]._meta.keys():
            var_data = model.state[category][var_name]
            init_data[category][var_name] = var_data
    with gzip.open(filename, "wb") as f:
        pickle.dump(init_data, f)
