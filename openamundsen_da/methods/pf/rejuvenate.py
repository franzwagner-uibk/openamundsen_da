"""openamundsen_da.methods.pf.rejuvenate

Create a rejuvenated prior ensemble for the next step from a posterior ensemble
without duplicating large state files.

Behavior
- Reads rejuvenation params from project YAML (data_assimilation.rejuvenation):
  - sigma_t: additive temperature noise
  - sigma_p: multiplicative precipitation noise (lognormal with mu=0)
- For each posterior member in the previous step:
  - Determine its source member directory via MEMBER_SOURCE_POINTER
    (or fall back to the posterior member itself if missing)
  - Read station CSVs from that source meteo directory, filter to the next
    step time window, apply perturbations, and write into the next step prior
    member meteo directory
  - Copy stations.csv unchanged
  - Copy the state pointer file (STATE_POINTER_JSON) from the posterior
    member's results into the next step prior member results directory
- Writes a compact manifest JSON under next_step/assim.

This avoids copying large state files and keeps ensembles light.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import (
    DA_BLOCK,
    REJUVENATION_BLOCK,
    REJ_SIGMA_T,
    REJ_SIGMA_P,
    DA_RANDOM_SEED,
    DEFAULT_TIME_COL,
    DEFAULT_TEMP_COL,
    DEFAULT_PRECIP_COL,
    MEMBER_PREFIX,
    STATE_POINTER_JSON,
    STATE_DEFAULT_NAME,
    MEMBER_SOURCE_POINTER,
    LOGURU_FORMAT,
)
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    list_member_dirs,
    meteo_dir_for_member,
    default_results_dir,
    find_step_yaml,
    find_project_yaml,
    infer_project_dir,
    open_loop_dir,
)
from openamundsen_da.util.parallel import pick_max_workers, run_tasks_with_pool
from openamundsen_da.util.meteo import filter_and_write_meteo


@dataclass
class RejuvenationParams:
    sigma_t: float
    sigma_p: float
    seed: Optional[int]


def _read_rejuvenation_params(setup_dir: Path) -> RejuvenationParams:
    """Read rejuvenation params; reuse prior_forcing sigmas by default.

    If rejuvenation.sigma_t/p are provided, they override; otherwise we fall
    back to data_assimilation.prior_forcing.{sigma_t,sigma_p}. Seed falls back
    to prior_forcing.random_seed if not set under rejuvenation.
    """
    setup_yaml = find_project_yaml(setup_dir)
    cfg = _read_yaml_file(setup_yaml) or {}
    da = cfg.get(DA_BLOCK) or {}
    rj = da.get(REJUVENATION_BLOCK) or {}
    prior = (da.get("prior_forcing") or {})
    # Defaults: reuse prior_forcing
    sigma_t = float(rj.get(REJ_SIGMA_T, prior.get("sigma_t", 0.0)))
    sigma_p = float(rj.get(REJ_SIGMA_P, prior.get("sigma_p", 0.0)))
    seed = rj.get("seed", prior.get(DA_RANDOM_SEED))
    return RejuvenationParams(
        sigma_t=sigma_t,
        sigma_p=sigma_p,
        seed=(int(seed) if seed is not None else None),
    )


def _strip_timezone(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert tz-aware timestamps to UTC and drop tz info for safe comparisons."""
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC")
    try:
        return ts.tz_localize(None)
    except Exception:
        return ts


def _normalize_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Return a tz-naive DatetimeIndex (converted from UTC if originally tz-aware)."""
    dt_idx = pd.to_datetime(idx, errors="coerce")
    if getattr(dt_idx, "tz", None) is not None:
        dt_idx = dt_idx.tz_convert("UTC").tz_localize(None)
    return dt_idx


def _read_next_step_dates(next_step_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    step_yaml = find_step_yaml(next_step_dir)
    step_cfg = _read_yaml_file(step_yaml) or {}
    try:
        start = pd.to_datetime(step_cfg["start_date"])  # type: ignore[index]
    except Exception as e:
        raise ValueError(f"Missing or invalid start_date in {step_yaml}") from e
    # Prefer setup end; fallback to step end_date
    try:
        seas_yaml = find_project_yaml(infer_project_dir(next_step_dir))
        seas_cfg = _read_yaml_file(seas_yaml) or {}
        end = pd.to_datetime(seas_cfg["end_date"])  # type: ignore[index]
    except Exception:
        try:
            end = pd.to_datetime(step_cfg["end_date"])  # type: ignore[index]
        except Exception as e:
            raise ValueError("Could not determine end_date from setup/step config") from e
    return start, end


def _inclusive_filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start = _strip_timezone(start)
    end = _strip_timezone(end)
    dt_idx = _normalize_datetime_index(df.index)
    mask = (dt_idx >= start) & (dt_idx <= end)
    out = df.loc[mask].copy()
    out.index = dt_idx[mask]
    return out


def _write_csv(df: pd.DataFrame, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


def _source_member_dir(posterior_member: Path) -> Path:
    ptr = posterior_member / MEMBER_SOURCE_POINTER
    if ptr.exists():
        try:
            d = json.loads(ptr.read_text(encoding="utf-8")) or {}
            md = d.get("member_dir")
            if md:
                p = Path(md)
                if not p.is_absolute():
                    p = (posterior_member / p).resolve()
                return p
        except Exception:
            pass
    return posterior_member


def _rejuvenate_member_task(
    member_idx: int,
    post_member: Path,
    src_member: Path,
    tgt_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dT: float,
    fP: float,
    project_dir: Path,
    source_meteo_dir: Optional[Path],
) -> dict:
    """Worker: rebase one member's meteo and copy state pointer."""
    member_name = f"{MEMBER_PREFIX}{member_idx:03d}"
    tgt_member = tgt_root / member_name
    tgt_meteo = meteo_dir_for_member(tgt_member)
    tgt_meteo.mkdir(parents=True, exist_ok=True)

    # Choose meteo source and write perturbed window
    src_meteo = Path(source_meteo_dir) if source_meteo_dir is not None else (Path(project_dir) / "meteo")
    filter_and_write_meteo(
        src_dir=src_meteo,
        dst_dir=tgt_meteo,
        start=start,
        end=end,
        delta_t=dT,
        f_p=fP,
    )

    # Copy state pointer if present (support root or results location)
    post_ptr_root = post_member / STATE_POINTER_JSON
    post_ptr_results = default_results_dir(post_member) / STATE_POINTER_JSON
    post_ptr = post_ptr_root if post_ptr_root.exists() else post_ptr_results
    copied_ptr = False
    if post_ptr.exists():
        try:
            data = json.loads(post_ptr.read_text(encoding="utf-8")) or {}
            target = data.get("path") or data.get("state_path")
        except Exception:
            data = None
            target = None
        if target:
            q = Path(target)
            if not q.is_absolute():
                q = (post_ptr.parent / q).resolve()
            try:
                rel = q.relative_to(tgt_member)
                out = {"path": str(rel)}
            except Exception:
                out = {"path": str(q)}
            (tgt_member / STATE_POINTER_JSON).write_text(json.dumps(out, indent=2), encoding="utf-8")
            copied_ptr = True

    return {
        "member": member_name,
        "source_member": src_member.name,
        "delta_T": dT,
        "f_p": fP,
        "copied_state_pointer": copied_ptr,
        "rebase_open_loop": True,
    }


def rejuvenate(
    *,
    setup_dir: Path,
    prev_step_dir: Path,
    next_step_dir: Path,
    source_ensemble: str = "posterior",
    target_ensemble: str = "prior",
    source_meteo_dir: Optional[Path] = None,
) -> dict:
    project_dir = infer_project_dir(next_step_dir)
    params = _read_rejuvenation_params(project_dir)
    start, end = _read_next_step_dates(next_step_dir)
    rng = np.random.default_rng(params.seed if params.seed is not None else None)

    src_members = list_member_dirs(Path(prev_step_dir) / "ensembles", source_ensemble)
    if not src_members:
        raise RuntimeError(f"No members under {prev_step_dir}/ensembles/{source_ensemble}")

    tgt_root = Path(next_step_dir) / "ensembles" / target_ensemble
    tgt_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, post_member in enumerate(src_members, start=1):
        src_member = _source_member_dir(post_member)
        dT = float(rng.normal(0.0, params.sigma_t)) if params.sigma_t else 0.0
        fP = float(rng.lognormal(mean=0.0, sigma=params.sigma_p)) if params.sigma_p else 1.0
        tasks.append((i, post_member, src_member, tgt_root, start, end, dT, fP, Path(setup_dir), source_meteo_dir))

    if not tasks:
        return {"members": 0, "copied_state_pointers": 0}

    workers = pick_max_workers(None, fallback=len(tasks), limit=len(tasks))
    logger.info("Rejuvenating {} member(s) with max_workers={}", len(tasks), workers)

    rows = run_tasks_with_pool(
        _rejuvenate_member_task,
        tasks,
        max_workers=workers,
        fallback_workers=len(tasks),
        label="rejuvenate",
    )
    copied_pointers = sum(int(r.get("copied_state_pointer")) for r in rows)
    for res in rows:
        logger.info(
            "[{m}] dT={dt:+.3f} f_p={fp:.3f} state_ptr={sp} rebase=True",
            m=res["member"],
            dt=res["delta_T"],
            fp=res["f_p"],
            sp=res["copied_state_pointer"],
        )

    # Also prepare open_loop for the next step and copy state pointer
    try:
        _copy_open_loop_to_next(Path(setup_dir), Path(prev_step_dir), Path(next_step_dir), start=start, end=end)
    except Exception as e:
        logger.warning("Could not prepare open_loop for next step: {}", e)

    out_dir = Path(next_step_dir) / "assim"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: r["member"])
    manifest = {
        "source_step": str(prev_step_dir),
        "target_step": str(next_step_dir),
        "source_ensemble": source_ensemble,
        "target_ensemble": target_ensemble,
        "sigma_t": params.sigma_t,
        "sigma_p": params.sigma_p,
        "seed": (int(params.seed) if params.seed is not None else None),
        "copied_state_pointers": int(copied_pointers),
        "members": rows_sorted,
    }
    (out_dir / "rejuvenate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"members": len(rows_sorted), "copied_state_pointers": copied_pointers}


def _copy_open_loop_to_next(
    setup_dir: Path,
    prev_step_dir: Path,
    next_step_dir: Path,
    *,
    start: "pd.Timestamp",
    end: "pd.Timestamp",
) -> None:
    """Prepare next-step open_loop meteo and write a state pointer.

    - Filters previous step open_loop meteo to [start..end] and writes into
      next step prior open_loop/meteo.
    - Writes next step prior open_loop/STATE_POINTER_JSON pointing to the
      previous step open_loop state file.
    """
    prev_ol = open_loop_dir(prev_step_dir)
    next_ol = open_loop_dir(next_step_dir)
    start = _strip_timezone(start)
    end = _strip_timezone(end)

    # Base meteo comes from project-level meteo directory
    met_prev = Path(setup_dir) / "meteo"
    met_next = next_ol / "meteo"
    met_next.mkdir(parents=True, exist_ok=True)

    stations_csv = met_prev / "stations.csv"
    if stations_csv.exists():
        (met_next / "stations.csv").write_bytes(stations_csv.read_bytes())

    for src in sorted(met_prev.glob("*.csv")):
        if src.name.lower() == "stations.csv":
            continue
        df = pd.read_csv(src, parse_dates=True, index_col=0)
        time_col = df.index.name or DEFAULT_TIME_COL
        df = _inclusive_filter(df, start, end)
        df.index = _normalize_datetime_index(df.index)
        idx_col_name = df.index.name or "index"
        df_out = df.reset_index().rename(columns={idx_col_name: time_col})
        _write_csv(df_out, met_next / src.name)

    # Copy a pointer to the previous step's open_loop state file
    res_prev = prev_ol / "results"
    cand = res_prev / STATE_DEFAULT_NAME
    if not cand.exists():
        picks = sorted(res_prev.glob("*.pickle.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
        cand = picks[0] if picks else None
    if cand and cand.exists():
        next_ol.mkdir(parents=True, exist_ok=True)
        try:
            rel = cand.resolve().relative_to(next_ol)
            out = {"path": str(rel)}
        except Exception:
            out = {"path": str(cand.resolve())}
        (next_ol / STATE_POINTER_JSON).write_text(json.dumps(out, indent=2), encoding="utf-8")


def cli_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-rejuvenate", description="Rejuvenate posterior into prior for next step (rebase on open_loop; no state duplication)")
    p.add_argument("--setup-dir", required=True, type=Path)
    p.add_argument("--prev-step-dir", required=True, type=Path)
    p.add_argument("--next-step-dir", required=True, type=Path)
    p.add_argument("--source-meteo-dir", type=Path, help="Explicit meteo source directory (stations.csv + per-station CSVs). Overrides rebase/compound base selection")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    try:
        summary = rejuvenate(
            setup_dir=Path(args.setup_dir),
            prev_step_dir=Path(args.prev_step_dir),
            next_step_dir=Path(args.next_step_dir),
            source_meteo_dir=(Path(args.source_meteo_dir) if args.source_meteo_dir is not None else None),
        )
        logger.info("Rejuvenated prior | members={} state_ptrs={}", summary.get("members"), summary.get("copied_state_pointers"))
        return 0
    except Exception as e:
        logger.error(f"Rejuvenation failed: {e}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())


