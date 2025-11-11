"""openamundsen_da.methods.pf.rejuvenate

Create a rejuvenated prior ensemble for the next step from a posterior ensemble
without duplicating large state files.

Behavior
- Reads rejuvenation params from project.yml (data_assimilation.rejuvenation):
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
    MEMBER_SOURCE_POINTER,
    LOGURU_FORMAT,
)
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    list_member_dirs,
    meteo_dir_for_member,
    default_results_dir,
    find_step_yaml,
    find_season_yaml,
    open_loop_dir,
)


@dataclass
class RejuvenationParams:
    sigma_t: float
    sigma_p: float
    seed: Optional[int]
    rebase_open_loop: bool


def _read_rejuvenation_params(project_dir: Path) -> RejuvenationParams:
    proj = Path(project_dir) / "project.yml"
    cfg = _read_yaml_file(proj) or {}
    da = cfg.get(DA_BLOCK) or {}
    rj = da.get(REJUVENATION_BLOCK) or {}
    sigma_t = float(rj.get(REJ_SIGMA_T, 0.0))
    sigma_p = float(rj.get(REJ_SIGMA_P, 0.0))
    seed = rj.get("seed")
    if seed is None:
        pr = (da.get("prior_forcing") or {})
        seed = pr.get(DA_RANDOM_SEED)
    rebase = bool(rj.get("rebase_open_loop", False))
    return RejuvenationParams(
        sigma_t=sigma_t,
        sigma_p=sigma_p,
        seed=(int(seed) if seed is not None else None),
        rebase_open_loop=rebase,
    )


def _read_next_step_dates(next_step_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    step_yaml = find_step_yaml(next_step_dir)
    step_cfg = _read_yaml_file(step_yaml) or {}
    try:
        start = pd.to_datetime(step_cfg["start_date"])  # type: ignore[index]
    except Exception as e:
        raise ValueError(f"Missing or invalid start_date in {step_yaml}") from e
    # Prefer season end; fallback to step end_date
    try:
        seas_yaml = find_season_yaml(next_step_dir.parent)
        seas_cfg = _read_yaml_file(seas_yaml) or {}
        end = pd.to_datetime(seas_cfg["end_date"])  # type: ignore[index]
    except Exception:
        try:
            end = pd.to_datetime(step_cfg["end_date"])  # type: ignore[index]
        except Exception as e:
            raise ValueError("Could not determine end_date from season/step config") from e
    return start, end


def _inclusive_filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, *, time_col: str) -> pd.DataFrame:
    t = pd.to_datetime(df[time_col], errors="coerce")
    mask = (t >= start) & (t <= end)
    return df.loc[mask].copy()


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


def rejuvenate(
    *,
    project_dir: Path,
    prev_step_dir: Path,
    next_step_dir: Path,
    source_ensemble: str = "posterior",
    target_ensemble: str = "prior",
    rebase_open_loop: Optional[bool] = None,
    source_meteo_dir: Optional[Path] = None,
) -> dict:
    params = _read_rejuvenation_params(project_dir)
    start, end = _read_next_step_dates(next_step_dir)
    rng = np.random.default_rng(params.seed if params.seed is not None else None)
    effective_rebase = bool(rebase_open_loop) if (rebase_open_loop is not None) else bool(params.rebase_open_loop)

    src_members = list_member_dirs(Path(prev_step_dir) / "ensembles", source_ensemble)
    if not src_members:
        raise RuntimeError(f"No members under {prev_step_dir}/ensembles/{source_ensemble}")

    tgt_root = Path(next_step_dir) / "ensembles" / target_ensemble
    tgt_root.mkdir(parents=True, exist_ok=True)

    copied_pointers = 0
    rows = []

    for i, post_member in enumerate(src_members, start=1):
        member_name = f"{MEMBER_PREFIX}{i:03d}"
        src_member = _source_member_dir(post_member)

        tgt_member = tgt_root / member_name
        tgt_meteo = meteo_dir_for_member(tgt_member)
        tgt_meteo.mkdir(parents=True, exist_ok=True)
        # Do not pre-create results/ to avoid the launcher skipping members; it will be created on first write

        # Sample stationary perturbations per member
        dT = float(rng.normal(0.0, params.sigma_t)) if params.sigma_t else 0.0
        fP = float(rng.lognormal(mean=0.0, sigma=params.sigma_p)) if params.sigma_p else 1.0

        # Read stations from source meteo:
        #  - explicit source_meteo_dir if provided
        #  - else source member meteo (compound) or open_loop meteo (rebase)
        if source_meteo_dir is not None:
            src_meteo = Path(source_meteo_dir)
        else:
            base_for_meteo = open_loop_dir(prev_step_dir) if effective_rebase else src_member
            src_meteo = meteo_dir_for_member(base_for_meteo)
        stations_csv = src_meteo / "stations.csv"
        if not stations_csv.exists():
            raise FileNotFoundError(f"Missing stations.csv in {src_meteo}")

        for csv in sorted(p for p in src_meteo.glob("*.csv") if p.name != "stations.csv"):
            df = pd.read_csv(csv)
            # Accept either 'date' (default) or 'time' as time column
            time_col = DEFAULT_TIME_COL if DEFAULT_TIME_COL in df.columns else ("time" if "time" in df.columns else None)
            if time_col is None:
                raise ValueError(f"{csv.name}: missing required time column ('date' or 'time')")
            df = _inclusive_filter(df, start, end, time_col=time_col)
            if (dT != 0.0) and (DEFAULT_TEMP_COL in df.columns):
                df[DEFAULT_TEMP_COL] = pd.to_numeric(df[DEFAULT_TEMP_COL], errors="coerce") + dT
            if (fP != 1.0) and (DEFAULT_PRECIP_COL in df.columns):
                df[DEFAULT_PRECIP_COL] = pd.to_numeric(df[DEFAULT_PRECIP_COL], errors="coerce") * fP
            _write_csv(df, tgt_meteo / csv.name)

        # Copy stations.csv unchanged
        (tgt_meteo / "stations.csv").write_bytes(stations_csv.read_bytes())

        # Copy state pointer if present (support root or results location)
        post_ptr_root = post_member / STATE_POINTER_JSON
        post_ptr_results = default_results_dir(post_member) / STATE_POINTER_JSON
        post_ptr = post_ptr_root if post_ptr_root.exists() else post_ptr_results
        if post_ptr.exists():
            # Place pointer at member root in the next step
            (tgt_member / STATE_POINTER_JSON).write_text(post_ptr.read_text(encoding="utf-8"), encoding="utf-8")
            copied_pointers += 1

        rows.append({
            "member": member_name,
            "source_member": src_member.name,
            "delta_T": dT,
            "f_p": fP,
            "copied_state_pointer": post_ptr.exists(),
            "rebase_open_loop": bool(effective_rebase),
        })
        logger.info("[{m}] dT={dt:+.3f} f_p={fp:.3f} state_ptr={sp} rebase={rb}", m=member_name, dt=dT, fp=fP, sp=bool(post_ptr.exists()), rb=bool(effective_rebase))

    out_dir = Path(next_step_dir) / "assim"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_step": str(prev_step_dir),
        "target_step": str(next_step_dir),
        "source_ensemble": source_ensemble,
        "target_ensemble": target_ensemble,
        "sigma_t": params.sigma_t,
        "sigma_p": params.sigma_p,
        "seed": (int(params.seed) if params.seed is not None else None),
        "copied_state_pointers": int(copied_pointers),
        "members": rows,
    }
    (out_dir / "rejuvenate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"members": len(rows), "copied_state_pointers": copied_pointers}


def cli_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-rejuvenate", description="Rejuvenate posterior into prior for next step (no state duplication)")
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--prev-step-dir", required=True, type=Path)
    p.add_argument("--next-step-dir", required=True, type=Path)
    p.add_argument("--rebase-open-loop", action="store_true", help="Use open_loop meteo as base (apply only rejuvenation noise)")
    p.add_argument("--source-meteo-dir", type=Path, help="Explicit meteo source directory (stations.csv + per-station CSVs). Overrides rebase/compound base selection")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    try:
        summary = rejuvenate(
            project_dir=Path(args.project_dir),
            prev_step_dir=Path(args.prev_step_dir),
            next_step_dir=Path(args.next_step_dir),
            rebase_open_loop=bool(args.rebase_open_loop),
            source_meteo_dir=(Path(args.source_meteo_dir) if args.source_meteo_dir is not None else None),
        )
        logger.info("Rejuvenated prior | members={} state_ptrs={}", summary.get("members"), summary.get("copied_state_pointers"))
        return 0
    except Exception as e:
        logger.error(f"Rejuvenation failed: {e}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
