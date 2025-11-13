"""openamundsen_da.pipeline.season

End-to-end season orchestrator with strict, opinionated behavior:

- Discovers step_* under a season directory and processes them in order.
- Step 00: cold start (no restart), dumps states at the end.
- Steps >= 01: strict warm start from member-root pointer; aborts on failure.
- For each step except the last:
  - Assimilate SCF on the next step start_date.
  - Resample to posterior using project.yml resampling defaults.
  - Rejuvenate posterior -> next-step prior (writes only member-root pointers).
- At the end: generates season plots (forcing + results).

Minimal CLI; defaults handle all formats/columns/behavior without user choices.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.core.launch import launch_members
from openamundsen_da.core.prior_forcing import build_prior_ensemble
from openamundsen_da.io.paths import read_step_config
from openamundsen_da.methods.pf.assimilate_scf import assimilate_scf_for_date
from openamundsen_da.methods.pf.rejuvenate import rejuvenate
from openamundsen_da.methods.pf.resample import resample_from_weights
from openamundsen_da.methods.viz.plot_season_ensemble import plot_season_both


def _list_steps_sorted(season_dir: Path) -> List[Path]:
    items: List[Tuple[datetime, Path]] = []
    for p in sorted(Path(season_dir).glob("step_*")):
        if not p.is_dir():
            continue
        cfg = read_step_config(p) or {}
        try:
            sd = cfg.get("start_date")
            start = datetime.fromisoformat(str(sd)) if sd else None
        except Exception:
            start = None
        items.append((start or datetime.min, p))
    items.sort(key=lambda t: (t[0], t[1].name))
    return [p for _, p in items]


def _next_step_start(steps: List[Path], idx: int) -> Optional[datetime]:
    if idx + 1 >= len(steps):
        return None
    cfg = read_step_config(steps[idx + 1]) or {}
    val = cfg.get("start_date")
    try:
        return datetime.fromisoformat(str(val)) if val else None
    except Exception:
        return None


def _find_aoi(project_dir: Path) -> Path:
    env_dir = Path(project_dir) / "env"
    cands = list(env_dir.glob("*.gpkg")) + list(env_dir.glob("*.shp"))
    if not cands:
        raise FileNotFoundError(f"No AOI vector found under {env_dir}")
    return sorted(cands)[0]


@dataclass
class OrchestratorConfig:
    project_dir: Path
    season_dir: Path
    max_workers: int = 4
    overwrite: bool = False
    log_level: str = "INFO"


def run_season(cfg: OrchestratorConfig) -> None:
    logger.remove()
    logger.add(sys.stdout, level=cfg.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    steps = _list_steps_sorted(cfg.season_dir)
    if not steps:
        raise FileNotFoundError(f"No steps found under {cfg.season_dir}")
    logger.info("Discovered {} step(s)", len(steps))

    # Ensure first step has its prior ensemble (project/meteo is required)
    if steps:
        meteo_dir = cfg.project_dir / "meteo"
        if not meteo_dir.is_dir():
            raise FileNotFoundError(f"Required meteo directory not found: {meteo_dir}")
        logger.info("Initializing prior ensemble for step {} …", steps[0].name)
        build_prior_ensemble(
            input_meteo_dir=meteo_dir,
            project_dir=cfg.project_dir,
            step_dir=steps[0],
            overwrite=bool(cfg.overwrite),
        )

    aoi = _find_aoi(cfg.project_dir)
    logger.info("Using AOI: {}", aoi)

    # Process each step
    for i, step_dir in enumerate(steps):
        step_name = Path(step_dir).name
        logger.info("== Step {} ==", step_name)

        # Launch ensemble (runner enforces strict cold/warm semantics by step)
        logger.info("Launching ensemble (prior) with max_workers={} overwrite={} ...", cfg.max_workers, cfg.overwrite)
        launch_members(
            project_dir=cfg.project_dir,
            season_dir=cfg.season_dir,
            step_dir=step_dir,
            ensemble="prior",
            max_workers=int(cfg.max_workers),
            overwrite=bool(cfg.overwrite),
            results_root=None,
            log_level=cfg.log_level,
            restart_from_state=None,
            dump_state=None,
            state_pattern=None,
        )

        # If not the last step: Assimilation -> Resample -> Rejuvenate
        next_start = _next_step_start(steps, i)
        if next_start is None:
            logger.info("Final step reached; skipping assimilation/resample/rejuvenate.")
            continue

        # Quick warm-start boundary check (best effort)
        try:
            curr_cfg = read_step_config(step_dir) or {}
            end_val = curr_cfg.get("end_date")
            if end_val is not None and next_start is not None:
                from datetime import datetime
                curr_end = datetime.fromisoformat(str(end_val))
                gap = (next_start - curr_end).total_seconds()
                if gap <= 0:
                    logger.warning("Next step start ({}) is not after current step end ({}). Warm start expects start = end + one model timestep.", next_start, curr_end)
                else:
                    logger.info("Step boundary gap: {} seconds. Ensure it equals exactly one model timestep.", int(gap))
        except Exception:
            # Best-effort; do not fail if step YAMLs are incomplete or unparsable
            pass

        # Assimilation date = next step start_date
        logger.info("Assimilating SCF for date {}", next_start.strftime("%Y-%m-%d"))
        weights = assimilate_scf_for_date(
            project_dir=cfg.project_dir,
            step_dir=step_dir,
            ensemble="prior",
            date=next_start,
            aoi=aoi,
            obs_csv=None,
        )
        assim_dir = Path(step_dir) / "assim"
        assim_dir.mkdir(parents=True, exist_ok=True)
        wcsv = assim_dir / f"weights_scf_{next_start.strftime('%Y%m%d')}.csv"
        weights.to_csv(wcsv, index=False)
        logger.info("Wrote weights -> {}", wcsv)

        # Resample to posterior
        logger.info("Resampling to posterior ...")
        resample_from_weights(
            step_dir=step_dir,
            source_ensemble="prior",
            weights_csv=wcsv,
            target_ensemble="posterior",
            seed=None,
            algorithm="systematic",
            ess_threshold=0.0,
            ess_threshold_ratio=None,
            overwrite=True,
        )

        # Rejuvenate posterior -> next prior
        logger.info("Rejuvenating posterior -> {} (prior) ...", steps[i + 1].name)
        rejuvenate(
            project_dir=cfg.project_dir,
            prev_step_dir=step_dir,
            next_step_dir=steps[i + 1],
            source_ensemble="posterior",
            target_ensemble="prior",
            source_meteo_dir=None,
        )

    # Season plots (both forcing and results)
    logger.info("Generating season plots ...")
    plot_season_both(season_dir=cfg.season_dir)
    logger.info("Season processing complete: {}", cfg.season_dir)


def cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="oa-da-season", description="Process a full season: run steps, assimilate, resample, rejuvenate, plot.")
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--season-dir", required=True, type=Path)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    run_season(
        OrchestratorConfig(
            project_dir=Path(args.project_dir),
            season_dir=Path(args.season_dir),
            max_workers=int(args.max_workers or 4),
            overwrite=bool(args.overwrite),
            log_level=str(args.log_level or "INFO"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
