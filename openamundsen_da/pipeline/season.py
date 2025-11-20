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
from openamundsen_da.methods.h_of_x.model_scf import compute_step_scf_daily_for_all_members
from openamundsen_da.methods.pf.rejuvenate import rejuvenate
from openamundsen_da.methods.pf.resample import resample_from_weights
from openamundsen_da.methods.pf.plot_weights import plot_weights_for_csv
from openamundsen_da.methods.pf.plot_ess_timeline import plot_season_ess_timeline
from openamundsen_da.methods.viz.plot_season_ensemble import plot_season_both, plot_season_results


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
    live_plots: bool = True


def _setup_logger(season_dir: Path, log_level: str) -> None:
    """Configure Loguru sinks for console and season file log."""
    logger.remove()
    logger.add(sys.stdout, level=log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)
    log_file = Path(season_dir) / f"{Path(season_dir).name}.log"
    logger.add(log_file, level=log_level.upper(), colorize=False, enqueue=True, format=LOGURU_FORMAT)


def run_season(cfg: OrchestratorConfig) -> None:
    # Console + file log under season root (e.g. season_2017-2018/season_2017-2018.log)
    _setup_logger(cfg.season_dir, cfg.log_level)

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
            state_pattern=None,
        )

        # After propagation: compute daily model SCF for all prior members in
        # this step so that season-level plots can use var_col='scf' via the
        # generated point_scf_aoi.csv files.
        try:
            compute_step_scf_daily_for_all_members(
                project_dir=cfg.project_dir,
                season_dir=cfg.season_dir,
                step_dir=step_dir,
                aoi_path=aoi,
                max_workers=int(cfg.max_workers),
                overwrite=bool(cfg.overwrite),
            )
        except Exception as exc:
            logger.warning("Model SCF daily computation failed for {}: {}", step_name, exc)

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

        # Assimilation date: use current step end_date (aligned with season.yml
        # assimilation_dates via the season_skeleton). This ensures that:
        # - The model has produced a daily raster for this date within this
        #   step's window.
        # - We never request a raster for a date beyond the step's end.
        # Fallback to next_start if end_date is missing or unparsable.
        assim_dt = None
        try:
            curr_cfg = read_step_config(step_dir) or {}
            end_val = curr_cfg.get("end_date")
            if end_val is not None:
                assim_dt = datetime.fromisoformat(str(end_val))
        except Exception:
            assim_dt = None
        if assim_dt is None:
            assim_dt = next_start

        logger.info("Assimilating SCF for date {}", assim_dt.strftime("%Y-%m-%d"))

        # Reuse existing weights if present and overwrite=False so that
        # re-running oa-da-season can skip already-assimilated steps.
        assim_dir = Path(step_dir) / "assim"
        assim_dir.mkdir(parents=True, exist_ok=True)
        wcsv = assim_dir / f"weights_scf_{assim_dt.strftime('%Y%m%d')}.csv"
        if wcsv.is_file() and not cfg.overwrite:
            logger.info("Weights CSV already exists for {}; overwrite=False -> reusing existing weights: {}", step_name, wcsv)
            # Downstream resampling/rejuvenation will read this file; no need
            # to recompute or touch assimilation for this step.
        else:
            try:
                weights = assimilate_scf_for_date(
                    project_dir=cfg.project_dir,
                    step_dir=step_dir,
                    ensemble="prior",
                    date=assim_dt,
                    aoi=aoi,
                    obs_csv=None,
                )
            except FileNotFoundError as exc:
                logger.error(
                    "SCF assimilation failed for step {} at date {}: {}. "
                    "Ensure obs_scf_MOD10A1_YYYYMMDD.csv exists under {}/obs for this date "
                    "or generate it via oa-da-scf.",
                    step_name,
                    assim_dt.strftime("%Y-%m-%d"),
                    exc,
                    step_dir,
                )
                raise
            weights.to_csv(wcsv, index=False)
            logger.info("Wrote weights -> {}", wcsv)

        # Resample to posterior
        posterior_root = Path(step_dir) / "ensembles" / "posterior"
        has_posterior = posterior_root.is_dir() and any(posterior_root.glob("member_*"))
        if has_posterior and not cfg.overwrite:
            logger.info("Posterior ensemble already exists and overwrite=False; skipping resampling.")
        else:
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
                overwrite=bool(cfg.overwrite),
            )

        # Rejuvenate posterior -> next prior
        rejuvenate_manifest = Path(steps[i + 1]) / "assim" / "rejuvenate_manifest.json"
        if rejuvenate_manifest.is_file() and not cfg.overwrite:
            logger.info("Rejuvenation manifest already exists for {}; overwrite=False -> skipping rejuvenation.", steps[i + 1].name)
        else:
            logger.info("Rejuvenating posterior -> {} (prior) ...", steps[i + 1].name)
            rejuvenate(
                project_dir=cfg.project_dir,
                prev_step_dir=step_dir,
                next_step_dir=steps[i + 1],
                source_ensemble="posterior",
                target_ensemble="prior",
                source_meteo_dir=None,
            )

        # Update season-wide plots after each assimilation/rejuvenation cycle so
        # users can monitor progress while the pipeline continues running. Plots
        # are written with deterministic filenames and therefore overwritten on
        # each update.
        if cfg.live_plots:
            try:
                logger.info("Updating season plots after assimilation step {} ...", step_name)
                # Forcing/results season plots
                plot_season_both(season_dir=cfg.season_dir)
                # Per-step weights plot at season level
                plot_weights_for_csv(wcsv)
                # Season-wide ESS timeline across all available assimilation dates
                try:
                    plot_season_ess_timeline(cfg.season_dir)
                except FileNotFoundError:
                    # Best-effort: skip if weights are not yet available
                    pass
            except Exception as exc:
                logger.warning("Season plotting failed after step {}: {}", step_name, exc)
        # Restore orchestrator logging sinks after plot_season_* reconfigures Loguru.
        _setup_logger(cfg.season_dir, cfg.log_level)

    # Final assimilation-level plots (weights per step + ESS timeline),
    # regardless of live_plots. Best-effort: failures do not abort.
    try:
        for step_dir in steps:
            assim_dir = Path(step_dir) / "assim"
            if not assim_dir.is_dir():
                continue
            candidates = sorted(assim_dir.glob("weights_scf_*.csv"))
            if not candidates:
                continue
            plot_weights_for_csv(candidates[-1])
        try:
            plot_season_ess_timeline(cfg.season_dir)
        except FileNotFoundError:
            pass
    except Exception as exc:
        logger.warning("Final assimilation plotting failed: {}", exc)

    # Season plots (both forcing and results)
    logger.info("Generating season plots ...")
    plot_season_both(season_dir=cfg.season_dir)
    # Also generate season-wide SCF results plot (model + obs overlay) when
    # SCF point files and obs summaries are available.
    try:
        plot_season_results(season_dir=cfg.season_dir, var_col="scf")
    except FileNotFoundError as exc:
        logger.warning("SCF season plot skipped: {}", exc)
    _setup_logger(cfg.season_dir, cfg.log_level)
    logger.info("Season processing complete: {}", cfg.season_dir)


def cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="oa-da-season", description="Process a full season: run steps, assimilate, resample, rejuvenate, plot.")
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--season-dir", required=True, type=Path)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--no-live-plots",
        dest="live_plots",
        action="store_false",
        help="Skip plotting during the season; only create final plots at the end.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    run_season(
        OrchestratorConfig(
            project_dir=Path(args.project_dir),
            season_dir=Path(args.season_dir),
            max_workers=int(args.max_workers or 4),
            overwrite=bool(args.overwrite),
            log_level=str(args.log_level or "INFO"),
            live_plots=bool(getattr(args, "live_plots", True)),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
