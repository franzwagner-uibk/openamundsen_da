"""openamundsen_da.pipeline.season

End-to-end season orchestrator with strict, opinionated behavior:

- Discovers step_* under a season directory and processes them in order.
- Step 00: cold start (no restart), dumps states at the end.
- Steps >= 01: strict warm start from member-root pointer; aborts on failure.
- For each step except the last:
  - Assimilate SCF on the next step start_date.
  - Resample to posterior using project.yml resampling defaults.
  - Rejuvenate posterior -> next-step prior (writes only member-root pointers).
- At the end: generates season plots (forcing + fraction overlay).

Minimal CLI; defaults handle all formats/columns/behavior without user choices.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.core.launch import launch_members
from openamundsen_da.core.prior_forcing import build_prior_ensemble
from openamundsen_da.io.paths import read_step_config, find_project_yaml, find_season_yaml
from openamundsen_da.util.roi import read_single_roi
from openamundsen_da.util.glacier_mask import resolve_glacier_mask
from openamundsen_da.util.da_events import load_assimilation_events, AssimilationEvent
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, start_perf_monitor
from openamundsen_da.methods.pf.assimilate_scf import (
    assimilate_scf_for_date,
    assimilate_wet_snow_for_date,
)
from openamundsen_da.methods.h_of_x.model_scf import compute_step_scf_daily_for_all_members
from openamundsen_da.methods.wet_snow.classify import classify_step_wet_snow
from openamundsen_da.methods.wet_snow.area import compute_step_wet_snow_daily_for_all_members
from openamundsen_da.methods.pf.rejuvenate import rejuvenate
from openamundsen_da.methods.pf.resample import resample_from_weights, _read_resampling_from_project
from openamundsen_da.methods.pf.plot_weights import plot_weights_for_csv
from openamundsen_da.methods.pf.plot_ess_timeline import plot_season_ess_timeline
from openamundsen_da.methods.viz.aggregate_fractions import aggregate_fraction_envelope
from openamundsen_da.observer.plot_fractions import cli_main as plot_fractions_cli
from openamundsen_da.methods.viz.plot_season_ensemble import plot_season_results
from openamundsen_da.methods.viz.plot_forcing_ensemble import cli_main as plot_forcing_cli


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


def _find_roi(project_dir: Path) -> Path:
    """Return the conventional ROI path env/roi.gpkg if present."""
    env_dir = Path(project_dir) / "env"
    roi = env_dir / "roi.gpkg"
    if roi.is_file():
        return roi
    cands = list(env_dir.glob("*.gpkg")) + list(env_dir.glob("*.shp"))
    if not cands:
        raise FileNotFoundError(f"No ROI vector found under {env_dir}")
    return sorted(cands)[0]


def _parse_datetime_opt(text: str | None) -> datetime | None:
    """Best-effort datetime parser for season/project config values."""
    if not text:
        return None
    t = str(text).strip().replace("_", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(t, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(t)
    except Exception:
        return None


def _load_wet_snow_threshold_percent(project_dir: Path) -> float:
    """Read wet-snow classification threshold (percent) from project.yml."""
    try:
        proj_yaml = find_project_yaml(project_dir)
        cfg = _read_yaml_file(proj_yaml) or {}
        da_cfg = cfg.get("data_assimilation") or {}
        wet_cfg = da_cfg.get("wet_snow") or {}
        if "classification_threshold_percent" in wet_cfg:
            return float(wet_cfg["classification_threshold_percent"])
        if "classification_threshold" in wet_cfg:
            return float(wet_cfg["classification_threshold"])
    except Exception:
        pass
    return 0.1


def _aggregate_and_copy_fraction(
    season_dir: Path,
    filename: str,
    value_col: str,
    output_name: str,
) -> tuple[Path | None, Path | None]:
    """Aggregate fraction envelopes and mirror them into plots/results."""
    env_path = aggregate_fraction_envelope(
        season_dir=season_dir,
        filename=filename,
        value_col=value_col,
        output_name=output_name,
    )
    copy_path: Path | None = None
    if env_path is not None:
        try:
            copy_path = Path(season_dir) / "plots" / "results" / Path(output_name).name
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(env_path, copy_path)
        except Exception as exc:
            logger.warning("Failed to copy {} -> {}: {}", env_path, copy_path, exc)
            copy_path = None
    return env_path, copy_path


@dataclass
class OrchestratorConfig:
    project_dir: Path
    season_dir: Path
    max_workers: int = 4
    overwrite: bool = False
    log_level: str = "INFO"
    live_plots: bool = True
    monitor_perf: bool = False
    perf_sample_interval: float = 5.0
    perf_plot_interval: float = 30.0


def _setup_logger(season_dir: Path, log_level: str) -> None:
    """Configure Loguru sinks for console and season file log."""
    logger.remove()
    logger.add(sys.stdout, level=log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)
    log_file = Path(season_dir) / f"{Path(season_dir).name}.log"
    logger.add(log_file, level=log_level.upper(), colorize=False, enqueue=True, format=LOGURU_FORMAT)


def run_season(cfg: OrchestratorConfig) -> None:
    run_start = datetime.utcnow()
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

    roi = _find_roi(cfg.project_dir)
    logger.info("Using ROI: {}", roi)
    glacier_cfg = resolve_glacier_mask(cfg.project_dir)
    glacier_path = glacier_cfg.path if glacier_cfg.enabled else None
    if glacier_path:
        logger.info("Glacier masking enabled -> {}", glacier_path)
    elif glacier_cfg.enabled:
        logger.warning("Glacier masking enabled in config but mask file not found; proceeding without masking")
    else:
        logger.info("Glacier masking disabled or no mask present; proceeding without masking")

    # Project/season metadata for DA and performance monitoring
    wet_snow_threshold = _load_wet_snow_threshold_percent(cfg.project_dir)
    logger.info("Wet-snow classification threshold set to {:.3f} % (project.yml or default)", wet_snow_threshold)

    proj_resolution = None
    proj_timestep = None
    proj_crs = None
    season_days = None
    ensemble_size = None
    try:
        proj_yaml = find_project_yaml(cfg.project_dir)
        proj_cfg = _read_yaml_file(proj_yaml) or {}
        if "resolution" in proj_cfg:
            try:
                proj_resolution = float(proj_cfg.get("resolution"))
            except Exception:
                proj_resolution = None
        if "timestep" in proj_cfg:
            proj_timestep = str(proj_cfg.get("timestep"))
        proj_crs = proj_cfg.get("crs")
        da_cfg = proj_cfg.get("data_assimilation") or {}
        pf_cfg = da_cfg.get("prior_forcing") or {}
        if "ensemble_size" in pf_cfg:
            try:
                ensemble_size = int(pf_cfg.get("ensemble_size"))
            except Exception:
                ensemble_size = None
    except Exception as exc:
        logger.warning("Perf monitor: failed to read project.yml metadata: {}", exc)

    # Season length (days) from season.yml
    try:
        seas_yaml = find_season_yaml(cfg.season_dir)
        seas_cfg = _read_yaml_file(seas_yaml) or {}
        start_val = seas_cfg.get("start_date")
        end_val = seas_cfg.get("end_date")
        start_dt = _parse_datetime_opt(str(start_val)) if start_val is not None else None
        end_dt = _parse_datetime_opt(str(end_val)) if end_val is not None else None
        if start_dt is not None and end_dt is not None:
            season_days = (end_dt.date() - start_dt.date()).days + 1
    except Exception as exc:
        logger.warning("Perf monitor: failed to read season.yml dates: {}", exc)

    # Assimilation configuration (variable/product per date)
    events = load_assimilation_events(cfg.season_dir)
    n_expected = max(0, len(steps) - 1)
    if len(events) < n_expected:
        raise ValueError(
            f"Configured {len(events)} assimilation event(s) but the season needs {n_expected}. "
            "Add events in season.yml (data_assimilation.assimilation_events) or adjust steps."
        )
    if len(events) > n_expected:
        logger.warning("More assimilation events ({}) than steps needing DA ({}); extra events will be ignored.", len(events), n_expected)

    # Approximate AOI area in km2 for performance summary
    roi_area_km2 = None
    try:
        gdf, _ = read_single_roi(Path(roi), required_field=None, to_crs=proj_crs if proj_crs is not None else None)
        roi_area_km2 = float(gdf.geometry.area.iloc[0]) / 1_000_000.0
    except Exception as exc:
        logger.warning("Perf monitor: failed to compute AOI area: {}", exc)

    perf_stop_event = None
    if cfg.monitor_perf:
        pm_cfg = PerfMonitorConfig(
            season_dir=cfg.season_dir,
            sample_interval_sec=float(cfg.perf_sample_interval or 5.0),
            plot_interval_sec=float(cfg.perf_plot_interval or 30.0),
            roi_area_km2=roi_area_km2,
            resolution_m=proj_resolution,
            timestep=proj_timestep,
            season_days=season_days,
            num_da_dates=len(events),
            num_workers=int(cfg.max_workers),
            ensemble_size=ensemble_size,
            run_start=run_start,
            tz_offset_hours=None,
        )
        perf_stop_event = start_perf_monitor(pm_cfg)

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
        # generated point_scf_roi.csv files.
        try:
            compute_step_scf_daily_for_all_members(
                project_dir=cfg.project_dir,
                step_dir=step_dir,
                aoi_path=roi,
                glacier_path=glacier_path,
                max_workers=int(cfg.max_workers),
                overwrite=bool(cfg.overwrite),
            )
        except Exception as exc:
            logger.warning("Model SCF daily computation failed for {}: {}", step_name, exc)

        # After propagation: also compute model wet-snow diagnostics (masks +
        # daily AOI fractions) for all prior members in this step so that
        # wet-snow plots are always available regardless of which observable
        # is assimilated.
        try:
            classify_step_wet_snow(
                step_dir=step_dir,
                members=None,
                threshold_percent=wet_snow_threshold,
                output_subdir="wet_snow",
                mask_prefix="wet_snow_mask",
                fraction_prefix="lwc_fraction",
                write_fraction=False,
                overwrite=bool(cfg.overwrite),
            )
            compute_step_wet_snow_daily_for_all_members(
                step_dir=step_dir,
                aoi_path=roi,
                glacier_path=glacier_path,
                max_workers=int(cfg.max_workers),
                overwrite=bool(cfg.overwrite),
                mask_subdir="wet_snow",
                mask_prefix="wet_snow_mask",
            )
        except Exception as exc:
            logger.warning("Model wet-snow diagnostics failed for {}: {}", step_name, exc)

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
                curr_end = datetime.fromisoformat(str(end_val))
                gap = (next_start - curr_end).total_seconds()
                if gap <= 0:
                    logger.warning(
                        "Next step start ({}) is not after current step end ({}). Warm start expects start = end + one model timestep.",
                        next_start,
                        curr_end,
                    )
                else:
                    logger.info("Step boundary gap: {} seconds. Ensure it equals exactly one model timestep.", int(gap))
        except Exception:
            # Best-effort; do not fail if step YAMLs are incomplete or unparsable
            pass

        # Assimilation date: map step i -> event i (skip last step)
        assim_dt = None
        ev: AssimilationEvent | None = None
        try:
            curr_cfg = read_step_config(step_dir) or {}
            start_val = curr_cfg.get("start_date")
            end_val = curr_cfg.get("end_date")
            start_dt = datetime.fromisoformat(str(start_val)) if start_val is not None else None
            end_dt = datetime.fromisoformat(str(end_val)) if end_val is not None else None
        except Exception:
            start_dt = None
            end_dt = None

        if i < len(events):
            ev = events[i]
            assim_dt = datetime.combine(ev.date, (start_dt or datetime.min).time())
            if start_dt is not None and end_dt is not None:
                if not (start_dt.date() <= ev.date <= end_dt.date()):
                    logger.warning(
                        "Configured DA date {} is outside step {} window ({} .. {})",
                        ev.date,
                        step_name,
                        start_dt.date(),
                        end_dt.date(),
                    )
        else:
            assim_dt = next_start

        if ev is None:
            logger.warning(
                "No assimilation event configured for {} -> skipping assimilation for {}",
                assim_dt.date(),
                step_name,
            )
            continue

        logger.info(
            "Assimilating {} (product {}) for date {}",
            ev.variable,
            ev.product,
            assim_dt.strftime("%Y-%m-%d"),
        )

        # Reuse existing weights if present and overwrite=False so that
        # re-running oa-da-season can skip already-assimilated steps.
        assim_dir = Path(step_dir) / "assim"
        assim_dir.mkdir(parents=True, exist_ok=True)
        if ev.variable == "wet_snow":
            weights_name = f"weights_wet_snow_{assim_dt.strftime('%Y%m%d')}.csv"
        else:
            weights_name = f"weights_scf_{assim_dt.strftime('%Y%m%d')}.csv"
        wcsv = assim_dir / weights_name
        if wcsv.is_file() and not cfg.overwrite:
            logger.info(
                "Weights CSV already exists for {}; overwrite=False -> reusing existing weights: {}",
                step_name,
                wcsv,
            )
            # Downstream resampling/rejuvenation will read this file; no need
            # to recompute or touch assimilation for this step.
        else:
            try:
                if ev.variable == "wet_snow":
                    weights = assimilate_wet_snow_for_date(
                        project_dir=cfg.project_dir,
                        step_dir=step_dir,
                        ensemble="prior",
                        date=assim_dt,
                        aoi=roi,
                        glacier_path=glacier_path,
                        obs_csv=None,
                    )
                else:
                    weights = assimilate_scf_for_date(
                        project_dir=cfg.project_dir,
                        step_dir=step_dir,
                        ensemble="prior",
                        date=assim_dt,
                        aoi=roi,
                        glacier_path=glacier_path,
                        obs_csv=None,
                        product=ev.product,
                    )
            except FileNotFoundError as exc:
                logger.error(
                    "Assimilation failed for step {} at date {}: {}. "
                    "Ensure the appropriate obs CSV exists under {}/obs for this date "
                    "or generate it via the corresponding observer CLI.",
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
            rs_cfg = _read_resampling_from_project(cfg.project_dir)
            algo = rs_cfg.algorithm or "systematic"
            ess_thr_abs = float(rs_cfg.ess_threshold or 0.0)
            ess_thr_ratio = rs_cfg.ess_threshold_ratio
            ratio_text = f"{ess_thr_ratio:.3f}" if ess_thr_ratio is not None else "None"
            logger.info(
                "Resampling to posterior ... (algorithm={} seed={} ess_thr_abs={} ess_thr_ratio={})",
                algo,
                rs_cfg.seed if rs_cfg.seed is not None else "auto",
                ess_thr_abs,
                ratio_text,
            )
            resample_from_weights(
                step_dir=step_dir,
                source_ensemble="prior",
                weights_csv=wcsv,
                target_ensemble="posterior",
                seed=rs_cfg.seed,
                algorithm=algo,
                ess_threshold=ess_thr_abs,
                ess_threshold_ratio=ess_thr_ratio,
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
                # Refresh aggregated fraction envelopes and copy them into plots/results.
                try:
                    _aggregate_and_copy_fraction(
                        season_dir=cfg.season_dir,
                filename="point_scf_roi.csv",
                value_col="scf",
                output_name="point_scf_roi_envelope.csv",
                    )
                except Exception as exc:
                    logger.warning("SCF envelope aggregation failed after step {}: {}", step_name, exc)
                try:
                    _aggregate_and_copy_fraction(
                        season_dir=cfg.season_dir,
                        filename="point_wet_snow_roi.csv",
                        value_col="wet_snow_fraction",
                        output_name="point_wet_snow_roi_envelope.csv",
                    )
                except Exception as exc:
                    logger.warning("Wet-snow envelope aggregation failed after step {}: {}", step_name, exc)
                # Per-step forcing plots (temperature in K, cumulative precip) for the current step
                try:
                    plot_forcing_cli([
                        "--step-dir", str(step_dir),
                        "--ensemble", "prior",
                        "--log-level", cfg.log_level,
                    ])
                except Exception as exc:
                    logger.warning("Forcing plot failed for {}: {}", step_name, exc)
                # Point results (SWE and snow depth), daily aggregated, in member mode for all stations
                try:
                    plot_season_results(
                        season_dir=cfg.season_dir,
                        var_col="swe",
                        mode="members",
                        resample="D",
                        resample_agg="mean",
                    )
                    plot_season_results(
                        season_dir=cfg.season_dir,
                        var_col="snow_depth",
                        mode="members",
                        resample="D",
                        resample_agg="mean",
                    )
                except Exception as exc:
                    logger.warning("Season point results plot failed after step {}: {}", step_name, exc)
                # Combined fraction overlay (SCF + wet snow), written under plots/results
                try:
                    plot_fractions_cli([
                        "--season-dir", str(cfg.season_dir),
                        "--project-dir", str(cfg.project_dir),
                        "--log-level", cfg.log_level,
                        "--mode", "band",
                    ])
                except Exception as exc:
                    logger.warning("Fraction overlay plot skipped after step {}: {}", step_name, exc)
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
            candidates = sorted(assim_dir.glob("weights_*_*.csv"))
            if not candidates:
                continue
            plot_weights_for_csv(candidates[-1])
        try:
            plot_season_ess_timeline(cfg.season_dir)
        except FileNotFoundError:
            pass
    except Exception as exc:
        logger.warning("Final assimilation plotting failed: {}", exc)

    # Season plots (point results + fraction overlay). Forcing plotted per step above.
    try:
        plot_season_results(
            season_dir=cfg.season_dir,
            var_col="swe",
            mode="members",
            resample="D",
            resample_agg="mean",
        )
        plot_season_results(
            season_dir=cfg.season_dir,
            var_col="snow_depth",
            mode="members",
            resample="D",
            resample_agg="mean",
        )
    except Exception as exc:
        logger.warning("Season point results plot failed: {}", exc)
    try:
        logger.info("Generating fraction overlay plot (SCF + wet snow) ...")
        plot_fractions_cli([
            "--season-dir", str(cfg.season_dir),
            "--project-dir", str(cfg.project_dir),
            "--mode", "band",
        ])
    except Exception as exc:
        logger.warning("Fraction overlay plot skipped: {}", exc)

    # Aggregate fraction envelopes (SCF and wet snow) for quick plotting/analysis
    try:
        _aggregate_and_copy_fraction(
            season_dir=cfg.season_dir,
            filename="point_scf_roi.csv",
            value_col="scf",
            output_name="point_scf_roi_envelope.csv",
        )
    except Exception as exc:
        logger.warning("SCF envelope aggregation failed: {}", exc)
    try:
        _aggregate_and_copy_fraction(
            season_dir=cfg.season_dir,
            filename="point_wet_snow_roi.csv",
            value_col="wet_snow_fraction",
            output_name="point_wet_snow_roi_envelope.csv",
        )
    except Exception as exc:
        logger.warning("Wet-snow envelope aggregation failed: {}", exc)

    # Generate combined SCF + wet-snow fraction plot (obs + ensemble bands + open loop)
    try:
        logger.info("Generating fraction overlay plot (SCF + wet snow) ...")
        plot_fractions_cli([
            "--season-dir", str(cfg.season_dir),
            "--project-dir", str(cfg.project_dir),
        ])
    except Exception as exc:
        logger.warning("Fraction overlay plot skipped: {}", exc)

    _setup_logger(cfg.season_dir, cfg.log_level)
    run_end = datetime.utcnow()
    duration = (run_end - run_start).total_seconds()
    logger.info("Season processing complete: {} (wall-clock {:.1f} s, ~{:.2f} h)", cfg.season_dir, duration, duration / 3600.0)

    if perf_stop_event is not None:
        perf_stop_event.set()


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
    p.add_argument(
        "--monitor-perf",
        action="store_true",
        help="Enable background performance monitor (CPU/RAM/disk) during the season run.",
    )
    p.add_argument(
        "--perf-sample-interval",
        type=float,
        default=5.0,
        help="Performance monitor sampling interval in seconds (default: 5).",
    )
    p.add_argument(
        "--perf-plot-interval",
        type=float,
        default=30.0,
        help="Performance monitor plot refresh interval in seconds (default: 30).",
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
            monitor_perf=bool(getattr(args, "monitor_perf", False)),
            perf_sample_interval=float(getattr(args, "perf_sample_interval", 5.0)),
            perf_plot_interval=float(getattr(args, "perf_plot_interval", 30.0)),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
