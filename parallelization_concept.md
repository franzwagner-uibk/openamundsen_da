# Parallel Orchestration Concept for openamundsen_da

This note lists where to parallelize remaining independent steps (per member/date) and how to avoid nested pools and I/O blowups.

## Principles
- One pool per logical stage; no nested pools.
- Clamp workers to available cores; for I/O-heavy steps, use fewer.
- Seed RNGs in parent, pass seeds into workers for reproducibility.
- Cache shared inputs (meteo DataFrames, raster slices) to avoid N× disk reads.
- Apply env (GDAL/PROJ, thread limits) before forking; import heavy libs inside workers.

## Stage-by-Stage
1) **Propagation**: keep `launch_members` as the top-level pool (already parallel).
2) **Prior forcing build** (`build_prior_ensemble`):
   - Pre-read/filter station CSVs once; pre-sample `(dT, f_p)` for all members.
   - Fan out per-member writes with a `ProcessPoolExecutor`; keep open_loop serial.
3) **Rejuvenation** (`rejuvenate`):
   - Precompute window; cache project-level meteo frames; pre-sample `(dT, f_p)`.
   - Parallel per-member perturbation + state-pointer copy; write manifest after join.
4) **Assimilation model eval** (`assimilate_scf_for_date`, wet-snow):
   - Parent: read obs, config, weight normalization.
   - Pool: per-member `compute_model_scf` / wet-snow fraction; return model values/residuals.
5) **Wet-snow classification** (`classify_step_wet_snow`):
   - Add `max_workers`; submit `_process_member` per member/date; cap workers for I/O.
6) **Daily AOI series** (SCF, wet-snow):
   - Already uses a pool; optionally preflight missing outputs to size the pool.
7) **Live plotting** (`_run_live_plots`):
   - Run in a background thread/process so the main pipeline can start the next step.

## Scheduling per step
1) Propagate prior (pool).
2) Parallel diagnostics: SCF daily + wet-snow daily + wet-snow masks.
3) Assimilation model eval (pool) → weights (serial).
4) Resample (serial, fast).
5) Rejuvenate per member (pool).
6) Trigger live plots asynchronously.

## Safety/Performance Notes
- Avoid re-reading the same CSV/raster N times; broadcast cached data to workers when feasible.
- Cap pools: CPU-bound → up to cores; I/O-bound (raster/meteo) → smaller.
- Keep logs concise in worker loops; collect summaries in parent.
- Guard overwrite semantics per member to keep reruns fast.
