# Compact Storage Lifecycle Design

## Goal

Make `data_assimilation.output.retention: compact` a restart-safe lifecycle
policy for single-domain and subdomain projects. A complete 100 m Euregio ES50
subdomain project must fit on a 3.6 TB project filesystem without changing model
inputs or scientific results. `retention: full` keeps member artifacts for
reanalysis.

## Invariants

- Generated forcing covers exactly the consuming step window. Values inside
  that window remain identical because the existing keyed perturbations are
  unchanged.
- Cleanup never targets shared setup inputs, observations, configuration,
  weights, resampling/rejuvenation ledgers, benchmark products, logs or final
  reports.
- A predecessor restart checkpoint is retained until every successor member
  has produced and validated its own checkpoint.
- Cleanup is consumer-gated, path-contained, idempotent and recorded atomically
  in a versioned retention ledger before paths become eligible for deletion.
- A deliberately cleaned artifact is distinguishable from a corrupt or
  unexpectedly missing artifact.
- Single-domain and subdomain leaves use the same lifecycle.
- `compact` retains every configured final grid metric, a compressed all-member
  point time-series NetCDF and the support needed to rerender configured plots
  and maps. `full` retains raw member artifacts.
- At 80% filesystem use no new step is admitted. A step may drain only when its
  conservative completion estimate stays below 90%. At 90% the active work is
  stopped at the last validated predecessor checkpoint and reported as a
  resumable low-disk interruption.

## Artifact states

1. `active`: needed by the current propagation or assimilation.
2. `checkpointed`: downstream outputs and the successor restart boundary have
   been validated.
3. `reproducible`: deterministic derivative with immutable inputs, keyed RNG
   metadata and a regeneration recipe.
4. `retained`: final science, provenance or configured rendering support.
5. `cleaned`: deliberately removed under an atomically committed ledger entry.

The retention ledger groups paths by artifact class and records count, bytes,
producer manifest digest, final consumer, regeneration recipe, planned time and
completion time. Cleanup resumes safely after interruption by reconciling the
planned entry with the contained paths.

## Lifecycle

- Before a step, estimate its forcing, grids, states and in-flight worker
  reserve. Check the filesystem containing the project directory.
- Generate only `step.start_date` through `step.end_date` forcing.
- After propagation and assimilation, write atomic compact grid and point
  fragments and the map-support fields needed by configured renderers.
- Once step `i + 1` has complete member states, remove step `i` restart states
  under compact retention.
- Once all consumers of a step's forcing and raw grids validate, remove those
  artifacts under compact retention. Keep them in full retention.
- Combine fragments atomically into final grid and point NetCDF files.
- On successful final render/report, remove final restart states and any
  remaining compact-eligible member artifacts.
- Subdomain merge keeps child compact products until the parent atomic merge
  and render validate, then applies the same child cleanup contract.

## Resume and failure behavior

Missing files remain fatal unless a matching completed cleanup-ledger entry
proves deliberate deletion and the downstream checkpoint is valid. A low-disk
pause never implies overwrite. Hash-identical inputs resume existing completed
members and steps; destructive rebuilding requires explicit `--overwrite`.

## Validation

Focused tests cover exact step windows, in-window numerical identity, ledger
atomicity and containment, consumer-gated cleanup, restart checkpoint retention,
low-disk admission, resumable subdomain behavior and full-mode preservation. A
small multi-step single/subdomain integration compares weights, compact arrays,
point series, benchmark tables and configured renders against full retention.
