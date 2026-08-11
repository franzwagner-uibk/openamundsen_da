# Compact Storage Lifecycle Design

## Goal and implemented increment

Make `data_assimilation.output.retention: compact` a restart-safe lifecycle
policy for single-domain and subdomain projects. A complete 100 m Euregio ES50
subdomain project must fit on a 3.6 TB project filesystem without changing model
inputs or scientific results. `retention: full` keeps member artifacts for
reanalysis.

This change is the first safe increment toward the full storage target. It
implements exact step-window forcing, coordinator-bounded admission, rolling
restart-checkpoint cleanup and validated final compaction. It does **not** yet
write per-step compact grid/forcing/point fragments, delete forcing or point
CSVs incrementally during propagation or terminate an openAMUNDSEN member in
the middle of propagation. Those remaining improvements require a separate
design because they change restart and worker-cancellation behavior.

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
  unexpectedly missing artifact. Planned retries recheck each current file's
  recorded size and SHA-256, and a path recreated after a completed batch is a
  new generation with a new batch.
- Single-domain and subdomain leaves use the same lifecycle.
- `compact` retains every configured final grid metric, a compressed all-member
  point time-series NetCDF and the support needed to rerender configured plots
  and maps. `full` retains raw member artifacts.
- At 80% filesystem use no new step is admitted. An already started step may
  resume only when its conservative completion estimate stays below 90%.
  Admission is checked at project/step boundaries; active openAMUNDSEN member
  processes are not terminated mid-propagation in this increment.
- Subdomain admission reserves accumulated forcing, point, raw-grid, compact
  product and one retained checkpoint growth for every unfinished leaf. A
  second rolling checkpoint is reserved for the largest leaves allowed by
  outer concurrency, together with one full parent atomic-merge temporary.
  The aggregate is recomputed from measured artifacts at every boundary, so
  active and queued leaves cannot spend against unreserved shared space.
- All selected leaf projects and the parent must share one filesystem. A mixed
  filesystem manifest fails before workers start rather than applying one
  misleading free-space value to different devices.
- Before observed artifacts exist, the estimator scales every forcing station
  file against its own time coverage, counts every configured grid
  variable and output timestamp at 8 bytes per cell/value, restart state at
  4096 bytes per cell/member, point values at 32 bytes, at least 40 default
  point columns plus configured layer multiplicity and explicit file and
  serialization margins. Atomic overwrites reserve a complete point, forcing
  or grid temporary beside the accepted file. Measurements can refit those
  bounds upward only.
  The fixed 5% filesystem reserve is additional operational headroom, not a
  substitute for model-grid, state or merge prediction.

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
- Once step `i + 1` has complete member states, remove step `i` restart states
  under compact retention.
- Keep step forcing, point CSVs and raw grids until the project-level compact
  stores, benchmark and configured render outputs validate. Then remove them
  through the retention ledger. Keep them in full retention.
- Write the final compact grid, all-member point and consumed-forcing NetCDFs
  to validated same-directory temporaries and promote them atomically. Persist
  satellite-event map-support fields before grid cleanup and compare its grid,
  ROI mask, probability domain, finite payload and values with the raw sources.
- Collapse overlapping point and forcing timestamps by numeric mean, exactly as
  the existing raw plot/benchmark readers do, and compare retained values with
  the raw sources before the cleanup ledger can delete them.
- On successful final render/report, remove final restart states and any
  remaining compact-eligible member artifacts.
- Subdomain merge keeps child compact products until the parent atomic merge
  and render validate, then applies the same child cleanup contract. Leaf
  `da_output_grids.nc` summaries remain retained because they are the source
  for leaf-level SWE and snow-depth map rerendering after raw-grid cleanup.

## Resume and failure behavior

Missing files remain fatal unless a matching completed cleanup-ledger entry
proves deliberate deletion and the downstream checkpoint is valid. A low-disk
pause never implies overwrite. Hash-identical inputs resume existing completed
members and steps; destructive rebuilding requires explicit `--overwrite`.

## Validation

Focused tests cover exact step windows, in-window numerical identity, ledger
atomicity and containment, consumer-gated cleanup, restart checkpoint retention,
coordinator-bounded forcing/grid/state/merge admission, overlapping-boundary
series equivalence, retained-value validation, leaf rerender support, resumable
subdomain behavior and full-mode preservation. Full-Euregio capacity validation
and incremental time-series/grid cleanup remain follow-up acceptance work; they
are not claimed by this increment. A conservative refusal on a 3.6 TB disk is
therefore possible until incremental grid/forcing cleanup reduces the predicted
peak; the admission check must not call such a run safe merely because 5% free
space remains.

### Conservative full-Euregio envelope

This is an admission envelope, not a measured production forecast. Using the
26,254 km2 Euregio boundary (about 2.63 million 100 m cells), 366 days, ES50
plus open loop, two daily raw grid variables, 14 compact metrics, 90 leaves,
3-hourly points, 40 default point columns and the deliberately worst-case
assumption that all 196 point definitions survive in every leaf gives about
5.7 TiB including the fixed 5% reserve. Approximate components are 0.89 TiB raw
member grids, 0.50 TiB retained restart baseline, 0.25 TiB leaf plus parent
compact grids, 0.08 TiB exact-window forcing, 3.83 TiB point CSV allowance and
0.16 TiB operational reserve, before the concurrency-bound second checkpoint.
Real leaf filtering should reduce the point term substantially, but that must be
demonstrated by preflight on the prepared setup rather than assumed.

Consequently this increment does not satisfy the 3.6 TB acceptance by itself.
After the required-grid-output completeness work is integrated, the smallest
additional lifecycle increment is to finalize and clean each successful leaf
immediately, then admit queued leaves against only active-leaf reservations,
retained compact leaf products and the parent atomic-merge reserve. Per-step
point/forcing fragments and cooperative mid-member stops remain further peak
reductions if the measured active-leaf envelope still exceeds the target.
