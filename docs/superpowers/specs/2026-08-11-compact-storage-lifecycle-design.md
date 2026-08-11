---
published: false
---

# Compact Storage Lifecycle Design

## Goal and implemented increment

Make `data_assimilation.output.retention: compact` a restart-safe lifecycle
policy for single-domain and subdomain projects. A complete 100 m Euregio ES50
subdomain project must fit on a 3.6 TB project filesystem without changing model
inputs or scientific results. `retention: full` keeps member artifacts for
reanalysis.

This change is a safe boundary-based increment toward the full storage target. It
implements exact step-window forcing, coordinator-bounded admission, rolling
restart-checkpoint cleanup, immediate successful-leaf finalization and validated
final compaction. It does **not** yet
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
- Each verified overwrite creates a new explicit ledger generation. The prior
  completed generation is marked superseded in the same durable write, remains
  available as audit history and is excluded from current consumer validation.
  A planned generation must finish or fail validation before another can start.
- Each planned batch records byte inventories for its retained consumers and
  actual producer member manifests, or a canonical completed-stage record for
  parent merge cleanup. Those dependencies are revalidated before every
  deletion, including an interrupted retry.
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
- Subdomain admission uses deterministic outer-worker-sized waves. Live
  filesystem use contains completed compact leaves, while projected growth
  reserves the active leaves, a second rolling checkpoint for the concurrent
  cohort, every queued leaf's compact products and stage-aware parent atomic
  merge/render temporaries. The aggregate is recomputed from measured artifacts
  at every boundary, so active and queued leaves cannot spend against
  unreserved shared space.
- All selected leaf projects and the parent must share one filesystem. A mixed
  filesystem manifest fails before workers start rather than applying one
  misleading free-space value to different devices.
- Before observed artifacts exist, the estimator counts the exact source rows
  and bytes selected from every forcing station file, counts every configured grid
  variable and output timestamp at 8 bytes per cell/value, restart state at
  4096 bytes per cell/member, point values at a 32-byte baseline plus margin,
  the current 40 default variables with a conservative upper bound for
  soil/snow layer columns and explicit file and
  serialization margins. Atomic overwrites reserve a complete point, forcing
  grid or map-support temporary beside the accepted file. Measurements can
  refit those bounds upward only.
  Step forcing plots reserve 4,400 bytes per station/member/day plus the same
  25% margin, calibrated from the archived Euregio run.
  Retained diagnostics, logs, member metadata and render products reserve at
  least the archived ES30 total of 8.01 GB, scaled from 31 to the configured
  member count and increased by 25%; observed bytes only raise that bound.
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
source and retained-consumer inventories, a digest of the actual producer
manifests, final consumer, regeneration recipe, planned time and completion
time. Cleanup resumes safely after interruption by reconciling the planned
entry with the contained paths and revalidating the retained consumer before
each remaining unlink.

## Lifecycle

- Before a step, estimate its forcing, grids, states and in-flight worker
  reserve. Check the filesystem containing the project directory.
- Generate only `step.start_date` through `step.end_date` forcing.
- Once step `i + 1` has complete member states, remove step `i` restart states
  under compact retention.
- Keep step forcing, point CSVs and raw grids until the project-level compact
  stores, benchmark and configured render outputs validate. Then remove them
  through the retention ledger. Keep them in full retention.
- After the compact forcing store matches the still-present raw sources and the
  stable render-completion manifest validates, remove step-local forcing PNGs as derived artifacts.
  Retain the accepted report and rerender project-wide forcing plots from the
  compact NetCDF.
- Write the final compact grid, all-member point and consumed-forcing NetCDFs
  to validated same-directory temporaries, flush them and their parent directory
  and promote them atomically. Persist
  satellite-event map-support fields before grid cleanup and compare its grid,
  ROI mask, probability domain, finite payload and values with the raw sources.
- Collapse overlapping point and forcing timestamps by numeric mean, exactly as
  the existing raw plot/benchmark readers do, and compare retained values with
  the raw sources before the cleanup ledger can delete them.
- On successful final render/report, remove final restart states and any
  remaining compact-eligible member artifacts.
- In full retention, keep and rebuild-validate raw SCF/wet-snow render sources;
  do not require the compact-only DA map-support archive. During overwrite, reserve a complete new
  checkpoint generation alongside every accepted checkpoint until promotion.
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
series and forcing-rerender equivalence, retained-value validation, leaf rerender support, resumable
subdomain behavior and full-mode preservation. Full-Euregio capacity validation
by a complete production run and incremental time-series/grid cleanup remain
follow-up work; they are not claimed by this increment. Admission remains
conservative and must refuse whenever the fixed thresholds are exceeded.

### Conservative full-Euregio envelope

This is an admission envelope, not a measured production forecast. Using the
26,254 km2 Euregio boundary (about 2.63 million 100 m cells), 366 days, ES50
plus open loop, two daily raw grid variables, 14 compact metrics, 90 leaves,
3-hourly points, the current 40 default variables upper-bounded as 64 scalar
CSV columns after soil/snow layer expansion and the deliberately worst-case
assumption that all 196 point definitions survive in every leaf gives about
14.9 TiB including the fixed 5% reserve. Approximate components are 0.89 TiB
raw member grids, 0.50 TiB retained restart baseline, 0.25 TiB leaf plus parent
compact grids, 0.08 TiB exact-window forcing plus 0.09 TiB for its atomic
compact export, 6.13 TiB point CSV allowance plus 6.75 TiB for its atomic
compact export and 0.16 TiB operational reserve. Map support is below 0.01 TiB
for this envelope. The concurrency-bound second checkpoint adds roughly
0.04 TiB for eight active leaves.
Real leaf filtering should reduce the point term substantially, but that must be
demonstrated by preflight on the prepared setup rather than assumed.

Consequently the original all-leaf increment did not satisfy the 3.6 TB
acceptance. The next implemented increment finalizes and cleans each successful
leaf immediately and admits queued leaves in deterministic outer-worker-sized
waves. Current filesystem usage contains the measured compact products from
completed leaves; projected growth contains only the active wave, rolling
checkpoints, the compact products still expected from queued leaves and the
stage-aware parent atomic merge/render reserve. A durable leaf finalization
manifest binds the retained compact analysis and parent support before raw
deletion, and failed leaves are never final-cleaned.

The prepared 90-leaf audit found 4,555 forcing station-leaf identities and only
78 explicit output points after actual leaf filtering. Immediate successful-leaf
cleanup measured about 0.9--1.0 TB peak and about 0.74 TB final. The stricter
declared envelope starts from a 0.250 TB clean production baseline and reserves
2.92 TB growth, or about 3.17 TB total, below the 3.426 TB admissible limit
(90% cap less the fixed 5% reserve) by about 0.25 TB. This bound includes the
largest eight leaves, 17,195,580 total prepared cells, a 221 GB parent atomic
merge bound and actual leaf point filtering. Retaining forcing PNGs from all 90
leaves would add about 465 GB and make the envelope refuse at about 3.64 TB.
With immediate cleanup, only the active eight-leaf wave is reserved; even the
conservative eight times the observed 107-station maximum is below 88 GB. The
retained-diagnostics allowance adds at least 16.47 GB for ES50. The recalculated
strict peak is no more than about 3.275 TB, leaving at least about 0.151 TB below
the 3.426 TB admissible limit. The strict final upper bound remains about 1.15 TB;
the measured final projection is about 0.28 TB. A verified overwrite additionally
reserves full checkpoint replacement coexistence; that overwrite-specific
reserve is not part of this clean first-run arithmetic.

This is a conservative admission result for the audited prepared setup, not a
completed full-Euregio production acceptance. The current P8's unrelated North
Tyrol tree would raise the baseline to about 0.933 TB, so it must be archived or
removed before the production run. Per-step compact fragments and cooperative
mid-member stops remain future resilience improvements rather than prerequisites
of this clean-filesystem envelope.
