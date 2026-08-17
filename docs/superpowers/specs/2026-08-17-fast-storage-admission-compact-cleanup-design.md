---
published: false
---

# Fast Storage Admission and Compact Cleanup Design

## Goal

Keep the existing conservative storage thresholds and retained-science checks
while removing repeated whole-tree work from storage admission and compact
cleanup. On the audited North Tyrol eight-leaf layout, preflight must finish in
at most five minutes, ordinary step-boundary admission must have p95 latency
below one second and physical cleanup of 2,373,290 files (195.3 GB) must finish
within 60 minutes, with 39 minutes as the target.

The completed 2017/18 project is evidence, not a rerun target. No subsequent
North Tyrol project may start until the performance gates and the existing
scientific-review gate both pass.

## Storage admission

One command-scoped source catalog records contained regular files by logical
path and filesystem identity. Hard-linked or symlinked step inputs reuse one
source record, content digest and forcing summary. The catalog is immutable
after preflight and contributes directly to the storage plan and scientific
identity. A changed source invalidates the plan rather than silently refreshing
one consumer.

Full filesystem estimation is allowed only at initial preflight and recovery
from missing, stale or inconsistent authority. Normal step, wave, finalization,
merge and render admission use immutable obligations, validated producer
accounting and one live filesystem-usage call. Observed rates may only raise
future bounds. The 80% soft limit, 90% emergency limit and 5% operational
reserve are unchanged.

## Compact runtime generation

Compact retention receives one project-owned runtime generation below
`.openamundsen-da/runtime/<generation>/`. Disposable step forcing, member point
CSV files, raw grids, restart states and step forcing plots live below that
root. Durable configuration, weights, resampling and producer manifests, logs,
compact NetCDFs, reports and maps remain outside it. Full retention continues
to use the existing artifact layout.

After compact consumers and reports validate, cleanup verifies quiescence,
containment and retained-consumer hashes. It atomically renames the complete
generation to a same-filesystem quarantine, fsyncs the transition and then
deletes the quarantined tree with bounded directory-level concurrency. Cleanup
is complete only when physical deletion and the durable ledger commit both
finish. Raw disposable sources are not rehashed after compact validation.

Retention schema v6 records the generation root, closed-world tree identity,
producer authority, retained-consumer evidence and cleanup state. It does not
store millions of source-path records. Completed schema-v5 ledgers remain
readable; an interrupted v5 generation follows the legacy resume path and is
never silently migrated.

## Run-audit corrections

- A step writes forcing only for stations with at least one selected row and
  writes a matching `stations.csv`. Nonempty values are byte-equivalent after
  the established precision policy, and compact forcing retains the union of
  stations present across the project.
- Output points are assigned by the final ROI raster mask with exactly one
  deterministic owning leaf. Polygon overlap cannot duplicate a point into a
  zero-valued ROI cell.
- Plot producers close every owned Matplotlib figure on success and failure.
- Parent GISCO assets are linked into leaf environments for network-disabled
  rendering.
- Expected FSC filtering, station availability and cross-variable benchmark
  gaps are retained in structured audit files and summarized once per leaf.
  Unexpected configured support remains fatal.
- Empty/all-NA compact-forcing frames are excluded explicitly before
  concatenation. Known rasterio georeference warnings are suppressed only
  after the real FSC transform, CRS, shape and nodata equivalence is tested.
- Container launch kits expose one canonical setup mount and project path.
  Manifest recovery preserves propagation, compact-export and cleanup timing
  as distinct phases.
- Failure evidence stores compact identities, counts, hashes and representative
  failed entries instead of copying full per-file ledgers.

## Performance monitoring

Live project size comes from ledger materialization and cleanup counters.
Recursive size reconciliation is limited to preflight, recovery and terminal
completion. CPU, memory and temperature samples remain at full CSV resolution;
the PNG renderer downsamples long histories, refreshes adaptively and breaks
lines at sampling or restart gaps. A companion phase timeline distinguishes
preflight, propagation, compact export, cleanup, merge, render and unmonitored
time. Temperature reporting includes percentiles and threshold durations but
does not infer throttling without host evidence.

## Delivery and acceptance

Delivery is split into three reviewable pull requests:

1. source-catalog admission and ledger-backed monitoring;
2. compact runtime generations and tree-level cleanup;
3. forcing, point ownership, rendering, launcher and recovery-observability
   corrections.

Tests must prove that normal boundaries perform no recursive or foreign-leaf
discovery, quarantine transitions recover from crashes and low disk, compact
and full retention remain distinct, nonempty forcing and model outputs are
unchanged, output points have one owner, figures close, offline maps retain
country context, path aliases fail closed and warning summaries retain their
full audit evidence.

P8 acceptance consists of a two-step eight-leaf scientific equivalence run, a
synthetic ext4/NVMe cleanup tree matching the audited file count and byte size,
and a read-only six-project preflight. Cleanup concurrency is chosen from
measured bounded candidates rather than the 48-core count. Exceeding 60 minutes
blocks the remaining queue after cleanup finishes safely.
