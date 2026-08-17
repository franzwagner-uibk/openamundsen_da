---
published: false
---

# Durable Incremental Storage Admission Design

> Updated by the approved
> `2026-08-17-fast-storage-admission-compact-cleanup-design.md`: full estimator
> rebuilds are now limited to initial preflight and authority recovery. Normal
> phase changes consume the immutable catalog/ledger plan and producer
> accounting.

## Goal

Preserve openAMUNDSEN-DA's conservative storage guarantees without repeatedly
rescanning every active project at each data-assimilation step. Full storage
planning belongs at preflight and genuine lifecycle transitions. Ordinary step
admission must use durable accounting plus one inexpensive filesystem-space
check.

This design is separate from the compact-cleanup scan-race repair. The race
repair keeps direct and recovery-time estimators safe while artifacts are being
deleted. Incremental admission removes those scans from the runtime hot path.
It does not weaken retention evidence, checkpoint safety or scientific output
validation.

## Rejected alternatives

- Per-worker estimator caches still create divergent state and repeat
  cross-leaf scans in every process.
- Filesystem preallocation would reserve hundreds of gigabytes physically,
  depends on filesystem behavior and conflicts with compact rolling cleanup.

The selected architecture uses one durable coordinator-owned reservation
ledger per run.

## Components

The existing storage estimator remains the conservative read-only planning
engine. Its responsibilities are split internally into three roles:

1. `build_storage_plan(...)` performs the expensive preflight and produces
   immutable per-project and per-step component obligations.
2. `StorageAdmissionCoordinator` is the sole owner of mutable reservation
   state. It applies lifecycle transitions, raises observed size bounds and
   decides admission.
3. `StorageAdmissionClient` is the small internal interface used by the project
   pipeline. Single-domain runs use an in-process client. Subdomain workers use
   a spawn-safe IPC client to the parent coordinator.

The subdomain parent creates the coordinator before starting workers. A small
coordinator thread serializes requests and writes the ledger. Workers never
scan sibling leaves or edit shared accounting state. No YAML field, CLI option
or public data-assimilation API is added.

## Durable ledger

The retained audit artifact is:

```text
results/storage/storage_reservation.json
```

It records:

- schema version, generation UUID and transition sequence;
- setup, project, input and prepared-step-plan identities;
- filesystem device and capacity, overwrite generation, retention modes and
  worker/wave topology;
- immutable component obligations and estimator margins;
- current coordinator phase and per-leaf phase/last admitted step;
- monotonic observed-size high-water marks;
- unmaterialized, transition, queued-retained, parent-finalization and
  operational reserves;
- the latest filesystem snapshot and projected headroom;
- full-estimate and lightweight-check counts and durations;
- idempotent request IDs and timestamps.

The coordinator uses the existing durable atomic-manifest writer. It commits a
transition and decision before replying. A retried request is idempotent by
generation, leaf, transition sequence and request ID, but current free space is
always checked again.

## Reservation arithmetic

At admission:

```text
projected_used = filesystem_used + remaining_peak_growth + operational_reserve
```

Filesystem use already contains materialized artifacts. A future obligation is
released only after its producer reports and validates materialization. An
observed byte rate may raise, but never lower, the bound for homologous
unmaterialized work. Reservation decreases only through an authoritative
lifecycle transition: validated materialization, safe predecessor-checkpoint
cleanup, compact leaf finalization, accepted parent merge or accepted parent
render.

Preflight stores step-specific forcing, grid, point, checkpoint, compact,
diagnostic, render and atomic-coexistence obligations. Subdomain planning
evaluates every reachable wave and reserves the maximum future phase peak; it
must not assume the first wave is largest.

## Runtime lifecycle

Full estimation or reconciliation occurs only:

- before workers start;
- on resume or ledger invalidation;
- before admitting a new subdomain wave;
- when a leaf or single-domain project enters finalization;
- before parent merge;
- after accepted merge and before parent render;
- at final completion.

At a normal step boundary, a producer supplies a small accounting summary from
known local outputs: counts and bytes for forcing, point data, grids, the new
checkpoint and cleanup. The coordinator validates monotonic state, updates
high-water bounds, computes remaining peak growth from the in-memory plan,
calls `shutil.disk_usage()` once, atomically records the decision and replies.
No configuration parse, forcing-row read, glob, recursive scan or sibling-leaf
stat is allowed in this path.

Existing scientific and retention manifests remain authoritative. Storage
summaries are accounting evidence, not replacements for provenance. A missing
or malformed summary triggers one serialized, targeted leaf/phase
reconciliation. Ambiguous lifecycle identity fails closed.

The fixed 80% soft limit, 90% emergency limit and 5% operational reserve remain
unchanged. An already running model process may drain under the existing
policy; this increment does not terminate openAMUNDSEN mid-propagation.

## Recovery and invalidation

A full rebuild or reconciliation is required when configuration, inputs,
prepared steps, selected leaves, ensemble/retention/output contracts,
filesystem device, overwrite generation or concurrency differ from the ledger.
Completed phases are reconstructed from authoritative run, member,
rejuvenation, retention, leaf-finalization and subdomain-stage manifests, never
from missing paths alone.

If the coordinator dies, clients refuse the next boundary. Work already in
propagation may drain. A crash after artifact creation but before accounting is
safe because the bytes are present in filesystem use; resume performs targeted
reconciliation before admitting more work. Low disk is recorded as
`paused_low_disk` and resumes non-destructively. Overwrite always starts a new
ledger generation and preserves the superseded audit generation.

Independent jobs on the same filesystem remain visible through every live
filesystem-space check. This design does not provide atomic reservation among
separately launched openAMUNDSEN-DA commands; a filesystem-wide broker or
preallocation would be a separate feature.

## Observability

Preflight reports leaf progress, elapsed time, component reserves and projected
headroom. Each lightweight check logs generation, phase, leaf/step,
filesystem use, future and transition reserves, operational reserve, projected
headroom, estimate age and latency. Reconciliation logs its reason, duration
and upward calibration changes. The retained ledger exposes full-estimate and
lightweight-check counts and timing for regression analysis.

## Validation

Unit and contract tests cover:

- deterministic per-step planning and maximum-wave arithmetic;
- identical single-domain and subdomain admission decisions;
- a step-boundary spy proving zero accumulated materialized-tree, prior-step or
  foreign-leaf discovery; one bounded current-producer/current-member
  completion inventory is permitted before the coordinator boundary;
- concurrent leaf requests without lost updates or double releases;
- every materialization, cleanup, finalization, wave, merge and render
  transition;
- upward-only calibration and fail-closed missing summaries;
- 80/90/5% thresholds and unrelated filesystem growth;
- duplicate, stale and out-of-order requests;
- atomic ledger failure, coordinator loss and low-disk resume;
- overwrite supersession and identity/device/concurrency invalidation.

Spawn-mode integration covers a two-leaf cleanup/admission race, coordinator
restart, compact and full single-domain runs and the full subdomain
wave/finalize/merge/render lifecycle.

Performance acceptance requires zero coordinated estimator calls at step
boundaries. Coordinator work is bounded by the frozen plan and its compact
ledger serialization (and therefore scales with prepared lifecycle entries),
not by accumulated scientific output trees. P8 North Tyrol 8x6 p95 admission
must remain below one second. The same inputs must retain at least
the current conservative preflight bound, identical scientific outputs and no
larger compact peak/final footprint.

## Documentation and delivery

Update the output inventory, performance guide, compact-storage design and
`Unreleased` changelog. Existing completed projects remain readable. A coherent
partial project without the ledger receives one full reconciliation; no
scientific migration is performed silently.

Implement as a separate PR series after the compact cleanup-race repair:

1. detailed static plan and ledger/state-machine tests;
2. single-domain coordinator/client;
3. subdomain IPC and wave lifecycle;
4. producer accounting summaries;
5. integration, performance regression and P8 acceptance.
