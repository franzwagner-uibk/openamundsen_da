---
published: false
---

# Station Event and Subdomain Finalization Contract

Status: approved by the user on 2026-08-11.

## Objective

Make subdomain event preparation describe the observations that the runtime can
actually assimilate. Station support is evaluated at the exact assimilation
timestamp on the model clock and by the same station IDs that are enabled for
DA. A finite value elsewhere on the same calendar date is not support.

Keep the top-level project schedule authoritative while allowing individual
leaves to drop events that have no local support. Every top-level event must
still have at least one supporting leaf. Final merge, rendering and compact
cleanup consume the explicit leaf event plan instead of treating a legitimate
leaf drop as a failed global event.

## Station Event Contract

The assimilation timestamp is the configured event date combined with the
project start time, matching the timestamp used by project propagation. The
subdomain filter reads finite, nonnegative station values only at that exact
timestamp. It intersects those values with same-ID metadata rows whose
`use_for_da` role is enabled. IDs are normalized only for matching; original
strings and leading zeros are preserved in files and diagnostics.

Unavailable station events are either dropped under the existing enabled
subdomain filter or rejected with an actionable error when dropping is not
enabled. Dropped-event records include the exact assimilation timestamp and
the deterministically sorted active station IDs. The retained event and its
likelihood calculation are otherwise unchanged. Single-domain scheduling and
event rendering remain unchanged.

## FSC Reference Footprint

Snow-cover summaries retain the legacy `cloud_fraction` and
`invalid_fraction` columns and add unambiguous reference-footprint metrics.
The reference footprint is the subdomain ROI excluding pixels classified as
water. `cloud_reference_fraction` is cloud divided by that footprint.
`invalid_reference_fraction` is non-cloud nodata divided by the same footprint;
cloud and water are not counted as invalid. Counts needed to aggregate these
fractions across tiles are persisted as well.

When present, the subdomain filter uses the new reference fractions for the
existing `max_cloud_fraction`, `max_invalid_fraction` and
`min_valid_fraction` settings. Older summaries without the new columns retain
their historical behavior. This keeps existing YAML readable while ensuring
new summaries apply the same quality semantics as the external scheduler.

## Final Event Plan

The existing `event_plan_by_subdomain.csv` remains the explicit final plan.
For every top-level event and leaf it contains exactly one deterministic
`kept` or `dropped` row. Validation rejects contradictions, an unrecorded
missing leaf event, duplicate rows and top-level events with no supporting
leaf. Before final rendering, every supporting leaf must also contain the
event's weights artifact; dropped leaves are skipped.

Top-level generated maps continue to be numbered from the top-level schedule.
SCF mosaics read only supporting leaves. Dropped leaf regions remain visible
through the existing dropped-event overlay. Final cleanup still requires all
top-level maps and the report, but legitimate per-leaf drops no longer make
those products incomplete.

## Error Handling and Verification

Errors identify the project, event timestamp, variable and affected station or
leaf IDs. Focused tests cover an exact station value versus a wrong-time value
on the same day, active-role ID intersection with leading-zero IDs, FSC class
mapping and reference fractions, legacy summary fallback, mixed kept/dropped
leaf rendering and genuine missing global support. Documentation and the
Unreleased changelog describe the user-visible contracts. No YAML field,
version, release or scientific likelihood change is introduced.
