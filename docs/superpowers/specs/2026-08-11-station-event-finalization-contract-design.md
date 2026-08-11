---
published: false
---

# Station Event and Subdomain Finalization Contract

Status: approved by the user on 2026-08-11.

## Objective

Make project validation describe the observations that the runtime can
actually assimilate. Station support is evaluated by one shared, timezone-aware
matcher and by the same station IDs that are enabled for DA. The unique nearest
observation must be within half the configured model timestep; a value elsewhere
on the same calendar date is not support.

Keep every project's `assimilation_events` authoritative. Event discovery,
quality filtering, substitution and role selection happen before and outside
openAMUNDSEN-DA. Core never chooses or drops an event. Individual leaf project
YAMLs may contain a subset of their top-level project's events, and every
top-level event must still have at least one supporting leaf. Final merge,
rendering and compact cleanup derive the explicit leaf event plan from those
project YAMLs.

## Station Event Contract

The assimilation timestamp is the configured event date combined with the
current step's start time, matching the timestamp used by project propagation.
Naive station timestamps are interpreted in the setup timezone. The matcher
accepts exactly one nearest finite, nonnegative value no farther than half a
model timestep from that timestamp. Equidistant ties and duplicate nearest
timestamps fail. It intersects those values with same-ID metadata rows whose
`use_for_da` role is enabled. IDs are normalized only for matching; original
strings and leading zeros are preserved in files and diagnostics.

Unavailable stations are skipped individually for one event and recorded in
the runtime log. Matched stations record their source timestamp and offset in
the station diagnostics. An event with no timely active station fails before model
propagation and again defensively during assimilation. The model point output
must exist at the exact model-clock timestamp. The same bounded matcher is used
for station analysis benchmarking. Existing single-station uncertainty
inflation remains unchanged.

## Final Event Plan

The generated `event_plan_by_subdomain.csv` records the resolved plan. For every
top-level event and leaf it contains exactly one deterministic `kept` or
`dropped` row. Presence in the leaf's `assimilation_events` means kept; absence
means dropped. Optional external audit rows may enrich the reason but never
override YAML. Validation rejects contradictions, duplicate rows, leaf-only
events and top-level events with no supporting leaf. Before final rendering,
every supporting leaf must also contain the event's weights artifact; omitted
leaves are skipped.

Top-level generated maps continue to be numbered from the top-level schedule.
SCF mosaics read only supporting leaves. Dropped leaf regions remain visible
through the existing dropped-event overlay. Final cleanup still requires all
top-level maps and the report, but legitimate per-leaf drops no longer make
those products incomplete.

## Error Handling and Verification

The legacy `subdomain_event_filter` key is rejected with a migration message;
the external scheduler writes final events instead. Errors identify the
project, event timestamp, variable and affected station or leaf IDs. Focused
tests cover within-window and wrong-year station values, time ties, exact model
outputs, benchmark matching, active-role IDs with leading zeros, mixed leaf
support and genuine missing global support. Documentation and the Unreleased
changelog describe the user-visible contracts. No new YAML field, version,
release or likelihood change is introduced.
