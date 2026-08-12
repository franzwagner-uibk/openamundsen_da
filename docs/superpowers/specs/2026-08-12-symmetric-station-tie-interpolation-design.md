---
published: false
---

# Symmetric station-tie interpolation design

## Purpose

Station HS and SWE values are matched to the model clock using the unique
nearest finite observation within half one model timestep. Hourly source data
can lack the exact midnight value while retaining observations at 23:00 and
01:00. Those values are equally near a 00:00 model state, so the former strict
matcher rejected them even though they bracket the target symmetrically.

The matcher will treat this one constrained case as midpoint interpolation.
It must not weaken the approved half-timestep limit or permit observations from
another day merely because they are the nearest values in the full series.

## Matching contract

The existing unique-nearest behavior remains the first choice. When exactly
two observations share the minimum offset, interpolation is accepted only
when all of these conditions hold:

- one timestamp is strictly before and one strictly after the model time;
- both offsets are equal and no larger than half the model timestep;
- the two timestamps are no more than 24 hours apart, inclusively;
- the caller's existing finite and nonnegative observation checks pass.

The effective scalar is the arithmetic mean of the two values. Because the
target is their temporal midpoint, this is linear interpolation. Its effective
matched time is the model time and its reported offset is zero. The match also
retains the two source timestamps, values and their real source offset for
internal logging and external-scheduler ranking.

Duplicate timestamps, same-side ties, more than two tied values and source
pairs wider than 24 hours remain errors. Exact model-output matching
(`require_exact=True`) never interpolates. Satellite acquisition-time matching
is unchanged.

## Consumers and outputs

The shared rule applies to station HS and SWE in assimilation, pre-run
validation and benchmark extraction. Observation sigma is unchanged because
the mean is temporal interpolation, not a combination of independent
measurements.

Each accepted interpolation is logged once per consuming operation at INFO,
including the station or source, target time, both source timestamps and
values, and the resulting mean. Existing CSV and NetCDF schemas do not gain
new fields. DA diagnostics store the model timestamp in `matched_obs_time` and
zero in `obs_time_offset_seconds` for the derived scalar.

The external North Tyrol scheduler mirrors this contract under a new scripts
policy v3. It counts an accepted interpolation as support but ranks it using
the real source offset so an exact observation remains preferable. The final
selection remains exclusively in project YAML `assimilation_events`.

## Compatibility and rollout

The rule is universal and introduces no YAML or CLI option. Existing unique
matches, exact model series, output schemas and project-wide residual-axis
behavior remain unchanged.

Core and scripts changes are reviewed separately. Core is merged and published
as an immutable main image first, without a release or tag. Scripts then adopt
that image and policy v3. Lenovo P8 receives only reviewed commits. A read-only
preflight precedes transactional regeneration of all six North Tyrol projects,
and the workflow stops before model propagation.

## Verification

Focused tests cover unique and exact matches, accepted midpoint means, the
inclusive 24-hour limit, malformed ties, unchanged sigma, INFO logging,
benchmark consistency, HS/SWE parity and scheduler source-offset ranking.
Regression coverage confirms that model-output and satellite matching remain
strict and that existing diagnostic schemas do not change.
