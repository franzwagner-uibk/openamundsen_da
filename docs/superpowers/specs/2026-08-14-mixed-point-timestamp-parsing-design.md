---
published: false
---

# Mixed Point Timestamp Parsing Design

## Problem

Compact point export currently calls `pandas.to_datetime` without an explicit
format. With current pandas, a column whose first row is date-only can lock the
inferred format to `%Y-%m-%d`; a later model timestamp such as
`2017-10-01 03:00:00` is then rejected. The North Tyrol 2017/18 pilot exposed
this only after one leaf completed propagation and entered compact
finalization.

The source rows are valid ISO timestamps. The failure is an output parsing
defect, not an observation, model or particle-filter defect.

## Decision

Validate the supported point-output timestamp forms explicitly, then parse them
with Python's ISO parser before building the pandas index:

```python
YYYY-MM-DD[ T]HH:MM:SS[Z|+HH:MM]
```

The date-time part is optional, so date-only rows remain valid. Point outputs
are model-timestep values and the compact NetCDF coordinate stores whole
seconds, so fractional seconds fail closed rather than being rounded by the
floating-point time encoding. Other separators and ambiguous dates also fail
closed instead of being silently interpreted. This works across every
supported pandas version while preserving accepted instants exactly. Uniformly
timezone-aware values, including rows with different UTC offsets, are converted
to UTC and made timezone-naive for the compact NetCDF coordinate. A column that
mixes timezone-aware and naive values is ambiguous and fails closed. The same
mode must be used by every point source file in the project, so the compact
coordinate cannot silently combine normalized UTC instants with literal naive
instants. Malformed and unsupported timestamps also fail closed.

No fallback parser, timestamp interpolation or coercion is introduced. Point
values, overlap collapsing, compact schemas and public interfaces remain
unchanged.

## Alternatives

1. Use pandas `format="mixed"`. This is unavailable in the supported pandas
   1.5 baseline and accepts some ambiguous non-ISO strings.
2. Manually branch between date-only and date-time formats. This duplicates
   pandas parsing and creates additional timezone and fractional-second edge
   cases.
3. Append midnight to date-only source strings before parsing. This mutates
   source text unnecessarily and remains less general than the strict mixed
   ISO parser.

## Validation

- Unit-test one CSV containing both date-only and date-time rows.
- Verify exact parsed instants and values.
- Verify timezone-aware mixed rows normalize to UTC as before.
- Verify different explicit UTC offsets normalize correctly and mixed
  aware/naive rows or source files fail closed.
- Verify ambiguous non-ISO strings are rejected.
- Verify malformed timestamps remain fatal.
- Reproduce compact point export with the real step-boundary pattern.
- Run the focused point/storage/finalization suite, the complete unit suite and
  CI before publishing a new immutable image.

## Rollout

Preserve the existing failure evidence. After green review and CI, validate a
small mixed-timestamp compact export using the exact image, transactionally
refresh the P8 runtime and perform the already approved single overwrite retry.
Scientific inputs and project configuration remain unchanged.
