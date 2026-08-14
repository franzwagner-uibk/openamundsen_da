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

Parse point-output timestamps with pandas' strict mixed-format parser:

```python
pd.to_datetime(values, format="mixed", errors="raise")
```

This accepts date-only and date-time ISO rows in one column while preserving
their exact instants. Existing timezone normalization remains unchanged:
timezone-aware values are converted to UTC and made timezone-naive for the
compact NetCDF coordinate. Malformed timestamps still fail closed.

No fallback parser, timestamp interpolation or coercion is introduced. Point
values, overlap collapsing, compact schemas and public interfaces remain
unchanged.

## Alternatives

1. Manually branch between date-only and date-time formats. This duplicates
   pandas parsing and creates additional timezone and fractional-second edge
   cases.
2. Append midnight to date-only source strings before parsing. This mutates
   source text unnecessarily and remains less general than the strict mixed
   ISO parser.

## Validation

- Unit-test one CSV containing both date-only and date-time rows.
- Verify exact parsed instants and values.
- Verify timezone-aware mixed rows normalize to UTC as before.
- Verify malformed timestamps remain fatal.
- Reproduce compact point export with the real step-boundary pattern.
- Run the focused point/storage/finalization suite, the complete unit suite and
  CI before publishing a new immutable image.

## Rollout

Preserve the existing failure evidence. After green review and CI, validate a
small mixed-timestamp compact export using the exact image, transactionally
refresh the P8 runtime and perform the already approved single overwrite retry.
Scientific inputs and project configuration remain unchanged.
