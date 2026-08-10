---
layout: default
title: Station Assimilation
parent: Advanced
nav_order: 1
---

# Station Assimilation
{: .no_toc }

ROI-based assimilation of station snow depth and SWE observations.
{: .fs-6 .fw-300 }

<details markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Overview

openAMUNDSEN-DA supports ROI-based assimilation of in situ station snow observations:

- `station_hs` for snow depth
- `station_swe` for snow water equivalent
- each active station contributes one likelihood term,
- stations with lower uncertainty influence the weights more strongly,
- stations with higher uncertainty influence the weights more weakly,
- all active stations on one assimilation date are combined into one ROI-level member weighting.

## Supported Variables

The following assimilation-event variables are supported:

```yaml
data_assimilation:
  assimilation_events:
    - date: 2023-01-10
      variable: station_hs
    - date: 2023-03-09
      variable: station_swe
```


## Required Inputs

### Station observation CSVs

Station observations live under:

```text
obs/stations/<station_id>.csv
```

Expected columns:

- `time`
- `snow_depth`
- `swe`

Typical layout:

```text
time,snow_depth,swe
2022-10-01 00:00:00,0.10,15.1
2022-10-01 01:00:00,0.10,14.8
...
```

Notes:

- `snow_depth` is expected in `m`
- `swe` is expected in `mm`
- a station file may exist even if only one variable is later assimilated
- missing values are allowed, but the requested variable must exist for the station/date to become active

### Station DA metadata

Station DA metadata live in:

```text
obs/stations/stations_da_metadata.csv
```

Expected columns:

- `station_id`
- `station_uncertainty_pct`
- `hs_sigma_abs_min`
- `swe_sigma_abs_min`

Optional map metadata columns:

- `id` and `name`
- `x` and `y` in the setup CRS
- `alt`
- `use_for_da` and `use_for_benchmark`

For subdomain preparation, `station_id`, `x` and `y` in this table are also a
complete station-selection contract. When the legacy
`stations_snow_depth.csv` table exists it remains authoritative; otherwise the
preparer selects same-ID observation series from these coordinates and disables
both roles for stations that lie only in the configured station buffer.

Example:

```text
station_id,station_uncertainty_pct,hs_sigma_abs_min,swe_sigma_abs_min
proviantdepot,20,0.05,15
latschbloder,80,0.20,
```

Notes:

- `station_uncertainty_pct` still falls back to the project-level default when the metadata value is empty.
- `hs_sigma_abs_min` is required for every active `station_hs` station.
- `swe_sigma_abs_min` is required for every active `station_swe` station.
- Missing required absolute sigma metadata is a hard configuration error.
- Before propagation, every station enabled for DA or benchmarking must have a
  same-ID observation CSV and openAMUNDSEN time-series output point. Matching is
  case-insensitive while the configured IDs are preserved. Stations disabled
  for both roles are exempt.
- Classified project-map station markers require `station_id`, `x`, `y`,
  `use_for_da` and `use_for_benchmark`. A benchmark-enabled station is drawn
  as a holdout and must not also be DA-active.

## Configuration

Project-level station DA settings live under:

```yaml
data_assimilation:
  station:
    default_station_uncertainty_pct: 25
    min_station_uncertainty_pct: 10
    single_station_factor: 2.0
```

Meaning of the keys:

- `default_station_uncertainty_pct`: fallback when `station_uncertainty_pct` is empty in metadata
- `min_station_uncertainty_pct`: lower bound to avoid overconfident station influence
- `single_station_factor`: extra sigma inflation when only one active station is available for a date

The absolute sigma floor is configured per station in `stations_da_metadata.csv`:

- `hs_sigma_abs_min` in `m` for `station_hs`
- `swe_sigma_abs_min` in `mm` for `station_swe`

## Effective Station Uncertainty

The station uncertainty used in DA is an **effective uncertainty**, not only sensor error.

It can absorb:

- instrument uncertainty
- local station quality and maintenance confidence
- exposure, slope, aspect, wind redistribution and vegetation around the station
- flat-field or sheltered-location effects
- broad point-to-ROI representativeness concerns

Interpretation:

- lower station uncertainty -> narrower likelihood -> stronger update
- higher station uncertainty -> wider likelihood -> weaker update

Because percentage-only scaling can become unrealistically strict near zero snow, the implementation always combines the relative term with the station-wise absolute sigma floor from metadata.

## Likelihood

For one assimilation date, each active station contributes a Gaussian log-likelihood term for every ensemble member.

Definitions:

- `y_s` = observed station value
- `H_m,s` = modeled point value at the same station for member `m`
- `sigma_s` = effective station sigma for station `s`

Residual:

```text
residual_m,s = H_m,s - y_s
```

Station contribution:

```text
log L_m,s ∝ -0.5 * (residual_m,s / sigma_s)^2
```

ROI-level member likelihood:

```text
log L_m,total = sum over all active stations s of log L_m,s
```

The normalized particle-filter weights are then computed from these summed log-likelihoods.

## Single-Station Handling

If only one active station is available on a DA date, the observation is still used, but more cautiously.

Version 1 policy:

- write a warning
- inflate the station sigma by `single_station_factor`

Effective single-station sigma:

```text
sigma_eff = sigma_base * single_station_factor
```

This keeps a single station from speaking too loudly at ROI scale while still allowing it to contribute.

## What Counts as an Active Station

A station is active on a DA date when:

- the station belongs to the ROI / setup station set,
- the required observation variable exists,
- the observation value is finite,
- the observation value is nonnegative,
- a unique nearest timestamp can be resolved,
- model point output exists for the station for all members.

Stations failing these checks are skipped and logged.

## Outputs and Diagnostics

For station DA dates, the pipeline writes:

- `weights_station_hs_YYYYMMDD.csv` or `weights_station_swe_YYYYMMDD.csv`
- `station_diagnostics_station_hs_YYYYMMDD.csv` or `station_diagnostics_station_swe_YYYYMMDD.csv`
- one diagnostic PNG per station-DA date

The diagnostics CSV contains station-level details used for the update, such as:

- `station_id`
- `member_id`
- `obs_value`
- `model_value`
- `sigma`
- residual terms
- final member weights

The fraction time-series plot also marks station DA dates with vertical lines:

- `HS` for `station_hs`
- `SWE` for `station_swe`

## Practical Interpretation

The station DA update does **not** force the ensemble mean directly to the station observation.

Instead, it reweights the members that already exist.

That means:

- if the ensemble already contains members close to the station observation, the posterior mean can move strongly toward the station
- if no ensemble member is close to the station observation, the DA can only move toward the best available members

This is standard particle-filter behavior and is especially important for strong, low-uncertainty station updates.

## Related Documentation

- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %})
- [Observation Processing]({{ site.baseurl }}{% link guides/observations.md %})
- [Running the model]({{ site.baseurl }}{% link running.md %})
- [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %})
