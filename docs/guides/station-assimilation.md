---
layout: default
title: Station Assimilation
parent: Guides
nav_order: 4
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

The implementation is intentionally **ROI-based**, not pixel-wise. Station observations are used to rank whole-ROI ensemble members, not to reconstruct a distributed snow field from one point measurement.

This is the key idea:

- each active station contributes one likelihood term,
- stations with lower uncertainty influence the weights more strongly,
- stations with higher uncertainty influence the weights more weakly,
- all active stations on one assimilation date are combined into one ROI-level member weighting.

## Philosophy

The station method follows the main openAMUNDSEN-DA design:

- one ROI is the assimilation unit,
- one weight is computed per ensemble member for that ROI,
- point observations are treated as evidence for the plausibility of an ROI-scale trajectory,
- no pixel-wise localization or point-to-grid spreading is performed in Version 1.

This means the method should be read as:

> station observations help choose the most plausible watershed-scale ensemble members

not as:

> one station directly corrects the full distributed snow field

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

Station events do **not** require a product tag.

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

### Optional station DA metadata

Optional metadata live in:

```text
obs/stations/stations_da_metadata.csv
```

Expected columns in Version 1:

- `station_id`
- `station_uncertainty_pct`

Example:

```text
station_id,station_uncertainty_pct
proviantdepot,20
latschbloder,80
```

If metadata are missing for a station, the project-level default uncertainty is used and a warning is written.

## Configuration

Project-level station DA settings live under:

```yaml
data_assimilation:
  station:
    default_station_uncertainty_pct: 25
    min_station_uncertainty_pct: 10
    hs_sigma_abs_min: 0.10
    swe_sigma_abs_min: 20.0
    single_station_factor: 2.0
```

Meaning of the keys:

- `default_station_uncertainty_pct`: fallback when no station-specific metadata are given
- `min_station_uncertainty_pct`: lower bound to avoid overconfident station influence
- `hs_sigma_abs_min`: minimum allowed HS sigma in `m`
- `swe_sigma_abs_min`: minimum allowed SWE sigma in `mm`
- `single_station_factor`: extra sigma inflation when only one active station is available for a date

## Effective Station Uncertainty

The station uncertainty used in DA is an **effective uncertainty**, not only sensor error.

It can absorb:

- instrument uncertainty
- local station quality and maintenance confidence
- siting concerns
- flat-field or sheltered-location concerns
- broad point-to-ROI representativeness concerns

For one active station, the base sigma is computed as:

```text
sigma_base = max(
    sigma_abs_min,
    max(station_uncertainty_pct, min_station_uncertainty_pct) / 100 * obs_value
)
```

Interpretation:

- lower station uncertainty -> narrower likelihood -> stronger update
- higher station uncertainty -> wider likelihood -> weaker update

Because percentage-only scaling can become unrealistically strict near zero snow, the implementation always uses the configured absolute sigma floor.

## Time Matching

Station assimilation uses the **nearest available timestamp** in the station CSV for the assimilation date.

Version 1 behavior:

- nearest timestamp matching is hard-coded
- ambiguous nearest matches are rejected
- invalid or missing station rows are skipped

This keeps the implementation simple and fail-fast.

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

## Current Scope and Limits

Version 1 intentionally does **not** do the following:

- no pixel-wise localization
- no point-to-grid spatial propagation
- no explicit spatial representativeness model
- no same-date joint multivariate update across `scf`, `wet_snow`, `station_hs`, and `station_swe`
- no explicit bias-correction model for individual stations

The current method is best described as:

```text
ROI-level particle-filter weighting using station-specific effective uncertainty
```

## Related Documentation

- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %})
- [Observation Processing]({{ site.baseurl }}{% link guides/observations.md %})
- [Workflow]({{ site.baseurl }}{% link workflow.md %})
- [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %})
