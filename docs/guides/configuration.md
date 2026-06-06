---
layout: default
title: Configuration Reference
parent: Guides
nav_order: 2
---

# Configuration Reference
{: .no_toc }

Configuration model for openAMUNDSEN-DA.
{: .fs-6 .fw-300 }

## Hierarchy
1. `<setup-name>.yml` (fallback `setup.yml`) - setup-wide openAMUNDSEN config.
2. `<project-name>.yml` (fallback `project.yml`) - project-specific data assimilation config and time span.
3. `step_XX.yml` - auto-generated step windows.

## `<setup-name>.yml` / `setup.yml` (setup level)
Use setup YAML for stable, shared settings that apply to all projects.

Typical keys:
- `domain`, `resolution`, `timestep`, `crs`, `timezone`
- openAMUNDSEN model and output settings
- environment block (`GDAL_DATA`, `PROJ_LIB`, ...)

Example:

```yaml
domain: rofental
resolution: 100
timestep: 3H
crs: epsg:32632
timezone: 1

output_data:
  grids:
    format: netcdf
    variables:
      - var: snow.swe
        name: swe
        freq: D
      - var: snow.depth
        name: hs
        freq: D
      - var: snow.lwc
        name: lwc
        freq: D
```

Do not place `obs` or `data_assimilation` in setup YAML.

## `<project-name>.yml` / `project.yml` (project level)
Use project YAML for data assimilation configuration for one project.

Required top-level keys:
- `start_date`
- `end_date`
- `obs`
- `data_assimilation`

Example:

```yaml
start_date: 2022-10-01
end_date: 2023-09-30

obs:
  stations:
    dir: obs/stations
  snowcover:
    dir: obs/snowcover
    product_tag: SNOWCOVER
    summary_csv: obs/summaries/project_2022_2023/scf_summary.csv
    classes:
      # Example only: use the class set required by your product
      valid: [0, 1, 2, 3, 4, 5]
      cloud: [205]
      water: [210]
      nodata: [255]
  wetsnow:
    dir: obs/wetsnow
    product_tag: WETSNOW
    summary_csv: obs/summaries/project_2022_2023/wet_snow_summary.csv
    classes:
      wet: [110]
      valid: [110, 125, 200, 210]
      exclude: [200, 210]

data_assimilation:
  prior_forcing:
    ensemble_size: 20
    random_seed: 42
    sigma_t: 0.5
    mu_p: 0.0
    sigma_p: 0.5
    humidity_perturbation_method: dew_point
    sigma_rh: 0.0
    sigma_sw: 0.0

  h_of_x:
    method: depth_threshold
    variable: hs
    params:
      h0: 0.01
      k: 80

  station:
    default_station_uncertainty_pct: 25
    min_station_uncertainty_pct: 10
    single_station_factor: 2.0

  wet_snow:
    classification_method: liquid_water_fraction # or liquid_water_amount
    classification_threshold_percent: 0.4 # used by liquid_water_fraction
    liquid_water_amount_threshold_mm: 5.0 # used by liquid_water_amount

  likelihood:
    scf:
      obs_sigma: 0.10
      use_binomial: true
      sigma_floor: 0.05
      sigma_cloud_scale: 0.10
      min_sigma: 0.03
    wet_snow:
      obs_sigma: 0.15
      use_binomial: false
    wet_snow_line:
      obs_sigma: 150.0
      use_binomial: false
      sigma_floor: 25.0
      min_sigma: 25.0
      min_support_coverage_ratio: 0.10
      min_model_finite_fraction: 1.0 # set 0.90 for WSLA sensitivity experiments

  resampling:
    algorithm: systematic
    ess_threshold_ratio: 0.5

  rejuvenation:
    sigma_t: 0.2
    sigma_p: 0.2
    humidity_perturbation_method: dew_point
    sigma_rh: 0.0
    sigma_sw: 0.0

  restart:
    dump_state: true
    state_pattern: model_state.pickle.gz
    cleanup_after_setup: true

  benchmark:
    independent_variables:
      - station_swe
    score_station_sigma_threshold: 200
    plots: true
    output_dir: results/benchmark

  output:
    retention: compact # options: compact | full; sub-domain projects default to full when omitted

  landcover_mask:
    enabled: true
    classes_to_exclude: [2, 8, 9, 10, 11, 12, 13]

  uncertainty:
    scf:
      enabled: false # enable uncertainty-aware SCF preprocessing + assimilation
      ingest:
        # Required when uncertainty is enabled:
        scf_variable: fsc
        uncertainty_variable: uncertainty
        time_variable: time
      assimilation:
        sigma_mode: formula # formula | uncertainty_layer
        aggregate_metric: unc_mean # used when sigma_mode=uncertainty_layer
      input_dir: obs/snowcover # used by oa-da-scf-uncertainty generator
      u_min: 10.0
      u_max: 20.0
      nodata_value: 255.0
      penalties:
        - name: forest
          source: landcover # one of: fsc | landcover | shadow
          enabled: true
          classes: [8, 9, 10, 11, 12]
          penalty: 20.0
        - name: shadow
          source: shadow
          enabled: false
          input_dir: obs/shadow
          classes: [1]
          penalty: 20.0
    wet_snow:
      enabled: false # enable uncertainty-aware wet-snow preprocessing + assimilation
      ingest:
        # Required when uncertainty is enabled:
        wet_snow_variable: wet_snow
        uncertainty_variable: uncertainty
        time_variable: time
      assimilation:
        sigma_mode: formula # formula | uncertainty_layer
        aggregate_metric: unc_mean # used when sigma_mode=uncertainty_layer
      input_dir: obs/wetsnow # used by oa-da-wetsnow-uncertainty generator
      base_uncertainty: 15.0
      nodata_value: 255.0
      penalties:
        - name: forest
          source: landcover # one of: wet_snow | landcover | shadow
          enabled: true
          classes: [8, 9, 10, 11, 12]
          penalty: 20.0
        - name: shadow
          source: shadow
          enabled: false
          input_dir: obs/shadow
          classes: [1]
          penalty: 20.0

  assimilation_events:
    - date: 2023-03-17
      variable: scf
      product: SNOWCOVER
    - date: 2023-03-24
      variable: station_hs
    - date: 2023-04-03
      variable: wet_snow
      product: S1
    - date: 2023-04-21
      variable: station_swe
```

Notes:
- `assimilation_events` defines which dates and variables are assimilated.
- Station observation assimilation uses `variable: station_hs` or `variable: station_swe` and does not require a product tag.
- Station observations live in `obs/stations/<station_id>.csv`; station DA metadata live in `obs/stations/stations_da_metadata.csv`.
- `data_assimilation.station` defines project-level percentage defaults and single-station inflation for ROI-based station assimilation.
- Station absolute sigma floors are configured per station in `stations_da_metadata.csv` via `hs_sigma_abs_min` and `swe_sigma_abs_min`.
- See [Station Assimilation]({{ site.baseurl }}{% link guides/station-assimilation.md %}) for the method logic, effective sigma definition, single-station handling, and diagnostics.
- Observation class mappings and product tags are configured under project YAML `obs.*`.
- `data_assimilation.benchmark` does not enable or disable benchmarking; the project pipeline always runs it. This block extends the benchmark scope and controls benchmark output location and plot writing. The benchmark presentation itself is fixed and lean: one assimilation-date skill plot plus two compact summary tables.
- `independent_variables` may currently list only the DA-supported families: `scf`, WSF (`wet_snow`), WSLA (`wet_snow_line`), `station_hs`, `station_swe`.
- `score_station_sigma_threshold` optionally excludes high-uncertainty station rows from non-sigma-aware benchmark metrics (`CRPSS`, `NER`) while leaving sigma-aware `zSkill` unchanged. The threshold is compared against the resolved station uncertainty percent from `obs/stations/stations_da_metadata.csv`.
- `prior_forcing.humidity_perturbation_method` controls humidity perturbations. Prior forcing and rejuvenation must use the same humidity perturbation method. The default `dew_point` converts `temp` and `rel_hum` to dew-point temperature, applies `sigma_rh` as a K-scale additive dew-point perturbation, caps dew point at perturbed air temperature and converts back to `rel_hum`. The legacy `relative_humidity` method applies `sigma_rh` as raw percentage-point RH noise and clips to `[0, 100]`.
- `prior_forcing.sigma_sw` adds a multiplicative `sw_in` perturbation using a positive factor; it is applied only for positive daytime shortwave values, so nighttime `sw_in` remains unchanged.
- If `humidity_perturbation_method` is omitted, it defaults to `dew_point`. If `sigma_rh` or `sigma_sw` are omitted, they default to `0.0` and the corresponding perturbation is disabled.
- Output stream labels are derived by benchmark semantics, not by config naming alone: a configured extra family can still appear as `semi_independent` in outputs, but only from the first same-variable or sister-station assimilation date onward.
- Land-cover mask uses `grids/lc_<domain>_<resolution>.asc` from setup-level paths and data assimilation mask classes from project YAML.
- For SCF uncertainty:
  - `enabled: true` activates strict uncertainty checks (fail-fast on missing/invalid config or layers).
  - `sigma_mode: uncertainty_layer` uses `aggregate_metric / 100` (for example `unc_mean`) with `min_sigma` floor.
  - NetCDF uses configured in-file variables; GeoTIFF requires `<stem>_uncertainty.tif`.
  - Cloud pixels should be handled as data gaps (masked), not as uncertainty-penalty pixels.
- Wet-snow uncertainty uses the same pattern (`ingest` + `assimilation`) and the same file-type behavior.
- Uncertainty generator keys:
  - `input_dir`, `u_min`, `u_max`, `base_uncertainty`, `nodata_value`, and `penalties[]` are used by `oa-da-scf-uncertainty` / `oa-da-wetsnow-uncertainty`.
  - `penalties[].input_dir` is required only for `source: shadow`.
- `output.retention: compact` writes `results/grids/da_output_grids.nc` and removes heavy member grid artifacts.
- `run_mode: subdomain` defaults to `output.retention: full` when retention is omitted, preserving the sub-domain NC grids required for exact generated DA-event map rerendering.
- `output.grids.variables[*]` controls both which compact grid variables are exported and which metrics are written for each variable. If this block is omitted, all grid variables and metrics are written for backward compatibility.
- Generated DA-event maps need `analysis_mean` and `analysis_increment` for `snowdepth_daily`, because their snow-depth response panels show the event-weighted posterior and posterior-minus-prior increment.
- `results/grids/da_output_grids.nc` is aggregated over all project steps (full project timeline).
- In `da_output_grids.nc`, `increment_<var>` is the open-loop departure: `ens_mean_<var> - open_loop_<var>`.
- Event analysis fields `analysis_mean_<var>` and `analysis_increment_<var>` are written where assimilation weights are available; `analysis_increment_<var>` is `analysis_mean_<var> - ens_mean_<var>`.

## `step_XX.yml` (step level)
Generated by `project_skeleton` and usually not edited manually.

```yaml
start_date: 2023-03-12 00:00:00
end_date: 2023-03-16 21:00:00
results_dir: results
```

## Validation behavior
Configuration is validated when running CLI commands such as:
- `oa-da-project`
- `python -m openamundsen_da.pipeline.project_skeleton`
- `oa-da-snowcover`
- `oa-da-assimilate-scf`

Typical early failures:
- missing project YAML data assimilation keys
- missing ROI or land-cover grid
- missing required output variables for assimilation
- invalid dates/timestep alignment

## Best Practices
- Keep setup YAML stable and shared across projects.
- Keep data assimilation experimentation and observation mappings in project YAML.
- Use one project per experiment/time span.
- Keep `assimilation_events` explicit and versioned in each project.
