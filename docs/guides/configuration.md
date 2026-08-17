---
layout: default
title: Configuration
parent: Documentation
nav_order: 3
---

# Configuration
{: .no_toc }

Strict setup, project and step configuration ownership for openAMUNDSEN-DA.
{: .fs-6 .fw-300 }

## Hierarchy
1. `<setup-name>.yml` - setup-wide openAMUNDSEN config.
2. `<project-name>.yml` - project-specific data assimilation config and time span.
3. `step_XX.yml` - auto-generated step windows.

Generic `setup.yml` and `project.yml` aliases are not accepted.
The setup root must contain exactly one non-legacy `.yml` file. This keeps the
configuration unambiguous when Docker mounts a named setup directory as
`/data`; the logical setup filename does not change at the mount boundary.

![Setup, project and step configuration ownership in openAMUNDSEN-DA]({{ site.baseurl }}/assets/images/diagrams/setup-project-configuration.png)

*Configuration ownership from the shared openAMUNDSEN setup through one data
assimilation project to generated step windows.*

## `<setup-name>.yml` (setup level)
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

`output_data.grids.format` is required and selects the one model-grid reader.
Use `netcdf` for grid-layout NetCDF or `geotiff` for deterministic,
georeferenced daily GeoTIFFs. Mixed artifacts and NetCDF `roi_pixel` layout are
rejected.

## `<project-name>.yml` (project level)
Use project YAML for data assimilation configuration for one project.

Required top-level keys:
- `start_date`
- `end_date`
- `run_mode`
- `obs`
- `data_assimilation`

Example:

```yaml
start_date: 2022-10-01
end_date: 2023-09-30
run_mode: single

obs:
  stations:
    dir: obs/stations
  snowcover:
    dir: obs/snowcover
    format: geotiff
    product_tag: SNOWCOVER
    acquisition_manifest: obs/satellite_acquisition_times.csv
    summary_csv: obs/summaries/project_2022_2023/scf_summary.csv
    classes:
      # Example only: use the class set required by your product
      valid: [0, 1, 2, 3, 4, 5]
      cloud: [205]
      water: [210]
      nodata: [255]
  wetsnow:
    dir: obs/wetsnow
    format: geotiff
    product_tag: WETSNOW
    acquisition_manifest: obs/satellite_acquisition_times.csv
    filename_time_parser: sentinel_1
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
    sigma_rh: 0.0 # dew-point temperature perturbation scale
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
      min_support_coverage_ratio: 0.10
    wet_snow:
      obs_sigma: 0.15
      use_binomial: false
      sigma_floor: 0.03
      sigma_cloud_scale: 0.10
      min_sigma: 0.03
      min_support_coverage_ratio: 0.10
    wet_snow_line:
      obs_sigma: 150.0
      use_binomial: false
      sigma_floor: 25.0
      min_sigma: 25.0
      min_support_coverage_ratio: 0.10
      min_model_finite_fraction: 1.0 # set 0.90 for WSLA sensitivity experiments
      min_wet_pixels_total: 50
      min_wet_bands: 1

  resampling:
    algorithm: systematic
    ess_threshold_ratio: 0.5
    seed: 43

  rejuvenation:
    sigma_t: 0.2
    sigma_p: 0.2
    sigma_rh: 0.0 # dew-point temperature perturbation scale
    sigma_sw: 0.0
    seed: 44

  restart:
    dump_state: true
    state_pattern: model_state.pickle.gz

  benchmark:
    independent_variables:
      - station_swe
    score_station_sigma_threshold: 200
    plots: true
    output_dir: results/benchmark

  output:
    retention: compact
    grids:
      format: netcdf # compact DA output is always NetCDF

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
      input_dir: obs/snowcover
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
      input_dir: obs/wetsnow
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
      # Optional when one scene exists on the date; required to disambiguate several scenes.
      observation_time: 2023-03-17T10:21:00Z
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
- `observation_time` is an optional full timezone-aware ISO-8601 selector.
  Date-only selection is valid when exactly one scene exists. Several scenes on
  one date require `observation_time`; the first row is never selected implicitly.
- Station observation assimilation uses `variable: station_hs` or `variable: station_swe` and does not require a product tag.
- Station observations live in `obs/stations/<station_id>.csv`; station DA metadata live in `obs/stations/stations_da_metadata.csv`.
- A station event runs at the event date combined with the active step's start
  time. The nearest same-ID, `use_for_da` observation must lie within half the
  setup timestep. Naive timestamps use the setup timezone. Exactly two
  equidistant values that symmetrically bracket the model time are averaged
  when they are no more than 24 hours apart; other ties and values farther away
  fail. The corresponding model point must exist exactly at the model-clock
  timestamp.
- `assimilation_events` is the final event selection. Discovery, quality
  filtering and date substitution must happen before project execution. The
  removed `subdomain_event_filter` key is rejected instead of mutating a project.
- Active DA and benchmark station IDs must resolve case-insensitively to both a same-ID observation CSV and a configured model output point. Explicit points are checked directly; default meteo points are resolved against the setup ROI.
- `data_assimilation.station` defines project-level percentage defaults and single-station inflation for ROI-based station assimilation.
- Station absolute sigma floors are configured per station in `stations_da_metadata.csv` via `hs_sigma_abs_min` and `swe_sigma_abs_min`.
- See [Station Assimilation]({{ site.baseurl }}{% link guides/station-assimilation.md %}) for the method logic, effective sigma definition, single-station handling, and diagnostics.
- Observation class mappings and product tags are configured under project YAML `obs.*`.
- `data_assimilation.benchmark` does not enable or disable benchmarking; the project pipeline always runs it. This block extends the benchmark scope and controls benchmark output location and plot writing. The benchmark presentation itself is fixed and lean: one assimilation-date skill plot plus two compact summary tables.
- `independent_variables` may currently list only the DA-supported families: `scf`, WSF (`wet_snow`), WSLA (`wet_snow_line`), `station_hs`, `station_swe`.
- `score_station_sigma_threshold` optionally excludes high-uncertainty station rows from non-sigma-aware benchmark metrics (`CRPSS`, `NER`) while leaving sigma-aware `zSkill` unchanged. The threshold is compared against the resolved station uncertainty percent from `obs/stations/stations_da_metadata.csv`.
- `prior_forcing.random_seed`, `resampling.seed` and `rejuvenation.seed` are
  required non-negative scientific seeds. Random draws use stable event/member/
  variable keys and are invariant to worker scheduling and member ordering.
- Each fractional observable used by `assimilation_events` requires its own
  complete `data_assimilation.likelihood.<observable>` mapping. The values shown
  above are the full contracts; missing, unknown, nonfinite or out-of-range
  likelihood settings are errors and are never replaced by defaults.
- `prior_forcing.sigma_rh` samples an additive dew-point temperature perturbation. When station CSVs contain both `temp` and `rel_hum`, the forcing helper converts temperature and relative humidity to dew point, applies the sampled dew-point offset, caps dew point at the perturbed air temperature and recalculates `rel_hum` in `[0, 100]`. Temperature perturbations also update `rel_hum` through this dew-point transform when both columns are available.
- `prior_forcing.sigma_sw` adds a multiplicative `sw_in` perturbation using a positive factor; it is applied only for positive daytime shortwave values, so nighttime `sw_in` remains unchanged.
- Rejuvenation is a fresh process-noise forcing refresh, not an MCMC
  resample-move step. It inherits omitted distribution parameters from
  `prior_forcing`, including `mu_p`. `rebase_open_loop` is unsupported because
  forcing is always rebuilt from the unmodified setup forcing.
- Output stream labels are derived by benchmark semantics, not by config naming alone: a configured extra family can still appear as `semi_independent` in outputs, but only from the first same-variable or sister-station assimilation date onward.
- Land-cover mask uses `grids/lc_<domain>_<resolution>.asc` from setup-level paths and data assimilation mask classes from project YAML.
- For SCF uncertainty:
  - `enabled: true` activates strict uncertainty checks (fail-fast on missing/invalid config or layers).
  - `sigma_mode: uncertainty_layer` uses `aggregate_metric / 100` (for example `unc_mean`) with `min_sigma` floor.
  - The aggregate is an effective, uncalibrated comparison-error sigma. The
    uncertainty layer must cover every valid observation pixel; missing,
    nonfinite, out-of-range or incomplete coverage is an error with no fixed
    sigma substitution.
  - NetCDF uses configured in-file variables; GeoTIFF requires `<stem>_uncertainty.tif`.
- Cloud pixels should be handled as data gaps (masked), not as uncertainty-penalty pixels.
- Wet-snow uncertainty uses the same pattern (`ingest` + `assimilation`) and the same file-type behavior.
- Uncertainty preprocessing keys:
  - `input_dir`, `u_min`, `u_max`, `base_uncertainty`, `nodata_value`, and `penalties[]` are used by `openamundsen-da observations snow-cover` and `openamundsen-da observations wet-snow`.
  - `penalties[].input_dir` is required only for `source: shadow`.
- `output.retention: compact` writes the configured grid summaries, compressed
  all-member point and consumed-forcing time series and satellite-event map
  support. Fresh runs route disposable member CSVs, grids, restart checkpoints
  and forcing plots through one generation-owned runtime tree. After
  benchmarking and rendering validate, that tree is quarantined and physically
  removed through retention-ledger schema v6 without a per-file raw inventory.
  Older member-local projects retain their schema-v5 batch cleanup path.
- `output.retention: full` preserves member forcing, points, grids and restart
  artifacts for reanalysis. `run_mode: subdomain` defaults to `full` when the
  key is omitted; single-domain projects default to `compact`.
- Retention values other than `compact` and `full` are configuration errors;
  they are never replaced silently.
- Perturbed forcing for every member covers only the exact inclusive
  `start_date` to `end_date` window in its consuming step YAML.
- Before admitting a step, the project filesystem must be below the fixed 80%
  soft limit and its remaining conservative plan plus the fixed 5% operational
  reserve must remain below 90%. Full planning occurs before execution and at
  lifecycle transitions. Ordinary step boundaries use durable producer
  accounting plus one filesystem usage check. A low-disk stop is recorded as
  resumable and never implies overwrite.
- Subdomain mode reserves accumulated forcing, point, raw-grid, compact-output
  (including satellite map support) and one retained restart-checkpoint growth
  for every unfinished leaf. It adds
  a second rolling checkpoint for the largest leaves allowed by outer
  concurrency and one full atomic parent-merge temporary. The reservation is
  reconciled from measured artifacts at every leaf-wave boundary; all selected
  projects must share the parent filesystem. This coordinator prevents leaves
  from exceeding the conservative reservation, but the deliberately broad
  first-run envelope can still refuse a workload that might fit in practice.
- First-run bounds use every configured grid variable and output timestamp,
  exact selected source rows and bytes in each forcing file, 8 bytes per grid
  cell/value, 4096 bytes per restart cell/member and the current 40 default
  point variables with conservative soil/snow layer expansion. Explicit
  variables and layers add to that bound. Atomic overwrite reserves the full
  point, forcing, grid and map-support replacement temporary. Observed point,
  grid and state artifacts can only refit these rates upward.
  The fixed 5% operational reserve remains separate from predicted model growth.
- Checks occur between steps and finalization stages; this increment does not
  terminate active openAMUNDSEN members mid-propagation.
- Compact point, forcing and grid cleanup occurs after successful project-level
  compaction, benchmarking and rendering. Predecessor restart checkpoints are
  removed incrementally between steps and deducted from the generation's
  producer accounting.
- Overlapping step-boundary timestamps in compact point and forcing NetCDFs use
  the same numeric mean as the raw-series plot and benchmark readers. Cleanup
  compares retained values, not only dimensions and identities, with the raw
  sources. Leaf `da_output_grids.nc` summaries remain available for rerendering
  leaf snow-depth and SWE maps after raw member-grid cleanup.
- Restart cleanup requires readable successor checkpoints for the open loop and
  exactly `member_001` through the configured ensemble size. A dump failure is
  fatal whenever another step follows. Grid and
  map-support cleanup also validates configured metric completeness, geometry,
  ROI/domain constraints and source values before deleting raw member grids.
- Compact and checkpoint temporaries are scientifically validated, flushed and
  atomically promoted before ledger-backed deletion. Schema v6 binds the whole
  contained runtime generation to byte-identical retained consumers and actual
  producer member manifests before same-filesystem quarantine; the legacy v5
  path retains its per-batch dependency and resumed-unlink checks.
- This boundary-based increment may conservatively refuse a full 100 m Euregio
  ES50 run on 3.6 TB. Immediate per-leaf finalization/cleanup and measured
  prepared-setup capacity validation remain required before claiming that
  production acceptance.
- `output.grids.variables[*]` controls both which compact grid variables are exported and which metrics are written for each variable. If this block is omitted, all grid variables and metrics are written for backward compatibility.
- Every explicit `output.grids.variables[*].var` or `name` must match a
  `setup.output_data.grids.variables[*].name`. This is validated before model
  propagation. Maintained snow setups request `snow.depth` as
  `snowdepth_daily` and `snow.swe` as `swe_daily` so both standard products are
  available to the compact exporter.
- Explicit compact output contracts are strict: every open-loop and member
  NetCDF must contain every requested source variable, and the completed
  `da_output_grids.nc` must contain every requested metric-variable pair.
  Validation reports all missing names together instead of writing a partial
  scientific result. Projects without an explicit compact-variable list keep
  the legacy all-available-output behavior.
- Compact DA summary NetCDFs use internal compressed storage encodings: snow depth at 0.001 m resolution and SWE/liquid-water content at integer millimeter resolution. This is not a YAML setting; CF-aware readers decode the variables back to physical values.
- Generated DA-event maps need `analysis_mean` and `analysis_increment` for `snowdepth_daily`, because their snow-depth response panels show the event-weighted posterior and posterior-minus-prior increment.
- `results/grids/da_output_grids.nc` is aggregated over all project steps (full project timeline).
- Compact projects additionally retain
  `results/points/ensemble_points.nc`,
  `results/forcing/ensemble_forcing.nc` and, when satellite observations are
  configured, `results/grids/da_map_support.nc`. These NetCDF files preserve
  open-loop and every ensemble member on explicit time/member/point or
  time/member/station dimensions and are readable with CF-aware tools such as
  xarray, netCDF4, R `ncdf4` and R `stars`.
- In `da_output_grids.nc`, `increment_<var>` is the open-loop departure: `ens_mean_<var> - open_loop_<var>`.
- Event analysis fields `analysis_mean_<var>` and `analysis_increment_<var>` are written where assimilation weights are available; `analysis_increment_<var>` is `analysis_mean_<var> - ens_mean_<var>`.
- Satellite operators require instantaneous `snowdepth_instantaneous`,
  `swe_instantaneous` and `liquid_water_content_instantaneous` model outputs.
  Observation time is matched to the unique nearest model timestep within half
  a timestep; ties and larger offsets are rejected.

## `step_XX.yml` (step level)
Generated by `openamundsen-da prepare` and not edited manually.

```yaml
start_date: 2023-03-12 00:00:00
end_date: 2023-03-16 21:00:00
results_dir: results
```

## Validation behavior
Configuration is validated when running CLI commands such as:
- `openamundsen-da observations snow-cover PROJECT_DIR`
- `openamundsen-da observations wet-snow PROJECT_DIR`
- `openamundsen-da prepare PROJECT_DIR`
- `openamundsen-da run PROJECT_DIR`

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
