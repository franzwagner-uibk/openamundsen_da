---
layout: default
title: Workflow
nav_order: 4
---

# Data Assimilation Workflow

{: .no_toc }

Understanding the particle filter data assimilation cycle.
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

The openamundsen_da framework implements a sequential particle filter for snow data assimilation. The workflow cycles through prior generation, forecast propagation, and observation-based updates.

![Data Assimilation Architecture]({{ site.baseurl }}/assets/images/DataAssimilation_Design%20_DOCS%20_Architecture.drawio.png)

**Key components**:

- **Prior Generation** (orange): Perturb meteorological forcing to create ensemble input
- **Forecast/Propagation** (purple): Run openAMUNDSEN for each ensemble member
- **Update Cycle** (blue): Observation processing, likelihood computation, resampling, rejuvenation
- **Configuration** (yellow): setup YAML controls openAMUNDSEN settings, project YAML controls data assimilation settings and assimilation dates

![Data Assimilation Experiment Cycle]({{ site.baseurl }}/assets/images/Particle_Filter%20_DOCS.drawio.png)

---

## Prior Ensemble Generation

### Meteorological Forcing Perturbation

**Temperature**: Additive Gaussian noise

```
T_perturbed = T_original + epsilon_T,  epsilon_T ~ N(0, sigma_T^2)
```

**Precipitation**: Multiplicative log-normal noise

```
P_perturbed = P_original * exp(epsilon_P),  epsilon_P ~ N(mu_P, sigma_P^2)
```

**Relative humidity**: Additive Gaussian noise with clipping

```
RH_perturbed = clip(RH_original + epsilon_RH, 0, 100),  epsilon_RH ~ N(0, sigma_RH^2)
```

**Incoming shortwave radiation**: Multiplicative log-normal noise for daytime values only

```
SW_perturbed = SW_original * exp(epsilon_SW),  epsilon_SW ~ N(0, sigma_SW^2),  applied only if SW_original > 0
```

**Command**:

```bash
docker compose run --rm oa \
  python -m openamundsen_da.core.prior_forcing \
  --input-meteo-dir /data/meteo \
  --project-dir /data/projects/project_2019-2020 \
  --step-dir /data/projects/project_2019-2020/steps/step_01_*
```

Optional flags: `--overwrite`, `--log-level <LEVEL>`

---

## Model Execution

### Parallel Ensemble Runs

```bash
docker compose run --rm oa \
  python -m openamundsen_da.core.launch \
  --project-dir /data/projects/project_2019-2020 \
  --setup-dir /data \
  --step-dir /data/projects/project_2019-2020/steps/step_01_* \
  --ensemble prior \
  --max-workers 8
```

Optional flags: `--overwrite`, `--state-pattern <glob>`, `--log-level <LEVEL>`

**Parallelism**: The `--max-workers` value is an upper bound. The launcher clamps to `min(max_workers, CPUs visible, #members)`. An `open_loop` (unperturbed, unassimilated baseline) is run alongside the ensemble and carried through all steps for reference/plots.

### State Management

Warm start uses the model state saved at the end of each step via `state_pointer.json`:

```json
{
  "path": "/abs/or/rel/path/to/model_state.pickle.gz"
}
```

---

## Observation Processing

Besides satellite SCF and wet-snow observations, the workflow also supports
ROI-based station assimilation for:

- `station_hs`
- `station_swe`

These station observations are not spatialized over the grid. They are used to
reweight whole-ROI ensemble members from station point comparisons. See
[Station Assimilation]({{ site.baseurl }}{% link guides/station-assimilation.md %})
for the full method description.

### Observation Uncertainty

openAMUNDSEN-DA supports three uncertainty handling patterns, selected per product:

- `enabled: false`: no uncertainty layer is used; data assimilation uses `sigma_mode: formula`.
- `enabled: true` with externally provided uncertainty: ingest uncertainty from NetCDF (same file) or GeoTIFF sidecar (`<stem>_uncertainty.tif`).
- `enabled: true` with openAMUNDSEN-DA generation: create sidecar layers first via `oa-da-scf-uncertainty` / `oa-da-wetsnow-uncertainty`, then ingest them like any other sidecar.

All uncertainty values are expected on a `0..100` scale, and uncertainty-enabled preprocessing is strict fail-fast on missing/invalid inputs.
In openAMUNDSEN-DA generation mode, per-pixel uncertainty is built from a baseline plus additive penalties from multiple configured class sources (for example forest land cover and shadow masks).

The example below shows this logic for a Rofental SCF scene. The left panel is the observed snow-cover fraction, the middle panel is the resulting uncertainty field, and the right panel shows the land-cover driver. The zoomed row makes the penalty structure visible: uncertainty remains spatially continuous on valid observation pixels and increases where configured covariates such as forest classes apply. Gaps such as cloud-covered pixels are not turned into "high-uncertainty observations"; they remain missing data.

![Example SCF uncertainty decomposition for the Rofental ROI]({{ site.baseurl }}/assets/images/tutorial/rofental_uncertainty.png)

Observation uncertainty is configured per product under `data_assimilation.uncertainty`.
Ingestion is file-type based:

- NetCDF: value + uncertainty are read from variables in the same file.
- GeoTIFF: uncertainty is read from `<source_stem>_uncertainty.tif` next to each source raster.
- If uncertainty is enabled and required layers are missing, preprocessing fails fast.

Example YAML:

```yaml
data_assimilation:
  uncertainty:
    scf:
      enabled: true # enable uncertainty-aware SCF preprocessing + assimilation
      ingest:
        # Required when uncertainty is enabled:
        scf_variable: fsc
        uncertainty_variable: uncertainty
        time_variable: time
      assimilation:
        sigma_mode: uncertainty_layer # formula | uncertainty_layer
        aggregate_metric: unc_mean # used when sigma_mode=uncertainty_layer
      input_dir: obs/snowcover # used by oa-da-scf-uncertainty
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
          enabled: false # set true when shadow rasters are available
          input_dir: obs/shadow # directory of shadow rasters
          classes: [1]
          penalty: 20.0
    wet_snow:
      enabled: true # enable uncertainty-aware wet-snow preprocessing + assimilation
      ingest:
        # Required when uncertainty is enabled:
        wet_snow_variable: wet_snow
        uncertainty_variable: uncertainty
        time_variable: time
      assimilation:
        sigma_mode: uncertainty_layer # formula | uncertainty_layer
        aggregate_metric: unc_mean # used when sigma_mode=uncertainty_layer
      input_dir: obs/wetsnow # used by oa-da-wetsnow-uncertainty
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
          enabled: false # set true when shadow rasters are available
          input_dir: obs/shadow # directory of shadow rasters
          classes: [1]
          penalty: 20.0
```

Uncertainty key reference:

- `enabled`: turns uncertainty-aware preprocessing and assimilation on/off for that product.
- `ingest.*_variable` / `ingest.uncertainty_variable` / `ingest.time_variable`: strict NetCDF variable names; no defaults.
- `assimilation.sigma_mode`: `formula` uses legacy likelihood sigma, `uncertainty_layer` uses configured uncertainty metric.
- `assimilation.aggregate_metric`: summary column name consumed in `uncertainty_layer` mode (typically `unc_mean`).
- `input_dir`: source directory for openAMUNDSEN-DA uncertainty generation tools (`oa-da-scf-uncertainty`, `oa-da-wetsnow-uncertainty`).
- `u_min`, `u_max` (SCF): triangular baseline uncertainty bounds.
- `base_uncertainty` (wet snow): baseline uncertainty for base wet-snow classes.
- `nodata_value`: nodata marker written by uncertainty generation tools.
- `penalties[].name`: free label used in generator summary diagnostics.
- `penalties[].source`: class source for matching (`fsc`/`wet_snow`, `landcover`, `shadow`).
- `penalties[].enabled`: activate/deactivate that rule while keeping it in config.
- `penalties[].input_dir`: required for `source: shadow`.
- `penalties[].classes`: raw class IDs to match.
- `penalties[].penalty`: additive uncertainty penalty in percentage points.

For uncertainty-aware assimilation (`sigma_mode: uncertainty_layer`), observation sigma is derived from summary metrics (for example `unc_mean`) instead of the likelihood formula mode.

To generate GeoTIFF uncertainty companions with openAMUNDSEN-DA:

1. generate uncertainty companion rasters,
2. then run observation summarizers (`oa-da-snowcover`, `oa-da-wetsnow`),
3. then create per-step obs CSVs (`oa-da-scf`, `oa-da-wetsnow-project`).

Best-practice split:

- Use land-cover exclusion for structurally unusable classes (for example ice/water/urban).
- Use uncertainty penalties for usable-but-uncertain classes (for example forest/shadow).
- Treat cloud pixels as gaps (masked), not as uncertainty-penalty contributors.

### Optional uncertainty companion generation

Run these when `data_assimilation.uncertainty.<variable>.enabled: true` and
you want openAMUNDSEN-DA to create `*_uncertainty.tif` files:

```bash
docker compose run --rm oa oa-da-scf-uncertainty \
  --setup-dir /data \
  --project-label project_2019-2020 \
  --overwrite

docker compose run --rm oa oa-da-wetsnow-uncertainty \
  --setup-dir /data \
  --project-label project_2019-2020 \
  --overwrite
```

This writes `*_uncertainty.tif` next to source SCF/wet-snow GeoTIFFs.

### Snow cover (GeoTIFF/NetCDF -> `scf_summary.csv`)

```bash
docker compose run --rm oa \
  oa-da-snowcover \
  --input-dir /data/obs/snowcover \
  --project-label project_2019-2020 \
  --setup-dir /data
```

Classes are read from `obs.snowcover.classes` in project YAML (defaults: valid 0-100, cloud 205, water 210, nodata 255). The project-level data assimilation land-cover mask is applied to observations automatically.

### Wet snow (categorical rasters -> `wet_snow_summary.csv`)

```bash
docker compose run --rm oa \
  oa-da-wetsnow \
  --input-dir /data/obs/wetsnow \
  --project-label project_2019-2020 \
  --setup-dir /data
```

Wet/valid/exclude classes come from `obs.wetsnow.classes` in the project YAML (for HRWSI WSM typically wet `110`, dry/no-snow `125`, with `200`/`210` excluded). ROI and land-cover masking mirror the snow-cover workflow.

By default, both summaries are written to `/data/obs/summaries/<project-label>/scf_summary.csv` and `wet_snow_summary.csv`. Override with `--output-root` if you want a different location.

### Per-step obs CSVs

```bash
docker compose run --rm oa oa-da-scf --project-dir /data/projects/project_2019-2020 --overwrite
docker compose run --rm oa oa-da-wetsnow-project --project-dir /data/projects/project_2019-2020 --overwrite
```

Outputs: `step_*/obs/obs_scf_<PRODUCT>_YYYYMMDD.csv` and `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv`, with product tags resolved from project YAML (`obs.snowcover.product_tag`, `obs.wetsnow.product_tag`).
Preparation is fail-fast: one event per non-final step is required, each event date must fall inside its step window, and each configured event date must exist in the corresponding summary CSV.

### Land-Cover Masking

Configured in project YAML:

```yaml
data_assimilation:
  landcover_mask:
    # Classes: 1 rock, 2 ice, 3 water, 4 grassland, 5 shrubland, 6 farmland,
    # 7 transitional, 8 deciduous 30-60, 9 deciduous 60-100, 10 mixed,
    # 11 coniferous 30-60, 12 coniferous 60-100, 13 built-up.
    enabled: true
    classes_to_exclude: [2, 8, 9, 10, 11, 12, 13]
```

Excluded land-cover classes are removed from both observations and model-derived fractions. A warning is logged if >50% of the ROI would be excluded; 100% exclusion fails.

---

## Data Assimilation

### H(x) Forward Operator

Maps model state (snow depth or SWE) to observation space (SCF).

**Methods**:

1. **Depth Threshold**:

   ```
   SCF = 1  if HS > h0
   SCF = 0  otherwise
   ```

2. **Logistic** (smooth):
   ```
   SCF = 1 / (1 + exp(-k * (HS - h0)))
   ```

**Configuration** (in project YAML):

```yaml
data_assimilation:
  h_of_x:
    method: depth_threshold # or "logistic"
    variable: hs # or "swe"
    params:
      h0: 0.01
      k: 80
```

### Likelihood Calculation

Gaussian likelihood function:

```
w_i is proportional to exp(-0.5 * ((y_obs - H(x_i)) / sigma_obs)^2)
```

Weights are normalized: `w_i = w_i / sum_j(w_j)`

### Effective Sample Size (ESS)

```
ESS = 1 / sum_i(w_i^2)
```

- ESS = N: All weights equal (no information from obs)
- ESS = 1: One particle dominates (particle degeneracy)
- ESS < threshold -> Trigger resampling

---

## Ensemble Update

### Systematic Resampling

**Configuration**:

```yaml
data_assimilation:
  resampling:
    algorithm: systematic
    ess_threshold_ratio: 0.5 # Resample if ESS < 0.5 * N
    seed: 42
```

**Behavior**:

- If `ESS >= threshold`: Skip resampling, mirror prior -> posterior
- If `ESS < threshold`: Resample

### Rejuvenation

After resampling, ensemble spread is reduced. Rejuvenation adds noise to maintain diversity.

Perturbations are always applied relative to the open loop forcing:

```
forcing_new = open_loop_forcing + new_perturbation
```

**Configuration**:

```yaml
data_assimilation:
  rejuvenation:
    sigma_t: 0.2 # Additive temperature noise (deg C)
    sigma_p: 0.2 # Lognormal sigma for precip factor (mu=0)
    sigma_rh: 0.0 # Additive relative humidity noise (percentage points)
    sigma_sw: 0.0 # Lognormal sigma for daytime shortwave factor (mu=0)
```

If rejuvenation sigmas are not set, they fall back to prior_forcing sigmas. `sigma_rh` and `sigma_sw` default to `0.0` when omitted.

### State Propagation

Copy posterior states + perturbed forcing to next step's prior:

```
step_N/ensembles/posterior/member_i/ -> step_N+1/ensembles/prior/member_j/
```

---

## Setup Pipeline

The setup pipeline automates all phases:

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.project \
  --setup-dir /data \
  --project-dir /data/projects/project_2019-2020 \
  --max-workers 8 \
  --monitor-perf
```

Optional: `--overwrite`, `--live-plots`, `--log-level <LEVEL>` (`--live-plots` enables in-run plotting; default is off and plots run once after completion)

**Pipeline steps** (per assimilation cycle):

1. Generate prior forcing
2. Run prior ensemble
3. Compute model H(x) (SCF/wet-snow)
4. Assimilate observations -> weights
5. Check ESS -> resample if needed
6. Rejuvenate -> next prior
7. Generate plots
8. Repeat for next step

**Outputs**:

- Per-step runs in `<step>/ensembles/{prior,posterior}`
- Weights and indices in `<step>/assim/`
- Rejuvenated next-step prior with `state_pointer.json`
- Setup plots under `<setup_dir>/plots/{forcing,results}`

---

## State cleanup

- Automatic: `data_assimilation.restart.cleanup_after_setup: true` (default, in project YAML) removes member state pickle files after a successful project run to save disk space.
- Manual: run the cleanup CLI to delete state files even if automatic cleanup is disabled.

Clean all projects under one setup:

```powershell
oa-da-clean-project --setup-dir /data/your_setup --all-projects --log-level INFO
```

Clean one project:

```powershell
oa-da-clean-project --setup-dir /data/your_setup --project-dir /data/your_setup/projects/project_YYYY-YYYY --log-level INFO
```

Only state pickle files are removed; `state_pointer.json` files are left in place.

---

## Configuration Reference

### Likelihood Settings

```yaml
data_assimilation:
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
```

### Warm Start Settings

```yaml
data_assimilation:
  restart:
    dump_state: true
    state_pattern: model_state.pickle.gz
```

---

## Next Steps

- [Configuration Guide]({{ site.baseurl }}{% link guides/configuration.md %}) - Detailed configuration reference
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments/index.md %}) - Step-by-step experiment setup
- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) - Command-line tools

---

## References

- Barella, R., Marin, C., Gianinetto, M., and Notarnicola, C. (2022). A novel approach to high resolution snow cover fraction retrieval in mountainous regions. IGARSS 2022 - IEEE International Geoscience and Remote Sensing Symposium, 3856-3859. https://doi.org/10.1109/IGARSS46834.2022.9884177.
- Nagler, T., Rott, H., Ripper, E., Bippus, G., and Hetzenecker, M. (2016). Advancements for snowmelt monitoring by means of Sentinel-1 SAR. Remote Sensing, 8(4), 348. https://doi.org/10.3390/rs8040348.
