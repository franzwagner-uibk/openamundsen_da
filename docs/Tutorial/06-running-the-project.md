---
layout: default
title: 6. Running the Model
parent: How to Use
nav_order: 6
permalink: /tutorial/running-the-project/
---

# 6. Running the Model

The preparation manifest and step inputs are now complete. Continue with the
`PROJECT_DIR` variable set in the preprocessing chapter. The remaining command
executes the open loop and ensemble, performs the configured updates and validates
the complete output set.

## Review runtime configuration

The setup YAML selects the 100 m openAMUNDSEN domain and its model grid format.
The project YAML selects the ensemble size, event sequence and compact result
variables. The tutorial reference uses `ensemble_size: 30` and eight events.
Selected scientific fields are shown below; the shipped YAML remains the
complete configuration.

```yaml
data_assimilation:
  prior_forcing:
    ensemble_size: 30
    random_seed: 1415935400
    sigma_t: 1.0
    mu_p: -0.055
    sigma_p: 0.6
    sigma_rh: 1.2
    sigma_sw: 0.15
  resampling:
    ess_threshold_ratio: 0.7
    seed: 1415935400
  rejuvenation:
    sigma_t: 1.0
    mu_p: -0.055
    sigma_p: 0.6
    sigma_rh: 1.2
    sigma_sw: 0.15
    seed: 1415935400
  likelihood:
    wet_snow_line:
      obs_sigma: 200.0
      min_model_finite_fraction: 0.95
  uncertainty:
    scf:
      enabled: true
      assimilation:
        sigma_mode: uncertainty_layer
        aggregate_metric: unc_mean
  output:
    retention: full
    grids:
      format: netcdf
      variables:
        - var: snowdepth_daily
          metrics: [open_loop, ens_mean, ens_std, ens_min, ens_max, increment, analysis_mean, analysis_increment]
        - var: swe_daily
          metrics: [open_loop, ens_mean, ens_std, ens_min, ens_max, increment]
```

The fixed seeds and keyed random-number scheme make the scientific ensemble
reproducible in the pinned release image. Worker count changes scheduling and
runtime but not the configured random sequence. The precipitation parameters
describe a lognormal multiplicative factor; `mu_p=-0.055` and `sigma_p=0.6`
give an arithmetic mean factor of about 1.133.

## Run the project

Purpose: execute the prepared single-domain project, benchmark it, render every
configured plot/map/report and finalize cleanup only after validation succeeds.

The positional argument is the project directory. Replace `<MAX_WORKERS>` with
a number no larger than the `<CPU_COUNT>` exposed to the container in Chapter 2;
do not type the angle brackets. For this ES30 project, at most 31 propagations
can run at once. On a 24-core machine, use `24`; on a machine exposing 48 CPUs,
use at most `31`. Use a lower value when memory is limited. The numerical-library
thread variables remain set to `1`.

`--json` emits the typed run result as one JSON envelope. There is intentionally
no run-level `--overwrite`: completed projects are immutable and interrupted
projects resume only from hash-identical inputs.

**🟢 Run command:**

```bash
openamundsen-da run "$PROJECT_DIR" --max-workers <MAX_WORKERS>
```

The runtime can take from tens of minutes to several hours depending on host,
Docker allocation and storage. Keep the container open. Performance monitoring
is automatic; do not start a separate monitor.

## Completion contract

The command returns success only when all of these are valid:

- all open-loop and member propagations;
- all eight assimilation updates and their diagnostics;
- `results/grids/da_output_grids.nc`;
- benchmark tables and score plots;
- configured plots and maps;
- `results/reports/project_report.pdf`; and
- automatic removal of package-owned restart state.

The atomic `results/run_manifest.json` always records the openAMUNDSEN-DA
software version. It also records `image` and `image_digest` when the runtime
supplies `OPENAMUNDSEN_DA_IMAGE` and `OPENAMUNDSEN_DA_IMAGE_DIGEST`; otherwise
those optional fields are empty. Completed/reused members, the output inventory
and cleanup record are included independently of this optional image metadata.
Interrupted or failed projects keep restart state.

Before interpretation, confirm that the manifest reports `status: success` and
that the performance plot, result overview, compact NetCDF and PDF report exist.
The next chapter explains these outputs without repeating the run command.

Continue with [7. Results and Diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}).
