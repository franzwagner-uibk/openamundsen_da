---
layout: default
title: 6. Running the Model
parent: How to Use
nav_order: 6
permalink: /tutorial/running-the-project/
---

# 6. Running the Model

The preparation manifest and step inputs are now complete. The remaining command
executes the open loop and ensemble, performs the configured updates and validates
the complete output set.

**🟢 Run command:**

```bash
PROJECT_DIR=/data/rofental/projects/project_2022_2023
echo "$PROJECT_DIR"
```

## Review runtime configuration

The setup YAML selects the 100 m openAMUNDSEN domain and its model grid format.
The project YAML selects the ensemble size, event sequence and compact result
variables. The tutorial reference uses `ensemble_size: 30` and eight events.

```yaml
data_assimilation:
  prior_forcing:
    ensemble_size: 30
    random_seed: 113
  resampling:
    ess_threshold_ratio: 0.7
    seed: 113
  rejuvenation:
    seed: 113
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

The frozen seeds make the scientific ensemble reproducible in the pinned release
image. Worker count changes scheduling and runtime but not the configured random
sequence.

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

The atomic `results/run_manifest.json` records the input digest, software/image
provenance, completed/reused members, output inventory and cleanup record.
Interrupted or failed projects keep restart state.

Before interpretation, confirm that the manifest reports `status: success` and
that the performance plot, result overview, compact NetCDF and PDF report exist.
The next chapter explains these outputs without repeating the run command.

Continue with [7. Results and Diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}).
