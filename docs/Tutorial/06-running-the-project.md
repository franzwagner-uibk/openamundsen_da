---
layout: default
title: 6. Running the project
parent: Tutorial
nav_order: 6
permalink: /tutorial/running-the-project/
---

# 6. Running the project

This chapter shows how to execute the tutorial project:

- first conceptually with the main building blocks (what the pipeline does),
- then practically with the recommended **project pipeline** command.

The tutorial baseline uses:

- **Rofental** (`examples/rofental`)
- **100 m** setup
- **ensemble size = 10**
- **October-June** season window

This configuration is intended to remain feasible on a normal computer.

The main run uses the project pipeline, but you already prepared the inputs manually in the previous chapter. This is intentional: understand first, automate second.

**Command focus (important):** The only mandatory command in this chapter is the project pipeline run. The step inspection and log-tail command are optional but useful.

All command blocks below are executed **inside the running tutorial container shell** started in [2. Dependencies]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %}). The container CPU limit and BLAS/OpenMP environment variables are configured in that startup command.

---

## Step-by-step flow on this page

{: .step }
> Follow this chapter in order: start the run, validate progress, then inspect outputs and rerun patterns.

Use this page in the following order:

1. read the pipeline overview (what the main run command will do)
2. optionally inspect one step folder (quick sanity check before a longer run)
3. run the **project pipeline command** (mandatory)
4. perform a small set of run-progress checks (recommended)
5. use the lower-level commands only to understand/debug internals (optional)

**Mandatory command on this page:** the project pipeline run in `Run the full project pipeline (recommended workflow)`.

If you are short on time, skip the optional step inspection and lower-level command section. Keep the pipeline run and the basic run-progress checks.

---

## What the project pipeline does (conceptual overview)

The project pipeline (`openamundsen_da.pipeline.project`) orchestrates the DA cycle step by step.

For each step it typically runs:

1. run the prior ensemble (open loop + ensemble members),
2. compute model-side diagnostics (e.g. SCF / wet-snow fractions),
3. read the per-step observation CSV from `steps/<step>/obs/`,
4. compute likelihood / weights,
5. resample if needed,
6. rejuvenate posterior particles into the next prior,
7. create plots and aggregate project-level diagnostics,
8. write compact DA outputs (e.g. `da_output_grids.nc`).

This is why the preprocessing chapter is critical: the pipeline expects the per-step
observation CSVs to already exist and match the configured event dates exactly.

{: .warning }
> The project pipeline does not replace observation preprocessing. Missing or inconsistent per-step observation CSVs will cause the run to fail.

{: .references }
> - [Framework]({{ site.baseurl }}{% link Tutorial/04-framework.md %}) for setup/project/step/member concepts
> - [Workflow Guide]({{ site.baseurl }}{% link workflow.md %}) for the broader DA process
> - [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) for pipeline and lower-level commands

---

## (Optional) Inspect one step folder before the run

This is a useful sanity check before launching a multi-minute run.

```bash
echo "Step folder: /data/rofental/projects/project_2022_2023/steps/step_00_init"
find /data/rofental/projects/project_2022_2023/steps/step_00_init -maxdepth 3 -type f | sort
```


What to expect before the first run:

- `step_00.yml`
- `obs/obs_scf_...csv` (for the first assimilation date)
- no model results yet

Reference snippet (`step_00_init/obs/` before the run):

```text
obs_scf_SNOWCOVER_20221003.csv
```

The first full run creates the ensemble folders, member outputs, diagnostics, plots,
and compact DA grid summaries.

<details markdown="block">
  <summary>What is created during the run (overview)</summary>

- step/member result folders
- DA diagnostics (weights, ESS)
- setup-level result plots
- ROI envelope CSVs
- compact DA NetCDF output
- project log with step-by-step execution messages
</details>

---

## Run the full project pipeline (recommended workflow)

Run this inside the tutorial container shell. `--setup-dir` points to the openAMUNDSEN setup, `--project-dir` selects the DA project, `--max-workers` caps pipeline parallelism, `--overwrite` allows reruns, and `--log-level INFO` keeps progress visible in the project log.

This is the main tutorial command for the full DA run.

**Configuration files used by this run**  
Setup config: `/data/rofental/rofental.yml` (openAMUNDSEN domain/model configuration)  
Project config: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml` (DA events, obs mapping, likelihood/resampling, outputs)

```bash
python -m openamundsen_da.pipeline.project \
  --setup-dir /data/rofental \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --max-workers 8 \
  --overwrite \
  --log-level INFO
```


Runtime coordination for stable performance:

- `--cpus 8` (set in the container startup command): caps container CPU usage (example value, adjust to your machine)
- BLAS/OpenMP env vars (set in the container startup command): prevent nested threading and oversubscription
- `--max-workers 8` (set here): upper bound for parallel workers in the pipeline

These values are examples. Adjust them to your machine and Docker CPU allocation before starting the tutorial container shell.

**Project YAML keys that strongly affect runtime/results**  
Reference YAML snippet (selected runtime-relevant keys)

File path: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`

```yaml
data_assimilation:
  prior_forcing:
    ensemble_size: 10

  assimilation_events:
    - date: "2022-10-03"
      variable: scf
      product: SNOWCOVER
    - date: "2023-03-28"
      variable: wet_snow
      product: WETSNOW

  output:
    retention: compact
    grids:
      format: netcdf
      variables:
        - var: snowdepth_daily
          metrics: [open_loop, ens_mean, ens_std, ens_min, ens_max, increment]
        - var: swe_daily
          metrics: [open_loop, ens_mean, ens_std, ens_min, ens_max, increment]
```

This snippet shows the three project settings that most visibly change runtime and outputs in the tutorial: ensemble size, number/timing of events, and which compact-grid variables/metrics are exported.

### Runtime expectations (what affects runtime)

Runtime depends on:

- CPU count / clock speed,
- storage speed,
- Docker Desktop overhead (Windows/macOS),
- whether this is the first run (cold caches) or a rerun,
- current project configuration (ensemble size, resolution, number of steps).

For the tutorial baseline (`100 m`, `ens=10`, `Oct-Jun`), a normal desktop/laptop
should be able to complete the run, but it is still a non-trivial workload.

First runs are often slower than reruns because of cold caches and filesystem effects.

Reference run (tutorial baseline, `ens=10`) completed in about `866.7 s` on one local test machine. Treat this only as a rough order of magnitude, not a target.

---

## Validate that the run is progressing correctly

{: .checks }
> Use these checks while the run is active to catch problems early.

### 1. Watch the project log

```bash
tail -n 80 /data/rofental/projects/project_2022_2023/project_2022_2023.log
```


Look for messages such as:

- step discovery (`Discovered ... step(s)`)
- prior ensemble launch (`Launching ensemble (prior) ...`)
- assimilation actions (`Assimilating scf`, `Assimilating wet_snow`)
- plot tasks completed
- final success (`Project processing complete`)

Reference snippet (successful log tail, condensed):

```text
... Plot task setup_ess_timeline completed
... Plot task setup_results_swe completed
... Plot task setup_results_snow_depth completed
... Wrote DA output summary NetCDF /data/projects/project_2022_2023/results/grids/da_output_grids.nc (18 step(s))
... Compact retention: deleted ... grid artifact file(s), freed ... MB
... Setup cleanup succeeded: deleted ... file(s), freed ... MB
... Project processing complete: /data/projects/project_2022_2023 (wall-clock 866.7 s, ~0.24 h)
```

### 2. Check that plots and outputs are being created (path-based check)

After a successful run, these paths should exist:

- `/data/rofental/projects/project_2022_2023/plots/perf/project_perf.png`
- `/data/rofental/projects/project_2022_2023/plots/results/fraction_timeseries.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_latschbloder_snow_depth_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_proviantdepot_swe_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/point_scf_roi_envelope.csv`
- `/data/rofental/projects/project_2022_2023/point_wet_snow_roi_envelope.csv`

Reference snippet (typical result files visible after a successful run):

```text
plots/perf/
  project_perf.png
  project_perf_metrics.csv

plots/results/
  fraction_timeseries.png
  lc_mask_report.csv
  point_scf_roi_envelope.csv
  point_wet_snow_roi_envelope.csv
  setup_results_point_latschbloder_snow_depth_2022_2023.png
  setup_results_point_latschbloder_swe_2022_2023.png
  setup_results_point_proviantdepot_snow_depth_2022_2023.png
  setup_results_point_proviantdepot_swe_2022_2023.png
```

Visual run-success teaser (performance plot):

![Project performance plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/project_perf.png)

For this quick check, confirm:

- the file exists and opens,
- the plot is populated (not empty/corrupt),
- runtime/resource traces are present (detailed interpretation comes later).

Visual run-success teaser (main results overview plot):

![Fraction time series (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/fraction_timeseries.png)

For this quick check, confirm:

- observation markers are present,
- the plot spans the configured tutorial season,
- the plot renders correctly (axes/legend visible).

Detailed interpretation of these plots is covered in [7. Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}).

### 3. Check the compact DA grid output (path-based check)

Expected output file:

- `/data/rofental/projects/project_2022_2023/results/grids/da_output_grids.nc`

**Output export configuration (project YAML)**  
File path: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`  
Relevant keys: `data_assimilation.output.retention`, `data_assimilation.output.grids.*`, `data_assimilation.output.grids.variables[*]`

{: .warning }
> Do not treat the existence of files alone as proof of a healthy run. Always check the
> log for errors and warnings and inspect DA diagnostics in the next chapter.

{: .references }
> - [Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}) for plot/diagnostic interpretation
> - [Advanced Performance]({{ site.baseurl }}{% link advanced/performance.md %}) for performance tuning after the baseline tutorial run

<details markdown="block">
  <summary>Suggested quick-check sequence during a run</summary>

1. Tail the project log
2. Confirm step progression and assimilation messages
3. Check whether plots start appearing under `plots/`
4. Check the final `Project processing complete` message
</details>

---

## Lower-level commands (how the pipeline relates to manual execution)

The tutorial recommends the project pipeline for normal operation, but it is still useful to understand the main lower-level building blocks it wraps.

Examples (step-level mechanics):

<details markdown="block">
  <summary>Optional low-level command examples (for learning/debugging)</summary>

### Prior forcing ensemble generation (step-level)

```bash
python -m openamundsen_da.core.prior_forcing \
  --input-meteo-dir /data/rofental/projects/project_2022_2023/meteo \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --step-dir /data/rofental/projects/project_2022_2023/steps/step_00_init \
  --overwrite
```

### Ensemble launch (step-level)

```bash
python -m openamundsen_da.core.launch \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --setup-dir /data/rofental \
  --step-dir /data/rofental/projects/project_2022_2023/steps/step_00_init \
  --ensemble prior \
  --max-workers 8 \
  --overwrite
```

</details>

Why this section is useful:

- it helps you debug specific steps,
- it explains what the pipeline is automating,
- it makes the framework behavior less "black box".

A full manual DA cycle (including all diagnostics, assimilation, resampling, and
rejuvenation internals) is possible but intentionally not the primary tutorial path.
The project pipeline is the recommended operational workflow.

Treat lower-level commands as learning/debugging tools; treat the project pipeline as the normal production workflow.

{: .references }
> - [Command-Line Interface]({{ site.baseurl }}{% link guides/cli.md %}) for the full command catalog
> - [Package Structure]({{ site.baseurl }}{% link reference/package-structure.md %}) if you want to map CLI commands to internal modules

---

## Rerun patterns during tutorial work (important)

You will often iterate while learning:

- edit `assimilation_events`,
- change observation mappings/classes,
- rerun preprocessing,
- rerun the project.

Recommended rerun sequence after changes:

1. `project_skeleton --overwrite`
2. `oa-da-scf --overwrite`
3. `oa-da-wetsnow-project --overwrite`
4. `openamundsen_da.pipeline.project --overwrite`

If you only change plotting or documentation, a full rerun is usually unnecessary.

---

## What to check before moving on

{: .checks }
> Confirm these outputs and diagnostics before opening the results chapter.

Before continuing, verify:

- `project_2022_2023.log` ends with a completion message,
- `plots/perf/project_perf.png` exists,
- `plots/results/fraction_timeseries.png` exists,
- `results/grids/da_output_grids.nc` exists.

These indicate that preprocessing, the DA run, plotting, and compact output export all
completed successfully.

---

## Next step

{: .references }
> Continue to the results and diagnostics chapter after the project run completed successfully.

Continue with [7. Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %})
to inspect:

- DA weights and ESS behavior,
- performance plots and metrics,
- result time series,
- CSV summaries and compact DA grid outputs.
