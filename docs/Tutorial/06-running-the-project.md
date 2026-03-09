---
layout: default
title: 5. Running the Model
parent: How to Use
nav_order: 5
permalink: /tutorial/running-the-project/
---

# 5. Running the Model

At this point the observation preprocessing is complete. The remaining task is to run the
project pipeline on the prepared setup and project configuration.

The pipeline executes the step-wise ensemble workflow, reads the matching per-step
observation CSVs, performs the configured assimilation updates, and writes diagnostics and
result products.

From this point on, the tutorial assumes you are inside the running tutorial container
shell at `/data/rofental`.

## Run the full project pipeline (recommended workflow)

Run this inside the tutorial container shell. `--setup-dir` points to the openAMUNDSEN setup, `--project-dir` selects the data assimilation project, `--max-workers` caps pipeline parallelism, `--overwrite` allows reruns, and `--log-level INFO` keeps progress visible in the project log.

This is the main tutorial command for the full data assimilation run.

**Configuration files used by this run**  
Setup config: `/data/rofental/rofental.yml` (openAMUNDSEN domain/model configuration)  
Project config: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml` (data assimilation events, obs mapping, likelihood/resampling, outputs)

**🟢 Run this command:**

```bash
oa-da-project \
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
    - date: "2023-01-01"
      variable: scf
      product: SNOWCOVER
    - date: "2023-03-09"
      variable: scf
      product: SNOWCOVER
    - date: "2023-05-11"
      variable: wet_snow
      product: WETSNOW
    - date: "2023-05-26"
      variable: scf
      product: SNOWCOVER
    - date: "2023-06-16"
      variable: wet_snow
      product: WETSNOW

  output:
    retention: full
    grids:
      format: netcdf
      variables:
        - var: snowdepth_daily
          metrics: [open_loop, ens_mean, ens_std, ens_min, ens_max, increment]
        - var: swe_daily
          metrics: [open_loop, ens_mean, ens_std, ens_min, ens_max, increment]
```

This snippet shows the three project settings that most visibly change runtime and outputs in the tutorial: ensemble size, number/timing of events, and which grid-summary variables/metrics are exported.

### How to configure data assimilation grid output content and dimensions (important)

The data assimilation output summary NetCDF (`results/grids/da_output_grids.nc`) is configured in the same
`data_assimilation.output.grids` block.

What you can configure directly here:

- `variables[*].var` / `name`: which model grid variables are exported into the data assimilation summary
- `variables[*].metrics`: which summary metrics are written for each variable (`open_loop`, `ens_mean`, `ens_std`, `ens_min`, `ens_max`, `increment`)
- `format`, `compress`, `retention`

How to interpret the `dims` in the NetCDF inspection output:

- `x`, `y` sizes come from the setup domain + resolution (for the tutorial: the Rofental `100 m` grid)
- `time*` dimensions come from the underlying model NetCDF outputs and the project period
- `snow_layer` appears for layer-resolved variables (e.g. liquid water content by snow layer)
- `nbnd=2` is the usual time-bounds dimension (start/end bounds)

Note on `grids.dims: [x, y, time]`:

- this is the intended standard dimension order for exported data assimilation grids in the config
- the current data assimilation summary writer still preserves source variable dimensions from the underlying model NetCDF files, so the inspected dataset may contain `time1`/`time2`, `snow_layer`, and `nbnd`

---

## Expected outputs after the run

After a successful run, these paths should exist:

- `/data/rofental/projects/project_2022_2023/plots/perf/project_perf.png`
- `/data/rofental/projects/project_2022_2023/plots/results/fraction_timeseries.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_latschbloder_snow_depth_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_proviantdepot_swe_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/point_scf_roi_envelope.csv`
- `/data/rofental/projects/project_2022_2023/point_wet_snow_roi_envelope.csv`

Typical result files after a successful run:

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

![Project performance plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/project_perf.png)

![Fraction time series (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/fraction_timeseries.png)

Detailed interpretation of these plots is covered in [7. Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}).

Before continuing, verify that the project log ends with a completion message and that `plots/perf/project_perf.png`, `plots/results/fraction_timeseries.png`, and `results/grids/da_output_grids.nc` exist.
