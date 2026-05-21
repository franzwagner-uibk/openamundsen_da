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
result products. It also always runs the scientific benchmarking stage at the end of the
project.

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
    ensemble_size: 30

  assimilation_events:
    - date: "2022-11-17"
      variable: station_hs
    - date: "2022-12-07"
      variable: station_hs
    - date: "2023-01-01"
      variable: station_hs
    - date: "2023-01-31"
      variable: station_hs
    - date: "2023-02-21"
      variable: station_hs
    - date: "2023-03-24"
      variable: wet_snow_line
      product: WETSNOW
    - date: "2023-04-26"
      variable: scf
      product: SNOWCOVER
    - date: "2023-05-26"
      variable: scf
      product: SNOWCOVER

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

This snippet shows the three project settings that most visibly change runtime and outputs in the tutorial: ensemble size, number/timing of events, and which grid-summary variables/metrics are exported. Generated DA-event maps need `analysis_mean` and `analysis_increment` for `snowdepth_daily`.

### Benchmark outputs written by the project run

Every `oa-da-project` run now writes benchmark artifacts under:

- `/data/rofental/projects/project_2022_2023/results/benchmark/`
- `/data/rofental/projects/project_2022_2023/results/plots/assim/scores/`

The benchmark result directory now contains:

- long-form case and score tables under `cases/` and `scores/`
- two compact curated tables under `tables/`
- a markdown summary plus manifest
- one headline DA-skill plot under `results/plots/assim/scores/performance_scores.png`

These files keep both a propagated whole-project benchmark view and an assimilation-date update-skill view. The headline plot itself is narrower: it shows only DA-date `prior` and weighted `posterior` skill, and adds a third station-only `zSkill` panel when sigma-aware station scores are available, while the whole-project propagated summary stays in `project_summary.csv`. Results are split into:

- `assimilation_fit`: the exact family/date pair was assimilated
- `semi_independent`: the exact family/date pair was not assimilated, but a same-variable or sister-station assimilation has already happened by that date
- `independent`: no same-variable assimilation has happened yet by that date and no active sister-station linkage applies

The benchmark figure is intentionally different from the main result overview. It focuses on assimilation-date DA performance (`CRPSS`, `NER`, and station-only `zSkill`) rather than state evolution, while the established result plots remain the place for ensemble spread and observation-vs-model context.

This is an observation-based benchmarking layer. It strengthens scientific score reporting, but it does not replace future holdout, LOOCV, or OSSE validation workflows.

### How to configure data assimilation grid output content and dimensions (important)

The data assimilation output summary NetCDF (`results/grids/da_output_grids.nc`) is configured in the same
`data_assimilation.output.grids` block.

What you can configure directly here:

- `variables[*].var` / `name`: which model grid variables are exported into the data assimilation summary
- `variables[*].metrics`: which summary metrics are written for each variable (`open_loop`, `ens_mean`, `ens_std`, `ens_min`, `ens_max`, `increment`, `analysis_mean`, `analysis_increment`)
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

- `/data/rofental/projects/project_2022_2023/results/plots/perf/project_perf.png`
- `/data/rofental/projects/project_2022_2023/results/plots/assim/ess/setup_ess_timeline_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/results/plots/assim/scores/performance_scores.png`
- `/data/rofental/projects/project_2022_2023/results/plots/assim/weights/setup_weights_overview_2022_2023.png`
  - for projects with many assimilation dates, continuation pages are written as `setup_weights_overview_2022_2023_page_02.png`, `..._page_03.png`, etc.
- `/data/rofental/projects/project_2022_2023/results/plots/assim/weights/DA_04_weights.png`
- `/data/rofental/projects/project_2022_2023/results/plots/results/result_overview.png`
- `/data/rofental/projects/project_2022_2023/results/plots/points/setup_results_point_latschbloder_snow_depth_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/results/plots/points/setup_results_point_proviantdepot_swe_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/results/misc/point_scf_roi_envelope.csv`
- `/data/rofental/projects/project_2022_2023/results/misc/point_wet_snow_roi_envelope.csv`
- `/data/rofental/projects/project_2022_2023/results/grids/da_output_grids.nc`
- `/data/rofental/projects/project_2022_2023/results/maps/`
- `/data/rofental/projects/project_2022_2023/results/reports/project_report.pdf`

Typical result files after a successful run:

```text
results/plots/assim/ess/
  setup_ess_timeline_2022_2023.png

results/plots/assim/scores/
  performance_scores.png

results/plots/assim/weights/
  DA_01_weights.png
  ...
  DA_10_weights.png
  setup_weights_overview_2022_2023.png
  setup_weights_overview_2022_2023_page_02.png   # only when the overview spans multiple A4-length pages

results/plots/perf/
  project_perf.png
  project_perf_metrics.csv

results/plots/results/
  result_overview.png
  result_overview_custom.png

results/plots/points/
  setup_results_point_latschbloder_snow_depth_2022_2023.png
  setup_results_point_latschbloder_swe_2022_2023.png
  setup_results_point_proviantdepot_snow_depth_2022_2023.png
  setup_results_point_proviantdepot_swe_2022_2023.png

results/misc/
  lc_mask_report.csv
  point_scf_roi_envelope.csv
  point_wet_snow_roi_envelope.csv

results/maps/
  setup_overview.png
  da_events/
    da_1.png
    ...
    da_10.png

results/reports/
  project_report.pdf                  # attempted at the end of oa-da-project
```

The report PDF is a best-effort final artifact. Its first page contains the setup/project summary plus a bottom `Content` table with page numbers first and section names second. It then collects the curated overview plots, setup map, setup weights overview pages, station snow-depth point plots, `performance_scores.png`, `project_perf.png`, and generated DA-event maps in temporal order. If report prerequisites are missing, `oa-da-project` logs the missing paths and a manual `python -m openamundsen_da.methods.viz.reports --project-dir ...` rerun command, but the completed model run remains successful.

![Project performance plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens15/project_perf.png)

![Result overview (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens15/result_overview.png)

Detailed interpretation of these plots is covered in [7. Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}).

Before continuing, verify that the project log ends with a completion message and that `results/plots/perf/project_perf.png`, `results/plots/assim/weights/setup_weights_overview_2022_2023.png`, `results/plots/results/result_overview.png`, and `results/grids/da_output_grids.nc` exist.

If you later change only visualization code or `plots.yml`, you can rerender the finished project plots without rerunning the DA pipeline:

```bash
oa-da-plot-project-plots --project-dir /data/projects/project_2022_2023
```

And you can rerender the full project-map catalog separately with:

```bash
oa-da-plot-project-maps --project-dir /data/projects/project_2022_2023
```
