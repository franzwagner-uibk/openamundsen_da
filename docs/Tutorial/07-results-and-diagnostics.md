---
layout: default
title: 6. Results and Diagnostics
parent: How to Use
nav_order: 6
permalink: /tutorial/results-and-diagnostics/
---

# 6. Results and Diagnostics

This chapter is the review pass after a completed project run. The recommended order is
simple: confirm that the run finished cleanly, inspect the assimilation diagnostics, and
only then interpret plots and compact output grids.

Most tutorial outputs live under:

- `/data/rofental/projects/project_2022_2023/`

The most important locations are:

- `project_2022_2023.log`
- `plots/perf/`
- `plots/assim/`
- `plots/results/`
- `results/grids/da_output_grids.nc`

## 1. Read the log first

Before interpreting plots, confirm that the run actually completed cleanly by checking the
end of `project_2022_2023.log`.

What to look for:

- `Project processing complete`
- data assimilation variable processing messages (`scf`, `wet_snow`)
- plot tasks completed
- data assimilation output summary NetCDF writing (`da_output_grids.nc`)
- cleanup messages (`Setup cleanup succeeded`; compact-retention deletion appears only in compact mode)

<details markdown="block">
  <summary>If the log is not clean (important troubleshooting note)</summary>

If the log is not clean, treat downstream plots and tables as potentially incomplete or misleading.

Warning signs:

- repeated missing observation messages
- failures in a specific step
- plot task errors
- no data assimilation output export message

[Advanced Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) (follow-up when the log shows errors)

</details>

---

## 2. Performance diagnostics (`plots/perf`)

The performance plots help you understand runtime distribution and resource behavior.

Files to inspect:

- `plots/perf/project_perf.png`
- `plots/perf/project_perf_metrics.csv`

Reference structure snippet (`plots/perf`)

```text
/data/rofental/projects/project_2022_2023/plots/perf/
  project_perf.png
  project_perf_metrics.csv
```


{: .checks }
> How to use these outputs:
>
> - compare runtime across tutorial reruns
> - estimate cost of increasing `ensemble_size`
> - estimate cost of changing resolution (`100 m` vs coarser)
> - identify unexpectedly slow stages (I/O, plotting, step-level hotspots)


Reference CSV snippet (performance metrics)

File path: `/data/rofental/projects/project_2022_2023/plots/perf/project_perf_metrics.csv`

| timestamp | cpu_total_pct | mem_used_pct | mem_used_gb | mem_total_gb |
| --- | --- | --- | --- | --- |
| 2026-02-21T21:28:14 | 0.00 | 4.10 | 1.01 | 24.45 |
| 2026-02-21T21:28:19 | 40.90 | 14.10 | 3.44 | 24.45 |
| 2026-02-21T21:28:24 | 53.30 | 16.00 | 3.91 | 24.45 |
| 2026-02-21T21:28:29 | 52.50 | 17.70 | 4.32 | 24.45 |
| 2026-02-21T21:28:34 | 61.90 | 19.40 | 4.74 | 24.45 |

Plot file to open:

- `/data/rofental/projects/project_2022_2023/plots/perf/project_perf.png`

Reference plot (tutorial baseline, `ens=10`):

![Project performance plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/project_perf.png)

_`project_perf.png` from the Rofental tutorial reference run (`100 m`, `ensemble_size=10`)._

What to read in the plot:

- **CPU utilization panel/curve**: look for sustained utilization during step processing and drops during lighter phases (I/O, orchestration).
- **Memory usage panel/curve**: check whether memory stays within a stable range and does not continuously climb (which can indicate a leak or runaway buffering).
- **Timing structure**: repeated patterns often correspond to repeated step execution.

{: .references }
> - [Advanced Performance]({{ site.baseurl }}{% link advanced/performance.md %}) (deeper interpretation of runtime behavior)

---

## 3. Assimilation diagnostics (`plots/assim`)

These plots are the core data assimilation diagnostics.

Typical outputs:

- `plots/assim/weights/step_*_weights.png`
- `plots/assim/ess/setup_ess_timeline_*.png`

What they show:

- **weights**: how strongly observations favor some particles over others
- **ESS (effective sample size)**: particle degeneracy indicator

Reference structure snippet (`plots/assim`, typical files)

```text
/data/rofental/projects/project_2022_2023/plots/assim/
  ess/
    setup_ess_timeline_2022_2023.png
  weights/
    step_02_weights.png
    step_03_weights.png
    ...
```


{: .checks }
> Interpretation guidelines:
>
> - ESS near ensemble size for many events:
>   - observations have weak discrimination or high observation error
> - ESS very low (near 1) frequently:
>   - strong degeneracy, aggressive resampling likely
>   - possibly too-small observation error (`obs_sigma`) or too-strong mismatch
> - abrupt differences between SCF and wet-snow events:
>   - normal and expected (different variables, coverage, and information content)

ESS is a diagnostic, not a simple "good/bad" score. Interpret it together with weights, variable type, and observation coverage.

Reference CSV snippet (weights for one wet-snow event)

File path: `/data/rofental/projects/project_2022_2023/steps/step_02_20230309-20230511/assim/weights_wet_snow_20230511.csv`

| member_id | wet_snow_model | wet_snow_obs | residual | sigma | log_weight | weight |
| --- | --- | --- | --- | --- | --- | --- |
| member_001 | 0.71 | 0.89 | 0.18 | 0.10 | -0.17 | 0.04 |
| member_002 | 0.86 | 0.89 | 0.03 | 0.10 | 1.33 | 0.19 |
| member_003 | 0.64 | 0.89 | 0.25 | 0.10 | -1.67 | 0.01 |
| member_004 | 0.75 | 0.89 | 0.14 | 0.10 | 0.45 | 0.08 |
| member_005 | 0.70 | 0.89 | 0.19 | 0.10 | -0.39 | 0.03 |

Plot files to open:

- `/data/rofental/projects/project_2022_2023/plots/assim/ess/setup_ess_timeline_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/plots/assim/weights/step_02_weights.png`

Reference ESS plot (tutorial baseline):

![ESS timeline plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/setup_ess_timeline_2022_2023.png)

_ESS timeline (`setup_ess_timeline_2022_2023.png`) from the tutorial reference run._

What to read in the ESS plot:

- each point corresponds to one assimilation event,
- lower ESS means stronger weight concentration (more degeneracy),
- differences between SCF and wet-snow events are expected because the observation types have different information content and spatial support.

Reference weights plot (example step):

![Weights plot for one assimilation step (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/step_02_weights.png)

_Weights plot for `step_02` (wet-snow event around `2023-05-11`)._

What to read in the weights plot:

- a **flat** distribution means the observation did not strongly discriminate particles,
- a **peaked** distribution means a few particles explain the observation much better,
- very strong peaks often coincide with low ESS and potential resampling pressure.

Exact weights differ between runs because the ensemble is stochastic. Focus on the structure (spread/concentration), not exact numeric values.

{: .references }
> - [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) (likelihood, resampling, rejuvenation)
> - [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) (theory and terminology)

---

## 4. Result plots (`plots/results`)

This directory contains the main user-facing validation and data assimilation result figures for the setup.

Typical files include:

- `result_overview.png` (SCF, wet-snow, ROI mean SWE, and ROI mean snow depth over time)
- station snow depth plots
- station SWE plots
- ROI envelope exports and land-cover masking report

Reference structure snippet (`plots/results`, typical files)

```text
/data/rofental/projects/project_2022_2023/plots/results/
  result_overview.png
  lc_mask_report.csv
  setup_results_point_latschbloder_snow_depth_2022_2023.png
  setup_results_point_latschbloder_swe_2022_2023.png
  setup_results_point_proviantdepot_snow_depth_2022_2023.png
  setup_results_point_proviantdepot_swe_2022_2023.png
```


### Result overview (high-level data assimilation behavior)

`result_overview.png` is often the fastest way to inspect:

- observation dates actually used,
- model-vs-observation fraction behavior,
- ROI mean SWE and snow-depth evolution relative to the open loop,
- whether SCF and wet-snow observations are present where expected.

The ROI SWE and snow-depth panels use the full ROI footprint rather than the land-cover-masked ROI used for SCF and wet-snow summaries.

What to inspect:

- are observation markers present at the configured data assimilation dates?
- do SCF and wet-snow events appear in the expected seasonal phases?
- are there obvious missing events or suspicious gaps?

### Station plots (snow depth / SWE)

These plots support validation and interpretation of model behavior at observation stations.

What to inspect:

- overall seasonal timing (accumulation / melt timing),
- amplitude differences (too much / too little snow),
- whether data assimilation shifts the ensemble envelope relative to the open loop,
- consistency across stations (important for tutorial interpretation).

In this tutorial setup, station SWE observations are expected in **mm** (see project config comment).
If a curve appears near zero against model SWE, check units first.

Land-cover masking affects how much of the ROI contributes to SCF/wet-snow summaries and
fractions. This report is useful here because it explains the masking context behind the
result plots and ROI envelope values shown in this chapter.

Reference CSV snippet (land-cover masking report)

File path: `/data/rofental/projects/project_2022_2023/plots/results/lc_mask_report.csv`

| class_code | class_name | cells | area_km2 | percent_of_roi |
| --- | --- | --- | --- | --- |
| 2 | ice | 3257 | 32.57 | 32.82 |
| 3 | water | 15 | 0.15 | 0.15 |
| 10 | mixed forest | 48 | 0.48 | 0.48 |
| 8,9,10,11,12 | forest | 69 | 0.69 | 0.70 |
| total | total | 3346 | 33.46 | 33.72 |

How to use this table:

- confirm that excluded/retained classes look plausible for the tutorial ROI
- check whether `percent_of_roi` suggests over-masking (unexpectedly little usable area)
- use it as context when SCF/wet-snow fractions look unexpectedly low/high

Recommended plot files to inspect (Rofental tutorial run):

- `/data/rofental/projects/project_2022_2023/plots/results/result_overview.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_latschbloder_snow_depth_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_proviantdepot_snow_depth_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_latschbloder_swe_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_proviantdepot_swe_2022_2023.png`

<details markdown="block">
  <summary>Suggested plot review order (practical)</summary>

1. `result_overview.png` (observation timing and availability)
2. ESS / weights plots (data assimilation behavior)
3. Station snow depth plots
4. Station SWE plots
5. NetCDF/grid outputs for spatial interpretation
</details>

### Reference plots (tutorial baseline)

Result overview:

![Result overview (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/result_overview.png)

_`result_overview.png`: check observation dates, SCF/wet-snow event timing, ROI mean SWE / snow-depth behavior, and gross model-vs-observation behavior._

What to read in this plot:

- whether observation markers exist at the configured data assimilation dates,
- whether SCF events cluster in snow-cover relevant periods and wet-snow events in melt-season periods,
- whether ROI mean SWE and snow depth shift away from or back toward the open loop during the season,
- whether data assimilation moves the ensemble envelope in the expected direction relative to the open loop.

Station snow depth example (`latschbloder`):

![Latschbloder snow depth plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/setup_results_point_latschbloder_snow_depth_2022_2023.png)

_Snow depth comparison at `latschbloder` (open loop + ensemble + observations)._

What to read in this plot:

- timing of snow accumulation and melt,
- whether observed values stay within (or near) the ensemble envelope,
- whether data assimilation visibly shifts the ensemble relative to the open loop around observation periods.

Station SWE example (`proviantdepot`):

![Proviantdepot SWE plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/setup_results_point_proviantdepot_swe_2022_2023.png)

_SWE comparison at `proviantdepot` (remember: station SWE observations are expected in **mm** in this tutorial setup)._

What to read in this plot:

- unit consistency first (obs in **mm** in this tutorial setup),
- amplitude mismatch (systematic bias) vs timing mismatch (phase error),
- whether data assimilation corrections remain small/local or systematically shift the trajectory.

<details markdown="block">
  <summary>More station reference plots (tutorial baseline)</summary>

`proviantdepot` snow depth:

![Proviantdepot snow depth plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/setup_results_point_proviantdepot_snow_depth_2022_2023.png)

`latschbloder` SWE:

![Latschbloder SWE plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/setup_results_point_latschbloder_swe_2022_2023.png)

</details>

{: .references }
> - [Observation Processing]({{ site.baseurl }}{% link guides/observations.md %}) (SCF / wet-snow preprocessing context)
> - [Workflow]({{ site.baseurl }}{% link workflow.md %}) (where these plots fit in the data assimilation workflow)

<a id="da-output-summary-netcdf"></a>
## 5. data assimilation output summary NetCDF (`da_output_grids.nc`)

The tutorial setup writes a data assimilation output summary NetCDF:

- `results/grids/da_output_grids.nc`

**Output configuration (project YAML)**  
File path: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`  
Relevant keys: `data_assimilation.output.retention`, `data_assimilation.output.grids.format`, `data_assimilation.output.grids.variables[*]`

This file is designed for:

- post-processing,
- visualization,
- comparison between runs,
- exporting selected variables in one merged file.

In the current tutorial configuration, `data_assimilation.output.retention: full` is enabled, so this summary NetCDF is written and the heavier member-grid artifacts are retained as well.

Reference output file path (data assimilation summary NetCDF):

- `/data/rofental/projects/project_2022_2023/results/grids/da_output_grids.nc`


Optional variable/dimension inspection (Python in the container).

**🟢 Run this command:**

```bash
python - <<'PY'
import xarray as xr
ds = xr.open_dataset("/data/rofental/projects/project_2022_2023/results/grids/da_output_grids.nc")
print(ds)
print("\nVariables:")
for name in ds.data_vars:
    print("-", name, ds[name].dims)
PY
```


{: .checks }
> What to expect conceptually in the summary NetCDF:
>
> - open-loop baseline fields
> - ensemble mean / spread fields
> - increments (`ens_mean - open_loop`) for configured variables/aggregations

Dimension names in the inspected NetCDF (for example `time1`, `time2`, `snow_layer`, `nbnd`) are typically inherited from the underlying model outputs. Configure **which variables/metrics** are exported in the project YAML under `data_assimilation.output.grids.variables[*]`; see [5. Running the Model]({{ site.baseurl }}{% link Tutorial/06-running-the-project.md %}) for the output-grid configuration note.

### Raster output

{: .checks }
> Key raster comparison from `results/grids/da_output_grids.nc`:
> - `open_loop_snowdepth_daily`
> - `ens_mean_snowdepth_daily`
>
> Date shown in the reference map: **2023-06-02**

![Snow depth map pair for 2023-06-02 (open loop vs ensemble mean)]({{ site.baseurl }}/assets/images/tutorial/rofental_tutorial_snow_depth_2023_06_02.png)

_`open_loop_snowdepth_daily` (left) vs `ens_mean_snowdepth_daily` (middle) on **2023-06-02**. Use the same color classes for both maps for direct interpretation._

![Snow depth increment map for 2023-06-02 (ensemble mean minus open loop)]({{ site.baseurl }}/assets/images/tutorial/rofental_tutorial_snow_depth_increment_2023_06_02.png)

_`increment_snowdepth_daily` on **2023-06-02** (`ens_mean_snowdepth_daily - open_loop_snowdepth_daily`)._

{: .checks }
> Note on raster workflow:
> - all output grids are stored in one NetCDF file: `results/grids/da_output_grids.nc` (see [5. data assimilation output summary NetCDF (`da_output_grids.nc`)](#da-output-summary-netcdf))
> - extract the layers/time slices you need in the GIS tool of your choice before styling or export

Reference snippet (NetCDF inspection, tutorial reference run):

```text
dims: {'time2': 272, 'snow_layer': 3, 'y': 150, 'x': 160, 'time1': 273, 'nbnd': 2}
vars:
  ens_max_liquid_water_content
  ens_max_snowdepth_daily
  ens_max_swe_daily
  ens_mean_liquid_water_content
  ens_mean_snowdepth_daily
  ens_mean_swe_daily
  ens_min_liquid_water_content
  ens_min_snowdepth_daily
  ens_min_swe_daily
  ens_std_liquid_water_content
  ens_std_snowdepth_daily
  ens_std_swe_daily
  increment_liquid_water_content
  increment_snowdepth_daily
  increment_swe_daily
  open_loop_liquid_water_content
  open_loop_snowdepth_daily
  open_loop_swe_daily
```

Use `results/grids/da_output_grids.nc` in a GIS software of your choice and visualize raster output.

Recommended map date(s): choose one date with active snow cover and one date near melt season.
Use the same date across `open_loop`, `ens_mean`, and `increment` maps.

### data assimilation increment map

The tutorial also includes a reference increment map above (`increment_snowdepth_daily`, date
**2023-06-02**). For additional diagnostics, export one or two extra increment dates from
`da_output_grids.nc` and compare against the same-date open-loop/ensemble-mean maps.

{: .references }
> - [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) (data assimilation output variable selection and metrics)
> - [Project Structure]({{ site.baseurl }}{% link project-structure.md %}) (where project outputs live)
> - [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) (how to interpret increments conceptually)

---

## 6. ROI envelopes and summary CSVs

The project root contains setup-level envelope time series:

- `point_scf_roi_envelope.csv`
- `point_wet_snow_roi_envelope.csv`

These summarize the ensemble spread over the ROI (mean/min/max and sample count).

These CSVs are lightweight outputs that are ideal for quick comparisons between runs without loading NetCDF files.

Reference file paths for ROI envelope outputs:

- `/data/rofental/projects/project_2022_2023/point_scf_roi_envelope.csv`
- `/data/rofental/projects/project_2022_2023/point_wet_snow_roi_envelope.csv`


{: .checks }
> Why these CSVs are useful:
>
> - quick numeric QA without opening plots
> - useful for external plotting notebooks or reports
> - easy comparison across experimental runs

Reference CSV snippet (SCF ROI envelope)

File path: `/data/rofental/projects/project_2022_2023/point_scf_roi_envelope.csv`

| date | value_mean | value_min | value_max | n |
| --- | --- | --- | --- | --- |
| 2022-10-01 | 0.00 | 0.00 | 0.00 | 11 |
| 2022-10-02 | 0.12 | 0.00 | 0.50 | 11 |
| 2022-10-03 | 0.08 | 0.00 | 0.40 | 11 |
| 2022-10-04 | 0.09 | 0.00 | 0.42 | 11 |

Reference CSV snippet (wet-snow ROI envelope)

File path: `/data/rofental/projects/project_2022_2023/point_wet_snow_roi_envelope.csv`

| date | value_mean | value_min | value_max | n |
| --- | --- | --- | --- | --- |
| 2022-10-01 | 0.00 | 0.00 | 0.00 | 11 |
| 2022-10-02 | 0.59 | 0.45 | 0.77 | 11 |
| 2022-10-03 | 0.04 | 0.00 | 0.14 | 11 |
| 2022-10-04 | 0.10 | 0.00 | 0.36 | 11 |

Interpretation:

- `value_mean`: ensemble mean over `open_loop + ensemble members`
- `value_min` / `value_max`: spread envelope
- `n`: number of contributing trajectories (for tutorial baseline: `10` members + `1` open loop = `11`)

---

## 7. Quick data assimilation sanity checklist (practical review routine)

{: .checks }
> Use this checklist as a quick review before trusting or comparing results.
>
> Use this checklist after each tutorial run:
>
> 1. Log ends with `Project processing complete`
> 2. `plots/perf/project_perf.png` exists
> 3. `plots/assim/` contains ESS and weight plots
> 4. `plots/results/result_overview.png` exists
> 5. Station plots exist (snow depth and SWE)
> 6. `point_scf_roi_envelope.csv` and `point_wet_snow_roi_envelope.csv` exist
> 7. `results/grids/da_output_grids.nc` exists and opens
>
> If one of these fails, check the log before changing configuration.

---

## 8. Season cleanup (optional)

openAMUDNSEN-DA contains a module that cleans up heavy files that are used within the data assimilation workflow and not needed anymore after running a project. The cleanup is wired into the project pipeline and activated by default.

{: .checks }
> Automatic cleanup is enabled by default via `data_assimilation.restart.cleanup_after_setup: true`.
> Use manual cleanup if you disabled automatic cleanup or if older seasons still contain state files.

Clean one season (`project_2022_2023`):

**🟢 Run this command:**

```bash
oa-da-clean-project \
  --setup-dir /data/rofental \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --log-level INFO
```

Clean all seasons under the same setup:

**🟢 Run this command:**

```bash
oa-da-clean-project \
  --setup-dir /data/rofental \
  --all-projects \
  --log-level INFO
```

What is removed:

- restart state pickle files under member `results/` directories,
- sub-domain workspace artifacts (if present).

{: .references }
> - [Workflow]({{ site.baseurl }}{% link workflow.md %}) (state cleanup behavior and project lifecycle)
