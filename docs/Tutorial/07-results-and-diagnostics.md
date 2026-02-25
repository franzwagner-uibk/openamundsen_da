---
layout: default
title: 7. Results and diagnostics
parent: Tutorial
nav_order: 7
permalink: /tutorial/results-and-diagnostics/
---

# 7. Results and diagnostics

This chapter explains how to inspect and interpret the outputs of the tutorial DA run.

The goal is not only to find files, but to understand:

- whether the run completed correctly,
- how the DA behaved (weights, ESS, resampling pressure),
- what the plots and tables represent,
- and where to look first when something seems wrong.

Review order matters: check the log first, then DA diagnostics, then interpret the plots and grids.

All command blocks below are executed **inside the running tutorial container shell** started in [2. Dependencies]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %}).

---

## Step-by-step flow on this page

{: .step }
> Inspect outputs in this order so you go from logs and diagnostics to plots and final products.

Use this page as a review routine after a completed run:

1. confirm completion in the project log
2. inspect performance diagnostics (`plots/perf`)
3. inspect DA diagnostics (`plots/assim`)
4. inspect result plots (`plots/results`)
5. inspect summary CSVs / ROI envelopes
6. inspect the DA summary NetCDF (`da_output_grids.nc`)

Do not start with plots alone. Always check the log first, because incomplete runs can still leave partial plots/files behind.

Many commands in this chapter are optional inspection helpers. The primary guidance is the **review order + file paths + reference snippets**.

---

## Project output structure (what to inspect first)

Most tutorial outputs live under:

- `rofental/projects/project_2022_2023/`

Key locations:

- `project_2022_2023.log` - full run log
- `plots/perf/` - runtime/performance plots and metrics
- `plots/assim/` - DA diagnostics (weights, ESS)
- `plots/results/` - result plots (fractions, station plots, envelopes)
- `results/grids/da_output_grids.nc` - DA output summary NetCDF
- `point_scf_roi_envelope.csv`, `point_wet_snow_roi_envelope.csv` - ROI time-series envelopes

Quick directory overview (optional helper):

Inspect the project output folder structure (optional helper for orientation).

```bash
find /data/rofental/projects/project_2022_2023 -maxdepth 3 -type d | sort
```

<details markdown="block">
  <summary>Why this directory overview is useful</summary>

It helps new users understand which outputs are:

- run logs,
- diagnostics,
- plotting products,
- project-level grid/result exports,
- and step/member internals.
</details>

---

## 1. Read the log first (always)

Before interpreting plots, confirm that the run actually completed cleanly.

Inspect the end of the project log before interpreting plots and tables.

```bash
tail -n 120 /data/rofental/projects/project_2022_2023/project_2022_2023.log
```


{: .checks }
> What to look for in a successful log tail:
>
> - `Project processing complete`
> - DA variable processing messages (`scf`, `wet_snow`)
> - plot tasks completed
> - DA output summary NetCDF writing (`da_output_grids.nc`)
> - cleanup messages (`Setup cleanup succeeded`; compact-retention deletion appears only in compact mode)

Reference snippet (successful log tail excerpt):

```text
2026-02-21 ... Plot task setup_ess_timeline completed
2026-02-21 ... Plot task setup_results_swe completed
2026-02-21 ... Plot task setup_results_snow_depth completed
2026-02-24 ... Wrote DA output summary NetCDF /data/projects/project_2022_2023/results/grids/da_output_grids.nc (6 step(s))
2026-02-24 ... Setup cleanup succeeded: deleted 66/66 file(s), freed 345.9 MB (patterns=model_state.pickle.gz)
2026-02-24 ... Project processing complete: /data/projects/project_2022_2023 (wall-clock 670.9 s, ~0.19 h)
```

{: .warning }
> If the log is not clean, treat downstream plots and tables as potentially incomplete or misleading.
>
> Warning signs:
>
> - repeated missing observation messages
> - failures in a specific step
> - plot task errors
> - no DA output export message

{: .references }
> - [Advanced Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) (follow-up when the log shows errors)

---

## 2. Performance diagnostics (`plots/perf`)

The performance plots help you understand runtime distribution and resource behavior.

Files to inspect:

- `plots/perf/project_perf.png`
- `plots/perf/project_perf_metrics.csv`

Quick file presence check for performance diagnostics.

```bash
ls -lh /data/rofental/projects/project_2022_2023/plots/perf
echo;
head -10 /data/rofental/projects/project_2022_2023/plots/perf/project_perf_metrics.csv
```


{: .checks }
> How to use these outputs:
>
> - compare runtime across tutorial reruns
> - estimate cost of increasing `ensemble_size`
> - estimate cost of changing resolution (`100 m` vs coarser)
> - identify unexpectedly slow stages (I/O, plotting, step-level hotspots)

Performance outputs are especially useful when comparing tutorial runs with different ensemble sizes or resolutions.

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

These plots are the core DA diagnostics.

Typical outputs:

- `plots/assim/weights/step_*_weights.png`
- `plots/assim/ess/setup_ess_timeline_*.png`

What they show:

- **weights**: how strongly observations favor some particles over others
- **ESS (effective sample size)**: particle degeneracy indicator

Quick listing of assimilation diagnostics files.

```bash
find /data/rofental/projects/project_2022_2023/plots/assim -type f | sort
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

This directory contains the main user-facing validation and DA-result figures for the setup.

Typical files include:

- `fraction_timeseries.png` (SCF / wet-snow fractions over time)
- station snow depth plots
- station SWE plots
- ROI envelope exports and land-cover masking report

Quick listing of result plot files.

```bash
ls -1 /data/rofental/projects/project_2022_2023/plots/results
```


### Fraction time series (high-level DA behavior)

`fraction_timeseries.png` is often the fastest way to inspect:

- observation dates actually used,
- model-vs-observation fraction behavior,
- whether SCF and wet-snow observations are present where expected.

What to inspect:

- are observation markers present at the configured DA dates?
- do SCF and wet-snow events appear in the expected seasonal phases?
- are there obvious missing events or suspicious gaps?

### Station plots (snow depth / SWE)

These plots support validation and interpretation of model behavior at observation stations.

What to inspect:

- overall seasonal timing (accumulation / melt timing),
- amplitude differences (too much / too little snow),
- whether DA shifts the ensemble envelope relative to the open loop,
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

- `/data/rofental/projects/project_2022_2023/plots/results/fraction_timeseries.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_latschbloder_snow_depth_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_proviantdepot_snow_depth_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_latschbloder_swe_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/plots/results/setup_results_point_proviantdepot_swe_2022_2023.png`

<details markdown="block">
  <summary>Suggested plot review order (practical)</summary>

1. `fraction_timeseries.png` (observation timing and availability)
2. ESS / weights plots (DA behavior)
3. Station snow depth plots
4. Station SWE plots
5. NetCDF/grid outputs for spatial interpretation
</details>

### Reference plots (tutorial baseline)

Fraction time series:

![Fraction time series (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/fraction_timeseries.png)

_`fraction_timeseries.png`: check observation dates, SCF/wet-snow event timing, and gross model-vs-observation behavior._

What to read in this plot:

- whether observation markers exist at the configured DA dates,
- whether SCF events cluster in snow-cover relevant periods and wet-snow events in melt-season periods,
- whether DA moves the ensemble envelope in the expected direction relative to the open loop.

Station snow depth example (`latschbloder`):

![Latschbloder snow depth plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/setup_results_point_latschbloder_snow_depth_2022_2023.png)

_Snow depth comparison at `latschbloder` (open loop + ensemble + observations)._

What to read in this plot:

- timing of snow accumulation and melt,
- whether observed values stay within (or near) the ensemble envelope,
- whether DA visibly shifts the ensemble relative to the open loop around observation periods.

Station SWE example (`proviantdepot`):

![Proviantdepot SWE plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/setup_results_point_proviantdepot_swe_2022_2023.png)

_SWE comparison at `proviantdepot` (remember: station SWE observations are expected in **mm** in this tutorial setup)._

What to read in this plot:

- unit consistency first (obs in **mm** in this tutorial setup),
- amplitude mismatch (systematic bias) vs timing mismatch (phase error),
- whether DA corrections remain small/local or systematically shift the trajectory.

<details markdown="block">
  <summary>More station reference plots (tutorial baseline)</summary>

`proviantdepot` snow depth:

![Proviantdepot snow depth plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/setup_results_point_proviantdepot_snow_depth_2022_2023.png)

`latschbloder` SWE:

![Latschbloder SWE plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/setup_results_point_latschbloder_swe_2022_2023.png)

</details>

{: .references }
> - [Observation Processing]({{ site.baseurl }}{% link guides/observations.md %}) (SCF / wet-snow preprocessing context)
> - [Workflow]({{ site.baseurl }}{% link workflow.md %}) (where these plots fit in the DA workflow)

---

## 5. ROI envelopes and summary CSVs

The project root contains setup-level envelope time series:

- `point_scf_roi_envelope.csv`
- `point_wet_snow_roi_envelope.csv`

These summarize the ensemble spread over the ROI (mean/min/max and sample count).

These CSVs are lightweight outputs that are ideal for quick comparisons between runs without loading NetCDF files.

Quick CSV inspection for ROI envelope outputs.

```bash
echo "SCF envelope:"
head -5 /data/rofental/projects/project_2022_2023/point_scf_roi_envelope.csv
echo;
echo "Wet-snow envelope:"
head -5 /data/rofental/projects/project_2022_2023/point_wet_snow_roi_envelope.csv
```


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

## 6. DA output summary NetCDF (`da_output_grids.nc`)

The tutorial setup writes a DA output summary NetCDF:

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

Quick file presence/size check for the DA summary NetCDF output.

```bash
ls -lh /data/rofental/projects/project_2022_2023/results/grids/da_output_grids.nc
```


Optional variable/dimension inspection (Python in the container).

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

Use this terminal snippet to verify two things before deeper analysis: the grid dimensions are plausible for the tutorial setup, and the expected open-loop/ensemble/increment variables were actually exported.

For the tutorial, GIS screenshots of selected NetCDF layers can be more intuitive than raw terminal dumps.

### Map placeholders (recommended tutorial additions)

Use `results/grids/da_output_grids.nc` in GIS software (or Python) and create screenshots
with a **consistent color scale** for direct comparison.

Recommended map date(s): choose one date with active snow cover and one date near melt season. Use the same date across `open_loop`, `ens_mean`, and `increment` maps.

### Placeholder (map pair: snow depth, open loop vs ensemble mean)

> Insert two maps for the same date from `da_output_grids.nc`:
> - `open_loop_snowdepth_daily`
> - `ens_mean_snowdepth_daily`
>  
> Use the same color scale and include the date in the caption.

### Placeholder (map pair: SWE, open loop vs ensemble mean)

> Insert two maps for the same date from `da_output_grids.nc`:
> - `open_loop_swe_daily`
> - `ens_mean_swe_daily`
>  
> Use the same color scale and include the date in the caption.

### Placeholder (map: DA increments)

> Insert one or two increment maps (same date) from `da_output_grids.nc`:
> - `increment_snowdepth_daily`
> - `increment_swe_daily`
>  
> Highlight where DA adds/removes snow relative to the open loop.

{: .references }
> - [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) (DA output variable selection and metrics)
> - [Project Structure]({{ site.baseurl }}{% link project-structure.md %}) (where project outputs live)
> - [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) (how to interpret increments conceptually)

---

## 7. Quick DA sanity checklist (practical review routine)

{: .checks }
> Use this checklist as a quick review before trusting or comparing results.
>
> Use this checklist after each tutorial run:
>
> 1. Log ends with `Project processing complete`
> 2. `plots/perf/project_perf.png` exists
> 3. `plots/assim/` contains ESS and weight plots
> 4. `plots/results/fraction_timeseries.png` exists
> 5. Station plots exist (snow depth and SWE)
> 6. `point_scf_roi_envelope.csv` and `point_wet_snow_roi_envelope.csv` exist
> 7. `results/grids/da_output_grids.nc` exists and opens
>
> If one of these fails, check the log before changing configuration.

---

## Next step

{: .references }
> Continue to the adaptation chapter after you can navigate and interpret the tutorial outputs:
>
> - [8. Adapting the example to your own project]({{ site.baseurl }}{% link Tutorial/08-adapting-to-your-own-project.md %}) (transfer the workflow from the Rofental tutorial setup to a new domain)
