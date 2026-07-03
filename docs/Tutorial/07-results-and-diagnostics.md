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
- `results/plots/perf/`
- `results/plots/assim/`
- `results/plots/results/`
- `results/maps/`
- `results/reports/`
- `results/misc/`
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

## 2. Performance diagnostics (`results/plots/perf`)

The performance plots help you understand runtime distribution and resource behavior.

Files to inspect:

- `results/plots/perf/project_perf.png`
- `results/plots/perf/project_perf_metrics.csv`

Reference structure snippet (`results/plots/perf`)

```text
/data/rofental/projects/project_2022_2023/results/plots/perf/
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

File path: `/data/rofental/projects/project_2022_2023/results/plots/perf/project_perf_metrics.csv`

| timestamp | cpu_total_pct | mem_used_pct | mem_used_gb | mem_total_gb | disk_fs_used_pct | disk_project_used_gb | cpu_temp_c | cpu_temp_source | thermal_sample_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-21T21:28:14 | 0.00 | 4.10 | 1.01 | 24.45 | 30.10 | 0.82 | 48.2 | psutil:k10temp:Tctl | true |
| 2026-02-21T21:28:19 | 40.90 | 14.10 | 3.44 | 24.45 | 30.10 | 0.82 | 66.4 | psutil:k10temp:Tctl | true |
| 2026-02-21T21:28:24 | 53.30 | 16.00 | 3.91 | 24.45 | 30.11 | 0.82 | 72.8 | psutil:k10temp:Tctl | true |
| 2026-02-21T21:28:29 | 52.50 | 17.70 | 4.32 | 24.45 | 30.11 | 0.82 | 74.1 | psutil:k10temp:Tctl | true |
| 2026-02-21T21:28:34 | 61.90 | 19.40 | 4.74 | 24.45 | 30.11 | 0.82 | 76.0 | psutil:k10temp:Tctl | true |

The CSV also keeps absolute filesystem used/free/total columns (`disk_fs_used_gb`, `disk_fs_free_gb`, `disk_fs_total_gb`) and optional critical-temperature metadata. If thermal sensors are unavailable, `cpu_temp_c` and `cpu_temp_crit_c` are blank and `thermal_sample_ok` is `false`.

Plot file to open:

- `/data/rofental/projects/project_2022_2023/results/plots/perf/project_perf.png`

Reference plot (tutorial baseline, `ensemble_size=30`):

![Project performance plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/project_perf.png)

_`project_perf.png` from the Rofental tutorial reference run (`100 m`, `ensemble_size=30`)._

What to read in the plot:

- **Relative utilization curves**: look for sustained CPU use during step processing, stable RAM use, and filesystem-used percentage that stays below critical disk pressure.
- **Project-size curve**: compare project directory growth against filesystem-used percentage. Project size is scanned at a throttled interval, so it can update in steps rather than every sample.
- **Thermal curve**: when host sensors are readable, CPU temperature is plotted on its own axis. Missing thermal data is expected in some containers and does not invalidate the CPU/RAM/disk diagnostics.
- **Timing structure**: repeated patterns often correspond to repeated step execution.

{: .references }
> - [Advanced Performance]({{ site.baseurl }}{% link advanced/performance.md %}) (deeper interpretation of runtime behavior)

---

## 3. Assimilation diagnostics (`results/plots/assim`)

These plots are the core data assimilation diagnostics.

Typical outputs:

- `results/plots/assim/weights/DA_XX_weights.png`
- `results/plots/assim/weights/setup_weights_overview_*.png`
- `results/plots/assim/ess/setup_ess_timeline_*.png`
- `results/plots/assim/scores/performance_scores.png`

What they show:

- **weights**: how strongly observations favor some particles over others
- **ESS (effective sample size)**: particle degeneracy indicator
- **headline benchmark scores**: update-date `CRPSS`, `NER`, and station-only `zSkill` when available

Reference structure snippet (`results/plots/assim`, typical files)

```text
/data/rofental/projects/project_2022_2023/results/plots/assim/
  ess/
    setup_ess_timeline_2022_2023.png
  scores/
    performance_scores.png
	weights/
	  DA_01_weights.png
	  ...
	  DA_08_weights.png
	  setup_weights_overview_2022_2023.png
    setup_weights_overview_2022_2023_page_02.png
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
> - abrupt differences between station HS, SCF, and wet-snow events:
>   - also normal and expected (different variables, support, and information content)

ESS is a diagnostic, not a simple "good/bad" score. Interpret it together with weights, variable type, and observation coverage.

Reference CSV snippet (weights for one station HS event)

File path: `/data/rofental/projects/project_2022_2023/steps/step_04_20230131-20230221/assim/weights_station_hs_20230221.csv`

| member_id | value_obs | value_model | residual | sigma | n_stations | weight |
| --- | --- | --- | --- | --- | --- | --- |
| member_001 | 0.22 | 0.47 | -0.25 | 0.20 | 2 | 0.10 |
| member_002 | 0.22 | 1.28 | -1.06 | 0.20 | 2 | 0.00 |
| member_003 | 0.22 | 0.70 | -0.48 | 0.20 | 2 | 0.00 |
| member_004 | 0.22 | 0.56 | -0.34 | 0.20 | 2 | 0.05 |
| member_005 | 0.22 | 0.48 | -0.26 | 0.20 | 2 | 0.10 |

Plot files to open:

- `/data/rofental/projects/project_2022_2023/results/plots/assim/ess/setup_ess_timeline_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/results/plots/assim/scores/performance_scores.png`
- `/data/rofental/projects/project_2022_2023/results/plots/assim/weights/setup_weights_overview_2022_2023.png`
  - if the setup has many assimilation dates, open the numbered continuation pages (`..._page_02.png`, `..._page_03.png`, ...) as well
- `/data/rofental/projects/project_2022_2023/results/plots/assim/weights/DA_04_weights.png`

Reference ESS plot (tutorial baseline):

![ESS timeline plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/setup_ess_timeline_2022_2023.png)

_ESS timeline (`setup_ess_timeline_2022_2023.png`) from the tutorial reference run._

What to read in the ESS plot:

- each point corresponds to one assimilation event,
- lower ESS means stronger weight concentration (more degeneracy),
- differences between station HS, SCF, and wet-snow events are expected because the observation types have different information content and spatial support.

Reference setup weights overview:

![Setup weights overview (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/setup_weights_overview_2022_2023.png)

_Setup-wide comparison of all eight assimilation events, grouped by observable family._

For larger projects, the setup overview is automatically split into multiple A4-length PNG pages that keep the first file name above and add numbered continuation pages.

Reference weights plot (example event):

![Weights plot for one assimilation event (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/DA_04_weights.png)

_Weights plot for `DA_04` (`station_hs` on `2023-02-21`)._

What to read in the weights plot:

- a **flat** distribution means the observation did not strongly discriminate particles,
- a **peaked** distribution means a few particles explain the observation much better,
- very strong peaks often coincide with low ESS and potential resampling pressure.

Exact weights differ between runs because the ensemble is stochastic. Focus on the structure (spread/concentration), not exact numeric values.

{: .references }
> - [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) (likelihood, resampling, rejuvenation)
> - [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) (theory and terminology)

---

## 4. Point plots (`results/plots/points`)

This directory contains the stitched station and ROI point plots for the project.

Typical files include:

- `result_overview.png` (fSCA, wet-snow, ROI mean SWE, and ROI mean snow depth over time)
- station snow depth plots
- station SWE plots
- ROI envelope exports and land-cover masking report

Reference structure snippet (`results/plots/points`, typical files)

```text
/data/rofental/projects/project_2022_2023/results/plots/points/
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
- whether fSCA and wet-snow observations are present where expected.

The ROI SWE and snow-depth panels use the full ROI footprint rather than the land-cover-masked ROI used for fSCA and wet-snow summaries.

What to inspect:

- are observation markers present at the configured data assimilation dates?
- do fSCA and wet-snow events appear in the expected seasonal phases?
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

Land-cover masking affects how much of the ROI contributes to fSCA/wet-snow summaries and
fractions. This report is useful here because it explains the masking context behind the
result plots and ROI envelope values shown in this chapter.

Reference CSV snippet (land-cover masking report)

File path: `/data/rofental/projects/project_2022_2023/results/misc/lc_mask_report.csv`

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
- use it as context when fSCA/wet-snow fractions look unexpectedly low/high

Recommended plot files to inspect (Rofental tutorial run):

- `/data/rofental/projects/project_2022_2023/results/plots/results/result_overview.png`
- `/data/rofental/projects/project_2022_2023/results/plots/points/setup_results_point_latschbloder_snow_depth_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/results/plots/points/setup_results_point_proviantdepot_snow_depth_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/results/plots/points/setup_results_point_latschbloder_swe_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/results/plots/points/setup_results_point_proviantdepot_swe_2022_2023.png`

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

![Result overview (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/result_overview.png)

_`result_overview.png`: check observation dates, fSCA/wet-snow event timing, ROI mean SWE / snow-depth behavior, and gross model-vs-observation behavior._

What to read in this plot:

- whether observation markers exist at the configured data assimilation dates,
- whether fSCA events cluster in snow-cover relevant periods and wet-snow events in melt-season periods,
- whether ROI mean SWE and snow depth shift away from or back toward the open loop during the season,
- whether data assimilation moves the ensemble envelope in the expected direction relative to the open loop.

Station snow depth example (`latschbloder`):

![Latschbloder snow depth plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/setup_results_point_latschbloder_snow_depth_2022_2023.png)

_Snow depth comparison at `latschbloder` (open loop + ensemble + observations)._

What to read in this plot:

- timing of snow accumulation and melt,
- whether observed values stay within (or near) the ensemble envelope,
- whether data assimilation visibly shifts the ensemble relative to the open loop around observation periods.

Station SWE example (`proviantdepot`):

![Proviantdepot SWE plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/setup_results_point_proviantdepot_swe_2022_2023.png)

_SWE comparison at `proviantdepot` (remember: station SWE observations are expected in **mm** in this tutorial setup)._

What to read in this plot:

- unit consistency first (obs in **mm** in this tutorial setup),
- amplitude mismatch (systematic bias) vs timing mismatch (phase error),
- whether data assimilation corrections remain small/local or systematically shift the trajectory.

<details markdown="block">
  <summary>More station reference plots (tutorial baseline)</summary>

`proviantdepot` snow depth:

![Proviantdepot snow depth plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/setup_results_point_proviantdepot_snow_depth_2022_2023.png)

`latschbloder` SWE:

![Latschbloder SWE plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/setup_results_point_latschbloder_swe_2022_2023.png)

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
The summary NetCDF uses compact internal storage for DA-owned snow grids: snow depth is stored at 0.001 m resolution, while SWE and liquid-water content are stored at integer millimeter resolution. Normal CF-aware readers such as xarray return decoded physical values.

Reference output file path (data assimilation summary NetCDF):

- `/data/rofental/projects/project_2022_2023/results/grids/da_output_grids.nc`

Adjacent completed projects with the same domain, grid, CRS, variables and compact encoding can be merged into one time-concatenated DA summary NetCDF:

```bash
oa-da-merge-project-grids \
  --setup /data/rofental \
  --project project_2020_2021 \
  --project project_2021_2022 \
  --project project_2022_2023 \
  --output-nc /data/rofental/results/grids/da_output_grids_2020_2023.nc
```

The merge command fails if variables, static coordinates, grid metadata or timestamps are incompatible. It preserves the usual DA summary variable names and records source-project provenance in global NetCDF attributes.

For adjacent completed projects, the snow point-result CSVs can also be stitched into a compact multi-year snow-plot bundle:

```bash
oa-da-plot-multi-project-snow \
  --setup /data/rofental \
  --project project_2020_2021 \
  --project project_2021_2022 \
  --project project_2022_2023 \
  --project project_2023_2024
```

By default, this writes station and ROI PNGs to `results/plots/multi_year_snow`. The plots show open loop, ensemble mean and the 5-95% member envelope; station observations are added where available. Station model and observation series are shown as daily means, negative station observations are masked and no DA-event markers are drawn. If `results/maps/setup_overview.png` exists in a source project, it is copied into the bundle as `context_map.png`.

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
> - open-loop departure fields (`increment = ens_mean - open_loop`)
> - DA-event analysis fields (`analysis_increment = analysis_mean - ens_mean`) on event dates with weights

Dimension names in the inspected NetCDF (for example `time1`, `time2`, `snow_layer`, `nbnd`) are typically inherited from the underlying model outputs. Configure **which variables/metrics** are exported in the project YAML under `data_assimilation.output.grids.variables[*]`; see [5. Running the Model]({{ site.baseurl }}{% link Tutorial/06-running-the-project.md %}) for the output-grid configuration note.

### Raster output

{: .checks }
> Generated map examples from the Rofental tutorial reference run:
> - `results/maps/da_events/da_6.png`: WSLA update on **2023-03-24**
> - `results/maps/da_events/da_8.png`: fSCA (`scf`) update on **2023-05-26**

![Generated DA-event map for the WSLA update on 2023-03-24]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/da_06_wsla_2023_03_24.png)

_`da_6.png`: open loop, prior, posterior and observed wet snow fraction maps with WSLA contours, elevation-band wet snow fraction maps, and corresponding snow-depth fields for the **2023-03-24** `wet_snow_line` update._

![Generated DA-event map for the fSCA update on 2023-05-26]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/da_08_scf_2023_05_26.png)

_`da_8.png`: open loop, prior, posterior and observed fSCA diagnostics plus corresponding snow-depth fields for the **2023-05-26** `scf` update._

{: .checks }
> Note on raster workflow:
> - all output grids are stored in one NetCDF file: `results/grids/da_output_grids.nc` (see [5. data assimilation output summary NetCDF (`da_output_grids.nc`)](#da-output-summary-netcdf))
> - generated DA-event maps use event dates with available particle-filter weights
> - for non-event dates, extract the layers/time slices you need in the GIS tool of your choice before styling or export

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
  analysis_increment_liquid_water_content
  analysis_increment_snowdepth_daily
  analysis_increment_swe_daily
  analysis_mean_liquid_water_content
  analysis_mean_snowdepth_daily
  analysis_mean_swe_daily
  increment_liquid_water_content
  increment_snowdepth_daily
  increment_swe_daily
  open_loop_liquid_water_content
  open_loop_snowdepth_daily
  open_loop_swe_daily
```

Use `results/grids/da_output_grids.nc` in a GIS software of your choice and visualize raster output.

Recommended manual map date(s): choose one date with active snow cover and one date near melt season.
Use the same date across `open_loop`, `ens_mean`, and `increment` maps. Generated DA-event maps use four columns: `open loop`, `prior`, `posterior`, and `reference`. Snow depth maps use `ens_mean` as the prior mean, `analysis_mean` as the event-weighted posterior mean, and `analysis_increment` as `posterior - prior`.

For the shipped examples, project maps are split into generated DA-event maps under `results/maps/da_events/` and custom YAML maps such as `setup_overview` at the root of `results/maps/`. Use `oa-da-plot-project-maps --project-dir /data/rofental/projects/project_2022_2023 --max-workers 4` to rerender the full combined map set in one command. Omit `--max-workers` to let the Docker container auto-select a recipe-level worker count from the visible CPUs. Overview panels use setup-local GISCO GeoJSONs under `env/`; if you want to prefetch them ahead of time, run `oa-da-fetch-overview-geojson --project-dir /data/rofental/projects/project_2022_2023`.
For `snowdepth_daily`, the map renderer uses the viridis palette together with a shared linear legend scale per render run. Tick labels are shown in `cm`, cells below `1 cm` stay transparent so only meaningful snow cover is colored, and the top of the snow-depth legend is derived from the plotted maps. Increment panels use a signed red-blue diverging palette: negative increments are red, positive increments are blue. In generated DA-event maps, positive `analysis_increment` means the DA event added snow; negative means it removed snow.

`oa-da-project` attempts to collect the project summary page, overview outputs, diagnostics, and DA-event maps into `results/reports/project_report.pdf` at the end of the run, after plots, maps, and benchmark-dependent overview panels are current. Report generation is best-effort in the pipeline: missing prerequisites are logged with a manual rerun command and do not fail the completed model run. To regenerate only the PDF later, run `oa-da-project-pdf --project-dir /data/rofental/projects/project_2022_2023`. The first PDF page contains basic setup YAML settings, wet-snow classification and liquid-water-content settings, DA-event counts, computing-cost stats, and a bottom `Content` table with page numbers first and section names second. The PDF then includes the overview plots, setup map, setup weights overview pages, station snow-depth point plots on one page, `performance_scores.png`, `project_perf.png`, and generated DA-event maps in temporal order. Source PNGs are placed at their shared export-DPI size rather than scaled down to fit a page; consecutive DA maps are packed onto a page only while the reserved bottom gap is preserved. Standalone per-event weights plots and other remaining plot/map PNGs are not included.

### data assimilation increment map

The tutorial includes generated DA-event maps above. For event-level diagnostics, use these maps or export
`analysis_increment_snowdepth_daily` from `da_output_grids.nc` on an assimilation date and compare it against the same-date prior/posterior mean maps.

{: .references }
> - [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) (data assimilation output variable selection and metrics)
> - [Project Structure]({{ site.baseurl }}{% link project-structure.md %}) (where project outputs live)
> - [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) (how to interpret increments conceptually)

---

## 6. ROI envelopes and summary CSVs

The project now writes these setup-level envelope time series under `results/misc/`:

- `point_scf_roi_envelope.csv`
- `point_wet_snow_roi_envelope.csv`

These summarize the ensemble spread over the ROI (mean/min/max and sample count).

These CSVs are lightweight outputs that are ideal for quick comparisons between runs without loading NetCDF files.

Reference file paths for ROI envelope outputs:

- `/data/rofental/projects/project_2022_2023/results/misc/point_scf_roi_envelope.csv`
- `/data/rofental/projects/project_2022_2023/results/misc/point_wet_snow_roi_envelope.csv`


{: .checks }
> Why these CSVs are useful:
>
> - quick numeric QA without opening plots
> - useful for external plotting notebooks or reports
> - easy comparison across experimental runs

Reference CSV snippet (SCF ROI envelope)

File path: `/data/rofental/projects/project_2022_2023/results/misc/point_scf_roi_envelope.csv`

| date | value_mean | value_min | value_max | n |
| --- | --- | --- | --- | --- |
| 2022-10-01 | 0.00 | 0.00 | 0.00 | 11 |
| 2022-10-02 | 0.12 | 0.00 | 0.50 | 11 |
| 2022-10-03 | 0.08 | 0.00 | 0.40 | 11 |
| 2022-10-04 | 0.09 | 0.00 | 0.42 | 11 |

Reference CSV snippet (wet-snow ROI envelope)

File path: `/data/rofental/projects/project_2022_2023/results/misc/point_wet_snow_roi_envelope.csv`

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
> 2. `results/plots/perf/project_perf.png` exists
> 3. `results/plots/assim/` contains ESS, weight, and score plots
> 4. `results/plots/results/result_overview.png` exists
> 5. Station plots exist (snow depth and SWE)
> 6. `results/misc/point_scf_roi_envelope.csv` and `results/misc/point_wet_snow_roi_envelope.csv` exist
> 7. `results/grids/da_output_grids.nc` exists and opens
>
> If one of these fails, check the log before changing configuration.

---

## 8. Project cleanup (optional)

openAMUNDSEN-DA contains a module that cleans up heavy files that are used within the data assimilation workflow and not needed anymore after running a project. The cleanup is wired into the project pipeline and activated by default.

{: .checks }
> Automatic cleanup is enabled by default via `data_assimilation.restart.cleanup_after_setup: true`.
> Use manual cleanup if you disabled automatic cleanup or if older projects still contain state files.

Clean one project (`project_2022_2023`):

**🟢 Run this command:**

```bash
oa-da-clean-project \
  --setup-dir /data/rofental \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --log-level INFO
```

Clean all projects under the same setup:

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
