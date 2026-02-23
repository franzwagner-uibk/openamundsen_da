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

{: .highlight }
> Review order matters: check the log first, then DA diagnostics, then interpret the plots and grids.

---

## Project output structure (what to inspect first)

Most tutorial outputs live under:

- `rofental/projects/project_2022_2023/`

Key locations:

- `project_2022_2023.log` - full run log
- `plots/perf/` - runtime/performance plots and metrics
- `plots/assim/` - DA diagnostics (weights, ESS)
- `plots/results/` - result plots (fractions, station plots, envelopes)
- `results/grids/da_output_grids.nc` - compact DA grid product
- `point_scf_roi_envelope.csv`, `point_wet_snow_roi_envelope.csv` - ROI time-series envelopes

Quick directory overview:

```bash
docker run --rm -v "/absolute/path/to/tutorial-workdir:/data" ghcr.io/franzwagner-uibk/openamundsen_da:latest sh -lc '
  find /data/rofental/projects/project_2022_2023 -maxdepth 3 -type d | sort
'
```

<details markdown="block">
  <summary>Why this directory overview is useful</summary>

It helps new users understand which outputs are:

- run logs,
- diagnostics,
- plotting products,
- compact results,
- and step/member internals.
</details>

---

## 1. Read the log first (always)

Before interpreting plots, confirm that the run actually completed cleanly.

```bash
docker run --rm -v "/absolute/path/to/tutorial-workdir:/data" ghcr.io/franzwagner-uibk/openamundsen_da:latest sh -lc '
  tail -n 120 /data/rofental/projects/project_2022_2023/project_2022_2023.log
'
```
<details markdown="block">
  <summary>Windows / PowerShell note (same command)</summary>

Use the same command on Windows in one of these ways:

- **Recommended:** run the Bash command in **WSL** (works as shown).
- **PowerShell:** keep the same Docker/image/container paths, adjust only:
  - host mount path syntax (e.g. `C:/...:/data`)
  - line continuation (PowerShell uses the backtick `` ` `` instead of `\`)

For Bash-specific host-shell constructs (for example heredocs), prefer WSL/Git Bash or use a PowerShell here-string equivalent.
</details>


What to look for:

- `Project processing complete`
- DA variable processing messages (`scf`, `wet_snow`)
- plot tasks completed
- compact DA output writing (`da_output_grids.nc`)
- cleanup/retention messages (expected in compact mode)

Reference snippet (successful log tail excerpt):

```text
2026-02-21 ... Plot task setup_ess_timeline completed
2026-02-21 ... Plot task setup_results_swe completed
2026-02-21 ... Plot task setup_results_snow_depth completed
2026-02-21 ... Wrote DA output summary NetCDF /data/projects/project_2022_2023/results/grids/da_output_grids.nc (18 step(s))
2026-02-21 ... Compact retention: deleted 3190 grid artifact file(s), freed 305.1 MB
2026-02-21 ... Setup cleanup succeeded: deleted 198/198 file(s), freed 1045.7 MB (patterns=model_state.pickle.gz)
2026-02-21 ... Project processing complete: /data/projects/project_2022_2023 (wall-clock 866.7 s, ~0.24 h)
```

Potential warning signs:

- repeated missing observation messages
- failures in a specific step
- plot task errors
- no DA output export message

{: .warning }
> If the log is not clean, treat downstream plots and tables as potentially incomplete or misleading.

{: .note }
> Cross-reference:
> - [Advanced Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %})

---

## 2. Performance diagnostics (`plots/perf`)

The performance plots help you understand runtime distribution and resource behavior.

Files to inspect:

- `plots/perf/project_perf.png`
- `plots/perf/project_perf_metrics.csv`

Quick check:

```bash
docker run --rm -v "/absolute/path/to/tutorial-workdir:/data" ghcr.io/franzwagner-uibk/openamundsen_da:latest sh -lc '
  ls -lh /data/rofental/projects/project_2022_2023/plots/perf
  echo;
  head -10 /data/rofental/projects/project_2022_2023/plots/perf/project_perf_metrics.csv
'
```
<details markdown="block">
  <summary>Windows / PowerShell note (same command)</summary>

Use the same command on Windows in one of these ways:

- **Recommended:** run the Bash command in **WSL** (works as shown).
- **PowerShell:** keep the same Docker/image/container paths, adjust only:
  - host mount path syntax (e.g. `C:/...:/data`)
  - line continuation (PowerShell uses the backtick `` ` `` instead of `\`)

For Bash-specific host-shell constructs (for example heredocs), prefer WSL/Git Bash or use a PowerShell here-string equivalent.
</details>


How to use this information:

- compare runtime across tutorial reruns,
- estimate cost of increasing `ensemble_size`,
- estimate cost of changing resolution (`100 m` vs coarser),
- identify unexpectedly slow stages (I/O, plotting, step-level hotspots).

{: .note }
> Performance outputs are especially useful when comparing tutorial runs with different ensemble sizes or resolutions.

Reference CSV snippet (performance metrics)

File path:

- `/data/rofental/projects/project_2022_2023/plots/perf/project_perf_metrics.csv`

```csv
timestamp,cpu_total_pct,mem_used_pct,mem_used_gb,mem_total_gb
2026-02-21T21:28:14,0.000,4.100,1.013,24.452
2026-02-21T21:28:19,40.900,14.100,3.442,24.452
2026-02-21T21:28:24,53.300,16.000,3.906,24.452
2026-02-21T21:28:29,52.500,17.700,4.316,24.452
2026-02-21T21:28:34,61.900,19.400,4.738,24.452
```

Plot file to open:

- `/data/rofental/projects/project_2022_2023/plots/perf/project_perf.png`

Reference plot (tutorial baseline, `ens=10`):

![Project performance plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/project_perf.png)

_`project_perf.png` from the Rofental tutorial reference run (`100 m`, `ensemble_size=10`)._

What to read in the plot:

- **CPU utilization panel/curve**: look for sustained utilization during step processing and drops during lighter phases (I/O, orchestration).
- **Memory usage panel/curve**: check whether memory stays within a stable range and does not continuously climb (which can indicate a leak or runaway buffering).
- **Timing structure**: repeated patterns often correspond to repeated step execution.

{: .note }
> Cross-reference:
> - [Advanced Performance]({{ site.baseurl }}{% link advanced/performance.md %})

---

## 3. Assimilation diagnostics (`plots/assim`)

These plots are the core DA diagnostics.

Typical outputs:

- `plots/assim/weights/step_*_weights.png`
- `plots/assim/ess/setup_ess_timeline_*.png`

What they show:

- **weights**: how strongly observations favor some particles over others
- **ESS (effective sample size)**: particle degeneracy indicator

Quick listing:

```bash
docker run --rm -v "/absolute/path/to/tutorial-workdir:/data" ghcr.io/franzwagner-uibk/openamundsen_da:latest sh -lc '
  find /data/rofental/projects/project_2022_2023/plots/assim -type f | sort
'
```
<details markdown="block">
  <summary>Windows / PowerShell note (same command)</summary>

Use the same command on Windows in one of these ways:

- **Recommended:** run the Bash command in **WSL** (works as shown).
- **PowerShell:** keep the same Docker/image/container paths, adjust only:
  - host mount path syntax (e.g. `C:/...:/data`)
  - line continuation (PowerShell uses the backtick `` ` `` instead of `\`)

For Bash-specific host-shell constructs (for example heredocs), prefer WSL/Git Bash or use a PowerShell here-string equivalent.
</details>


Interpretation guidelines:

- ESS near ensemble size for many events:
  - observations have weak discrimination or high observation error
- ESS very low (near 1) frequently:
  - strong degeneracy, aggressive resampling likely
  - possibly too-small observation error (`obs_sigma`) or too-strong mismatch
- abrupt differences between SCF and wet-snow events:
  - normal and expected (different variables, coverage, and information content)

{: .highlight }
> ESS is a diagnostic, not a simple "good/bad" score. Interpret it together with weights, variable type, and observation coverage.

Reference CSV snippet (weights for one wet-snow event)

File path:

- `/data/rofental/projects/project_2022_2023/steps/step_05_20230309-20230328/assim/weights_wet_snow_20230328.csv`

```csv
member_id,wet_snow_model,wet_snow_obs,residual,sigma,log_weight,weight
member_001,0.06530984204131227,0.2661,0.20079015795868774,0.1,-0.6321878168643655,0.09226043764559255
member_002,0.05407047387606318,0.2661,0.21202952612393683,0.1,-0.8641794376276875,0.07315816937269848
member_003,0.030528554070473876,0.2661,0.2355714459295261,0.1,-1.3910487470770088,0.043196284798504535
member_008,0.106318347509113,0.2661,0.159781652490887,0.1,0.10713773615344424,0.1932415527015183
member_010,0.024756986634264885,0.2661,0.2413430133657351,0.1,-1.5286759452332963,0.03764225766385648
```

Plot files to open:

- `/data/rofental/projects/project_2022_2023/plots/assim/ess/setup_ess_timeline_2022_2023.png`
- `/data/rofental/projects/project_2022_2023/plots/assim/weights/step_05_weights.png`

Reference ESS plot (tutorial baseline):

![ESS timeline plot (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/setup_ess_timeline_2022_2023.png)

_ESS timeline (`setup_ess_timeline_2022_2023.png`) from the tutorial reference run._

What to read in the ESS plot:

- each point corresponds to one assimilation event,
- lower ESS means stronger weight concentration (more degeneracy),
- differences between SCF and wet-snow events are expected because the observation types have different information content and spatial support.

Reference weights plot (example step):

![Weights plot for one assimilation step (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/step_05_weights.png)

_Weights plot for `step_05` (wet-snow event around `2023-03-28`)._

What to read in the weights plot:

- a **flat** distribution means the observation did not strongly discriminate particles,
- a **peaked** distribution means a few particles explain the observation much better,
- very strong peaks often coincide with low ESS and potential resampling pressure.

{: .note }
> Exact weights differ between runs because the ensemble is stochastic. Focus on the structure (spread/concentration), not exact numeric values.

{: .note }
> Cross-reference:
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

Quick check:

```bash
docker run --rm -v "/absolute/path/to/tutorial-workdir:/data" ghcr.io/franzwagner-uibk/openamundsen_da:latest sh -lc '
  ls -1 /data/rofental/projects/project_2022_2023/plots/results
'
```
<details markdown="block">
  <summary>Windows / PowerShell note (same command)</summary>

Use the same command on Windows in one of these ways:

- **Recommended:** run the Bash command in **WSL** (works as shown).
- **PowerShell:** keep the same Docker/image/container paths, adjust only:
  - host mount path syntax (e.g. `C:/...:/data`)
  - line continuation (PowerShell uses the backtick `` ` `` instead of `\`)

For Bash-specific host-shell constructs (for example heredocs), prefer WSL/Git Bash or use a PowerShell here-string equivalent.
</details>


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

{: .note }
> In this tutorial setup, station SWE observations are expected in **mm** (see project config comment).
> If a curve appears near zero against model SWE, check units first.

Reference CSV snippet (land-cover masking report)

File path:

- `/data/rofental/projects/project_2022_2023/plots/results/lc_mask_report.csv`

```csv
class_code,class_name,cells,area_km2,percent_of_roi
2,ice,3257,32.570000,32.8210
3,water,15,0.150000,0.1512
10,mixed forest,48,0.480000,0.4837
"8,9,10,11,12",forest,69,0.690000,0.6953
total,total,3346,33.460000,33.7179
```

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

{: .note }
> Cross-reference:
> - [Observation Processing]({{ site.baseurl }}{% link guides/observations.md %}) for SCF / wet-snow preprocessing context
> - [Workflow]({{ site.baseurl }}{% link workflow.md %}) for how these plots fit into the broader DA workflow

---

## 5. ROI envelopes and summary CSVs

The project root contains setup-level envelope time series:

- `point_scf_roi_envelope.csv`
- `point_wet_snow_roi_envelope.csv`

These summarize the ensemble spread over the ROI (mean/min/max and sample count).

{: .highlight }
> These CSVs are lightweight outputs that are ideal for quick comparisons between runs without loading NetCDF files.

Inspect a snippet:

```bash
docker run --rm -v "/absolute/path/to/tutorial-workdir:/data" ghcr.io/franzwagner-uibk/openamundsen_da:latest sh -lc '
  echo "SCF envelope:"
  head -5 /data/rofental/projects/project_2022_2023/point_scf_roi_envelope.csv
  echo;
  echo "Wet-snow envelope:"
  head -5 /data/rofental/projects/project_2022_2023/point_wet_snow_roi_envelope.csv
'
```
<details markdown="block">
  <summary>Windows / PowerShell note (same command)</summary>

Use the same command on Windows in one of these ways:

- **Recommended:** run the Bash command in **WSL** (works as shown).
- **PowerShell:** keep the same Docker/image/container paths, adjust only:
  - host mount path syntax (e.g. `C:/...:/data`)
  - line continuation (PowerShell uses the backtick `` ` `` instead of `\`)

For Bash-specific host-shell constructs (for example heredocs), prefer WSL/Git Bash or use a PowerShell here-string equivalent.
</details>


How this helps:

- quick numeric QA without opening plots,
- useful for external plotting notebooks or reports,
- easy comparison across experimental runs.

Reference CSV snippet (SCF ROI envelope)

File path:

- `/data/rofental/projects/project_2022_2023/point_scf_roi_envelope.csv`

```csv
date,value_mean,value_min,value_max,n
2022-10-01,0.0,0.0,0.0,11
2022-10-02,0.12143250688705232,0.0,0.5036363636363637,11
2022-10-03,0.08418732782369143,0.0001515151515151,0.4,11
2022-10-04,0.38706611570247934,0.0827272727272727,0.4175757575757576,11
```

Reference CSV snippet (wet-snow ROI envelope)

File path:

- `/data/rofental/projects/project_2022_2023/point_wet_snow_roi_envelope.csv`

```csv
date,value_mean,value_min,value_max,n
2022-10-01,0.0,0.0,0.0,11
2022-10-02,0.5881613829669722,0.4532199270959903,0.7702004860267315,11
2022-10-03,0.035554512316359164,0.0034933171324422,0.1376063183475091,11
2022-10-04,0.3391969512868662,0.12773390036452,0.3789489671931956,11
```

Interpretation:

- `value_mean`: ensemble mean over `open_loop + ensemble members`
- `value_min` / `value_max`: spread envelope
- `n`: number of contributing trajectories (for tutorial baseline: `10` members + `1` open loop = `11`)

---

## 6. Compact DA grid output (`da_output_grids.nc`)

The tutorial setup writes a compact NetCDF summary of the DA outputs:

- `results/grids/da_output_grids.nc`

{: .note }
> **Output configuration (project YAML)**  
> File: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`  
> Relevant keys: `data_assimilation.output.retention`, `data_assimilation.output.grids.format`, `data_assimilation.output.grids.variables[*]`

This file is designed for:

- post-processing,
- visualization,
- comparison between runs,
- exporting selected variables without keeping all raw member grids.

Quick inspection of file presence/size:

```bash
docker run --rm -v "/absolute/path/to/tutorial-workdir:/data" ghcr.io/franzwagner-uibk/openamundsen_da:latest sh -lc '
  ls -lh /data/rofental/projects/project_2022_2023/results/grids/da_output_grids.nc
'
```
<details markdown="block">
  <summary>Windows / PowerShell note (same command)</summary>

Use the same command on Windows in one of these ways:

- **Recommended:** run the Bash command in **WSL** (works as shown).
- **PowerShell:** keep the same Docker/image/container paths, adjust only:
  - host mount path syntax (e.g. `C:/...:/data`)
  - line continuation (PowerShell uses the backtick `` ` `` instead of `\`)

For Bash-specific host-shell constructs (for example heredocs), prefer WSL/Git Bash or use a PowerShell here-string equivalent.
</details>


Optional variable/dimension inspection (Python in the container):

```bash
docker run --rm -v "/absolute/path/to/tutorial-workdir:/data" ghcr.io/franzwagner-uibk/openamundsen_da:latest python - <<'PY'
import xarray as xr
ds = xr.open_dataset("/data/rofental/projects/project_2022_2023/results/grids/da_output_grids.nc")
print(ds)
print("\nVariables:")
for name in ds.data_vars:
    print("-", name, ds[name].dims)
PY
```
<details markdown="block">
  <summary>Windows / PowerShell note (same command)</summary>

Use the same command on Windows in one of these ways:

- **Recommended:** run the Bash command in **WSL** (works as shown).
- **PowerShell:** keep the same Docker/image/container paths, adjust only:
  - host mount path syntax (e.g. `C:/...:/data`)
  - line continuation (PowerShell uses the backtick `` ` `` instead of `\`)

For Bash-specific host-shell constructs (for example heredocs), prefer WSL/Git Bash or use a PowerShell here-string equivalent.
</details>


What to expect conceptually:

- open-loop baseline fields,
- ensemble mean / spread fields,
- increments (`ens_mean - open_loop`) for configured variables/aggregations.

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

{: .note }
> For the tutorial, GIS screenshots of selected NetCDF layers can be more intuitive than raw terminal dumps.

### Map placeholders (recommended tutorial additions)

Use `results/grids/da_output_grids.nc` in GIS software (or Python) and create screenshots
with a **consistent color scale** for direct comparison.

{: .note }
> Recommended map date(s): choose one date with active snow cover and one date near melt season. Use the same date across `open_loop`, `ens_mean`, and `increment` maps.

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

{: .note }
> Cross-reference:
> - [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) (DA output variable selection and metrics)
> - [Project Structure]({{ site.baseurl }}{% link project-structure.md %}) (where project outputs live)
> - [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) (how to interpret increments conceptually)

---

## 7. Quick DA sanity checklist (practical review routine)

Use this checklist after each tutorial run:

1. Log ends with `Project processing complete`
2. `plots/perf/project_perf.png` exists
3. `plots/assim/` contains ESS and weight plots
4. `plots/results/fraction_timeseries.png` exists
5. Station plots exist (snow depth and SWE)
6. `point_scf_roi_envelope.csv` and `point_wet_snow_roi_envelope.csv` exist
7. `results/grids/da_output_grids.nc` exists and opens

If one of these fails, check the log before changing configuration.

---

## Next step

Continue with [8. Adapting the example to your own project]({{ site.baseurl }}{% link Tutorial/08-adapting-to-your-own-project.md %})
to learn how to transfer this workflow from the Rofental tutorial setup to a new domain.
