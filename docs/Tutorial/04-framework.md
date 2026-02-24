---
layout: default
title: 4. Framework
parent: Tutorial
nav_order: 4
permalink: /tutorial/framework/
---

# 4. Framework

This chapter explains the **internal structure** of openAMUNDSEN-DA at the level that is
most useful for running and debugging projects.

It answers questions like:

- Where does configuration live?
- What is a step?
- What is the difference between `open_loop`, prior, and posterior?
- How do observations enter the DA cycle?
- Which outputs are generated where?

Goal of this chapter: make the framework behavior predictable before you run commands.

Any command blocks shown in this chapter are executed **inside the running tutorial container shell** started in [2. Dependencies]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %}).

---

## Step-by-step flow on this page

{: .step }
> Use this chapter as the framework reference map for the rest of the tutorial.

Recommended reading order on this page:

1. read the visual overview note (how this chapter relates to chapter 1)
2. read sections `1`, `3`, `4`, and `5` first (core framework behavior)
3. use sections `2`, `6`, and `7` as file/path-oriented reference while running the tutorial
4. treat command blocks on this page as optional orientation checks (not mandatory steps)

This chapter is primarily for understanding and navigation. The mandatory execution sequence starts in [5. Pre-processing]({{ site.baseurl }}{% link Tutorial/05-pre-processing.md %}).

## Visual overview (recommended before reading details)

The integrated conceptual overview graphic is introduced in
[1. openAMUNDSEN-DA]({{ site.baseurl }}{% link Tutorial/01-openamundsen-da.md %}).
Use it there as a roadmap, then come back to this chapter for the concrete setup/project/step/member structure and file-level details.

{: .references }
> - [Workflow]({{ site.baseurl }}{% link workflow.md %}) for the end-to-end process view
> - [Project Structure]({{ site.baseurl }}{% link project-structure.md %}) for folder layout details
> - [Package Structure]({{ site.baseurl }}{% link reference/package-structure.md %}) for module-level code orientation
> - [DA Methods Reference]({{ site.baseurl }}{% link reference/da-methods.md %}) for the mathematical background of weighting/resampling/rejuvenation

---

## 1. Setup vs project vs step (the core hierarchy)

openAMUNDSEN-DA uses a layered structure:

1. **Setup** (top level)
2. **Project** (one DA experiment inside a setup)
3. **Step** (one assimilation window inside a project)
4. **Member** (one ensemble realization inside a step)

This hierarchy is reflected directly in the folder structure.

### Setup (top-level folder)

The setup contains shared inputs and the base openAMUNDSEN configuration.

For the tutorial case:

- setup folder: `/data/rofental`
- setup config: `/data/rofental/rofental.yml`

Typical setup contents:

- `rofental.yml` (openAMUNDSEN setup config)
- `env/` (ROI vectors, optional subdomain geometries)
- `grids/` (DEM, SVF, SRF, land-cover, ROI masks)
- `meteo/` (forcing station metadata and time series)
- `obs/` (raw observation rasters and station observations)
- `projects/` (one or more DA projects)

Reference YAML snippet (setup config, selected keys)

File path: `/data/rofental/rofental.yml`

```yaml
domain: "rofental"
resolution: 100
timestep: "3H"
crs: "epsg:25832"

input_data:
  grids:
    dir: grids
  meteo:
    format: csv
    crs: "epsg:25832"

output_data:
  timeseries:
    format: csv
    write_freq: D
  grids:
    format: netcdf
    compress: true
```

What to notice in this setup snippet:

- the setup YAML defines the model domain and generic openAMUNDSEN I/O behavior,
- it does not include DA-specific sections (`obs`, `data_assimilation`),
- those DA settings are added in the project YAML shown next.

Reference snippet (`/data/rofental/meteo/stations.csv`):

| id | name | x | y | alt |
| --- | --- | --- | --- | --- |
| bellavista | Bella Vista | 636823 | 5182569 | 2805 |
| proviantdepot | Proviantdepot | 639377 | 5187724 | 2659 |
| latschbloder | Latschbloder | 637854 | 5184641 | 2919 |

This station table is shown here to anchor the setup concept to a real shared input file (`meteo/`) that exists before any DA project is generated.

In this tutorial, the setup is the bundled `examples/rofental` case copied from the container image.

The setup YAML (`rofental.yml`) should remain pure openAMUNDSEN configuration. DA-specific configuration belongs to the project YAML.

### Project (DA experiment configuration)

A project defines one DA experiment within a setup:

- time range,
- observation mappings,
- assimilation events,
- likelihood/resampling/rejuvenation settings,
- DA output settings.

For the tutorial:

- project folder: `/data/rofental/projects/project_2022_2023`
- project YAML: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`

Key project-level sections in the tutorial example:

- `start_date`, `end_date`
- `obs.*` (observation directories, product tags, class mappings)
- `data_assimilation.*` (ensemble forcing, H(x), likelihood, resampling, rejuvenation, events, outputs)

Reference YAML snippet (project DA config, selected keys)

File path: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`

```yaml
start_date: "2022-10-01"
end_date: "2023-06-30"

obs:
  snowcover:
    dir: obs/snowcover
    product_tag: SNOWCOVER
  wetsnow:
    dir: obs/wetsnow
    product_tag: WETSNOW

data_assimilation:
  prior_forcing:
    ensemble_size: 10
  wet_snow:
    classification_threshold_percent: 0.5
  assimilation_events:
    - date: "2023-01-01"
      variable: scf
      product: SNOWCOVER
    - date: "2023-05-11"
      variable: wet_snow
      product: WETSNOW
```

What to notice in this project snippet:

- the project YAML adds DA-specific observation mappings and assimilation events,
- `assimilation_events` define the dates that later become step boundaries,
- this separation is why one setup can host multiple DA projects.

### Step (one assimilation window)

A step is a time segment between assimilation events (plus the initialization step).

Examples:

- `step_00_init`
- `step_01_YYYYMMDD-YYYYMMDD`
- ...

Steps are generated by `project_skeleton` from:

- project start/end dates
- `data_assimilation.assimilation_events`

Reference snippet (first generated step names):

```text
step_00_init
step_01_20230101-20230309
step_02_20230309-20230511
step_03_20230511-20230526
step_04_20230526-20230616
step_05_20230616-20230630
```

This naming pattern is the concrete result of your configured `assimilation_events`: change the event dates, and the generated step windows change as well.

{: .warning }
> If you change `assimilation_events`, the step structure changes. Regenerate the step skeleton and per-step observation CSVs before running the project again.

### Member (ensemble realization)

Within each step, the framework works with:

- `open_loop` (baseline, no DA updates)
- `member_001`, `member_002`, ... (ensemble realizations)

These live under step ensemble folders (e.g. `ensembles/prior/`, `ensembles/posterior/`).

<details markdown="block">
  <summary>Why this hierarchy is useful</summary>

- You can reuse one setup for multiple DA projects.
- You can inspect one step without parsing the entire run.
- You can compare members, open loop, and posterior effects explicitly.
- You can debug preprocessing separately from the DA pipeline.
</details>

---

## 2. Copy the tutorial setup (Rofental case study)

The tutorial setup copy is now part of [2. Dependencies]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %}),
where you start the interactive tutorial container shell and run:

```bash
cp -a /workspace/examples/rofental /data/rofental
```

The tutorial uses explicit paths in commands instead of shell variables so users always see where files are read and written.

### Quick structure check

```bash
find /data/rofental -maxdepth 2 -type d | sort
```

What you should see (at minimum):

- `/data/rofental/env`
- `/data/rofental/grids`
- `/data/rofental/meteo`
- `/data/rofental/obs`
- `/data/rofental/projects`

The example starts with raw SCF and wet-snow rasters. Summary CSVs are generated later in the preprocessing chapter.

---

## 3. How observations flow through the framework

This is the most important conceptual workflow for understanding preprocessing and DA.

![Observation preprocessing-to-DA flow diagram]({{ site.baseurl }}/assets/images/tutorial/diagrams/preprocessing-observation-flow.svg)

_Flow from raw SCF/wet-snow rasters to project summary CSVs, then to per-step observation CSVs consumed by each DA step._

Concrete example (SCF):

```text
Raw raster:
  /data/rofental/obs/snowcover/s2_fsc_snowflake_rofental_2023_01_01.tif

Summary row (project-level):
  /data/rofental/obs/summaries/project_2022_2023/scf_summary.csv
  date=2023-01-01, scf=1.0000, cloud_fraction=0.0

Per-step one-row CSV:
  /data/rofental/projects/project_2022_2023/steps/step_00_init/obs/obs_scf_SNOWCOVER_20230101.csv
```

### Why two preprocessing stages?

#### Stage A: Summary generation (`oa-da-snowcover`, `oa-da-wetsnow`)

Purpose:

- interpret product classes correctly
- clip to ROI
- apply land-cover exclusions
- compute availability/quality metrics
- create a stable project-level observation table

Default output location for these summary commands is `obs/summaries/<project_label>/` (tutorial example: `obs/summaries/project_2022_2023/`). Later preprocessing commands read from those summary CSVs.

#### Stage B: Per-step obs generation (`oa-da-scf`, `oa-da-wetsnow-project`)

Purpose:

- align summaries with `assimilation_events`
- validate dates and step windows
- generate the exact one-row CSVs consumed during DA execution

The DA run becomes reproducible once the per-step observation CSVs exist. These are the actual observation inputs used by the pipeline.

<details markdown="block">
  <summary>Why the framework does not read raw rasters directly during DA</summary>

Reading raw rasters inside the DA loop would mix:

- heavy geospatial preprocessing,
- class interpretation,
- and the DA runtime logic.

By separating preprocessing, the framework keeps the DA pipeline simpler, faster to
debug, and easier to validate.
</details>

---

## 4. Prior, posterior, and `open_loop`

These terms are central to understanding the step-wise DA cycle.

### `open_loop`

`open_loop` is the unperturbed baseline simulation:

- no ensemble perturbation resampling logic applied as a particle member,
- no DA updates,
- used as a reference in plots and DA output products.

Why it matters:

- baseline for interpretation (`increment = ens_mean - open_loop`)
- helps quantify what DA changed
- useful even when DA performs poorly (it is the comparison anchor)

### Prior ensemble

The **prior** is the ensemble state **before** assimilating the current step observation.

It is generated from:

- forcing perturbations (initially),
- and later from posterior-to-prior handover (rejuvenation) between steps.

### Posterior ensemble

The **posterior** is the ensemble state **after** assimilating the observation and applying resampling.

This posterior then seeds the next step (via rejuvenation) to form the next prior.

Think of each step as a loop: prior -> observe -> weight -> resample -> posterior -> rejuvenate -> next prior.

{: .references }
> - [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) for method terminology and theory context
> - [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) for how likelihood/resampling settings control this cycle

---

## 5. The DA cycle inside one step

The project pipeline performs a repeated DA loop across steps.

### Simplified step sequence

1. Run the prior ensemble for the step (`open_loop` + members)
2. Derive model-side diagnostics / observation-space summaries (e.g. SCF, wet-snow fractions)
3. Read the prepared observation CSV for the step event date
4. Compute likelihood and particle weights
5. Compute ESS (effective sample size)
6. Resample if ESS threshold criteria are met
7. Rejuvenate posterior members into the next step prior
8. Continue to next step

### Final step behavior

The final step is special:

- there is no next DA event after it,
- so the pipeline does not perform the full assimilation->next-step handover cycle.

You will see this in the log as a final-step message indicating that assimilation/resample/rejuvenate is skipped.

<details markdown="block">
  <summary>What ESS means in practice (brief intuition)</summary>

ESS measures how concentrated the particle weights are:

- high ESS: many particles still contribute
- low ESS: only a few particles dominate

Low ESS often triggers resampling, but ESS alone is not a complete quality score. It must
be interpreted together with the variable, observation coverage, and model/obs mismatch.
</details>

---

## 6. What gets written during a project run

During and after a run, the framework writes outputs at different levels:

### Step/member-level outputs (internal runtime products)

- member result files (grids, time series)
- DA intermediate artifacts
- weights / diagnostics by step

These are useful for debugging but can become large.

### Project-level diagnostics and plots (primary tutorial outputs)

- `plots/assim/` (weights, ESS)
- `plots/perf/` (runtime/performance)
- `plots/results/` (fractions, station plots, envelopes)
- `point_*_envelope.csv` (ROI envelopes)

### DA summary NetCDF output

- `results/grids/da_output_grids.nc`

This provides an analysis-friendly summary output for selected variables/metrics.

The tutorial example in this tutorial revision uses `retention: full`, so the summary NetCDF is written and member grid artifacts are retained as well.

{: .references }
> - [Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}) for concrete examples of these outputs
> - [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) for DA output variable/metric selection

---

## 7. Run modes (single-domain vs sub-domain)

openAMUNDSEN-DA supports different execution modes:

### Single-domain mode

- one domain / one setup ROI
- one DA project run over the full setup domain
- this is the mode used in this tutorial

### Sub-domain mode

- the ROI is split into multiple subdomains
- subdomain runs can be orchestrated separately/collectively
- useful for scaling and domain decomposition workflows

The tutorial uses **single-domain mode** to keep the learning path focused. Subdomain workflows build on the same concepts but add orchestration complexity.

---

## 8. Framework checklist before you continue

{: .checks }
> Confirm these points before starting preprocessing and project execution.

Before moving to preprocessing, you should be able to explain:

1. the difference between setup, project, step, and member
2. why observations are preprocessed into summaries and per-step CSVs
3. the roles of `open_loop`, prior, and posterior
4. how `assimilation_events` shape the step structure

<details markdown="block">
  <summary>Quick self-check (optional)</summary>

If you edit a wet-snow assimilation date in `assimilation_events`, which components need
to be regenerated before rerunning the project?

Expected answer (conceptually):

- step skeleton
- per-step observation CSVs (at least the affected variable, usually rerun both preprocessors for consistency)
- then the project run
</details>

---

## Next step

{: .references }
> Continue to preprocessing after the setup/project/step/member hierarchy is clear.

Continue with [5. Pre-processing]({{ site.baseurl }}{% link Tutorial/05-pre-processing.md %}) to generate:

- project-level observation summaries from raw rasters,
- the project step skeleton,
- and the per-step observation CSVs used by the DA pipeline.
