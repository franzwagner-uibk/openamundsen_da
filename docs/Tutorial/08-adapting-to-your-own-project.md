---
layout: default
title: 8. Adapting the example to your own project
parent: Tutorial
nav_order: 8
permalink: /tutorial/adapting-to-your-own-project/
---

# 8. Adapting the example to your own project

This chapter explains how to use the Rofental tutorial workflow as a template for a new
domain and project.

The recommended strategy is:

- start from a **working example**,
- change one layer at a time,
- keep the preprocessing and validation loop tight,
- avoid changing many configuration dimensions simultaneously.

{: .highlight }
> The safest path is to preserve the workflow structure from the tutorial and only replace data/configuration step by step.

---

## Step-by-step flow on this page

{: .step }
> Treat this chapter as a migration checklist: adapt a few things first, then expand gradually.

This chapter is a **migration guide**, not a command-by-command execution page.

Recommended use:

1. read `What you must adapt first`
2. follow `Recommended migration workflow`
3. use `Common adaptation mistakes` as a review checklist before running
4. use the final checklist when creating a new project baseline

{: .note }
> Keep the tutorial Rofental setup as a known-good reference while adapting your own domain. Compare structure and outputs frequently instead of changing everything at once.

---

## What to copy from the tutorial workflow (unchanged at first)

Keep these concepts and steps unchanged initially:

1. setup/project/step directory structure
2. project skeleton generation
3. observation preprocessing sequence:
   - summary generation (`oa-da-snowcover`, `oa-da-wetsnow`)
   - per-step obs creation (`oa-da-scf`, `oa-da-wetsnow-project`)
4. project pipeline execution
5. output inspection workflow (logs -> DA diagnostics -> plots -> NetCDF)

This gives you a stable debugging baseline.

<details markdown="block">
  <summary>Why a stable reference run is so important</summary>

When a new project fails, you need a known-good baseline to compare:

- file structure
- preprocessing outputs
- DA diagnostics
- runtime behavior

The tutorial Rofental run provides that baseline.
</details>

---

## What you must adapt first (minimum required changes)

### 1. Setup-level model configuration (`<setup>.yml`)

This remains **pure openAMUNDSEN config** (no DA blocks).

Typical changes:

- domain definition
- grid resolution
- DEM / terrain inputs
- forcing station metadata and paths
- model physics settings

{: .references }
> - openAMUNDSEN documentation (external): [https://doc.openamundsen.org/](https://doc.openamundsen.org/)

### 2. Project-level DA configuration (`project_YYYY_YYYY.yml`)

This is where DA-specific behavior is configured.

Typical changes:

- `start_date`, `end_date`
- `obs.*` paths, product tags, class mappings
- `data_assimilation.assimilation_events`
- likelihood settings (`obs_sigma`, etc.)
- resampling / rejuvenation settings
- output retention/grid export settings

{: .references }
> - [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %})

### 3. Observation data and class mappings

This is one of the most important parts when switching projects/products.

You must ensure:

- the raw raster product class codes are known,
- class mapping in project YAML matches the actual product,
- product tags in `assimilation_events` match the configured tags exactly,
- selected assimilation dates exist in the generated summary CSVs.

{: .warning }
> Do not rely on guessed defaults. The framework is designed to fail fast when required
> observation configuration is missing. This is intentional and helps prevent silent mistakes.

{: .note }
> This is especially important for class mappings and product tags. Explicit configuration is safer than fallback behavior.

---

## Recommended migration workflow (step-by-step)

### Phase A: Clone and trim

1. Copy the Rofental example to a new setup directory
2. Replace/prepare the setup-level openAMUNDSEN inputs
3. Keep the project YAML simple at first (few DA events)

Why:

- you want the framework behavior to stay familiar while changing data sources

{: .highlight }
> Clone first, then trim. Do not redesign the whole setup structure during the first migration step.

### Phase B: Validate observations before DA

1. Run `oa-da-snowcover` and/or `oa-da-wetsnow`
2. Inspect summary CSVs (`scf_summary.csv`, `wet_snow_summary.csv`)
3. Select a small set of good assimilation dates
4. Update `assimilation_events`
5. Run `project_skeleton`, `oa-da-scf`, `oa-da-wetsnow-project`

Why:

- DA debugging is much easier when observation coverage and quality are already verified

### Phase C: Run a small/cheap DA test first

Start with a cheaper configuration:

- coarser resolution (if possible),
- fewer assimilation events,
- smaller ensemble (e.g. 5-10) for debugging,
- shorter date range (snow season subset).

Then inspect:

- logs,
- ESS behavior,
- result plots,
- output grid file.

Only after this works reliably should you increase complexity/cost.

{: .note }
> A cheap debugging run is for validating wiring and workflow correctness, not for final scientific conclusions.

---

## Choosing assimilation dates (practical guidance)

Use the summary CSVs, not visual guessing alone.

Selection criteria typically include:

- date coverage inside your project period
- acceptable cloud fraction (for FSC)
- sufficient valid pixel count
- clear signal relevance for the variable (e.g. wet-snow onset and melt-season phases)

<details markdown="block">
  <summary>Practical date-selection workflow</summary>

1. Generate summaries from raw observations
2. Inspect cloud fraction / valid pixel count (SCF)
3. Inspect wet-snow coverage dates
4. Choose a small set of events across the season
5. Update `assimilation_events`
6. Rebuild step skeleton and per-step observation CSVs
</details>

Recommended tutorial-like strategy:

- a few SCF dates covering early / peak / late season
- a few wet-snow dates during melt onset and progression
- keep the initial event count small enough to debug quickly

{: .references }
> - [Observation Processing Guide]({{ site.baseurl }}{% link guides/observations.md %})

---

## Common adaptation mistakes (and how to avoid them)

{: .warning }
> Check this section whenever a new project behaves unexpectedly after configuration changes.

### 1. Mixing setup-level and project-level configuration

Problem:

- DA settings are accidentally placed in the setup YAML.

Rule:

- setup YAML = openAMUNDSEN only
- project YAML = openAMUNDSEN-DA additions (`obs`, `data_assimilation`, dates)

### 2. Changing many parameters at once

Problem:

- impossible to tell whether failures come from observations, forcing, geometry, or DA tuning.

Fix:

- change one category at a time and rerun.

### 3. Using incompatible observation class mappings

Problem:

- summary values look wrong or missing because classes are interpreted incorrectly.

Fix:

- verify product class definitions and update YAML explicitly.

### 4. Forgetting to rebuild steps after editing `assimilation_events`

Problem:

- per-step observation creation fails or writes into mismatched step windows.

Fix:

- rerun `project_skeleton --overwrite` before per-step preprocessing.

### 5. Interpreting plots without checking units

Problem:

- apparent mismatch is caused by units (e.g. SWE in m vs mm).

Fix:

- verify units in observation files and plotting expectations first.

---

## Minimal checklist for a new project (copy/paste audit list)

{: .checks }
> Use this as the final pre-run audit before starting a new project experiment.

Before your first real DA run, confirm:

- setup YAML is valid openAMUNDSEN config (no DA block)
- project YAML contains `start_date`, `end_date`, `obs`, and `data_assimilation`
- observation class mappings are explicitly configured
- product tags are explicitly configured and match `assimilation_events`
- summary CSVs were generated successfully
- `assimilation_events` dates exist in the summary CSVs
- project skeleton was rebuilt after finalizing event dates
- per-step obs CSVs were generated successfully
- a short test run completes and produces diagnostics/plots/NetCDF outputs

{: .highlight }
> This checklist is the minimum quality gate before spending time on parameter tuning.

---

## Suggested next experiments after the tutorial

Once your first project works, these are good controlled experiments:

1. Change `ensemble_size` and compare ESS/runtime
2. Change grid resolution and compare runtime/spatial detail
3. Tune `obs_sigma` for SCF and wet snow and compare ESS behavior
4. Adjust resampling thresholds and compare particle diversity
5. Add/remove assimilation dates and compare impact on seasonal trajectories

Document each change and compare outputs against the same validation plots.

---

## Where to go deeper

{: .references }
> Use these links after the tutorial to expand beyond the baseline workflow.

- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %})
- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %})
- [Observation Processing Guide]({{ site.baseurl }}{% link guides/observations.md %})
- [Workflow Guide]({{ site.baseurl }}{% link workflow.md %})
- [Advanced Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %})
- [Advanced Performance]({{ site.baseurl }}{% link advanced/performance.md %})
- openAMUNDSEN docs (external): [https://doc.openamundsen.org/](https://doc.openamundsen.org/)

---

## Tutorial status after this chapter

You should now be able to:

- run the bundled Rofental tutorial end to end,
- interpret the main DA outputs,
- and adapt the workflow structure to a new project with controlled changes.
