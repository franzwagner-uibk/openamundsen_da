---
layout: default
title: 3. Workflow
parent: Tutorial
nav_order: 3
permalink: /tutorial/workflow/
---

# 3. Workflow

This chapter gives the **big-picture workflow** of openAMUNDSEN-DA before we execute
commands. The goal is to understand what the framework is doing and why the tutorial is
structured in this order.

If you skip this mental model, the later commands can feel like a black box.

{: .highlight }
> Read this chapter as the workflow map. The later chapters are the executable path through this map.

---

## What this tutorial is doing (in one sentence)

We run a **reproducible Rofental data assimilation project** where:

- openAMUNDSEN simulates snow states for an ensemble,
- satellite observations (SCF and wet snow) are preprocessed into DA-ready inputs,
- the framework assimilates these observations step by step,
- and we inspect diagnostics, plots, and compact DA outputs.

---

## The workflow phases (high-level)

The tutorial follows these phases:

1. **Prepare the example setup**
2. **Understand the framework structure**
3. **Preprocess observations**
4. **Run the DA project**
5. **Inspect results and diagnostics**
6. **Adapt the workflow to your own project**

This sequence is intentional:

- preprocessing depends on project configuration,
- the project run depends on preprocessing outputs,
- diagnostics only make sense after a validated run.

{: .note }
> The tutorial order follows a dependency chain. Skipping steps usually causes downstream errors that look unrelated at first.

---

## Core framework objects (mental model)

These terms appear throughout the tutorial and in logs/files:

- **Setup**: top-level folder with openAMUNDSEN config and input data (`rofental.yml`, `grids/`, `meteo/`, `obs/`, `env/`)
- **Project**: DA configuration unit under `projects/project_YYYY_YYYY/`
- **Step**: one assimilation window (time segment between two DA events)
- **Member**: one ensemble realization (plus `open_loop`)
- **Observation summary**: project-level table (`scf_summary.csv`, `wet_snow_summary.csv`) derived from raw rasters
- **Per-step observation CSV**: one-row input consumed by the DA step at a configured event date

{: .note }
> Cross-reference:
> - [Framework chapter]({{ site.baseurl }}{% link Tutorial/04-framework.md %})

{: .highlight }
> Key idea: the DA step consumes **prepared per-step CSV observations**, not raw raster imagery directly.

---

## Workflow diagram (conceptual)

The operational workflow looks like this:

```text
Raw observation rasters (SCF / wet snow)
        |
        v
Project-level summaries (scf_summary.csv / wet_snow_summary.csv)
        |
        v
Per-step observation CSVs (aligned to assimilation_events)
        |
        v
Step-wise DA project pipeline
  - prior ensemble run
  - diagnostics / H(x)
  - assimilation (weights)
  - resampling
  - rejuvenation
  - next step
        |
        v
Plots, CSV summaries, diagnostics, compact DA NetCDF
```

This separation is one of the key design strengths of the framework:

- raw observation processing is explicit and reusable,
- DA execution is deterministic with respect to the prepared CSV inputs,
- debugging is easier because each stage has clear outputs.

---

## Manual commands vs project pipeline

The tutorial will show **both**:

### 1. Manual building blocks (for understanding/debugging)

You will see the individual commands for:

- project skeleton generation,
- observation summary generation,
- per-step observation CSV generation,
- selected lower-level runtime components.

This is important for understanding how the framework works internally.

### 2. Project pipeline (for actual execution)

For the main run, we use:

- `python -m openamundsen_da.pipeline.project`

This is the recommended operational workflow for end users because it orchestrates the
full step-by-step DA process consistently.

Rule of thumb:

- use **manual commands** to understand and debug,
- use the **pipeline** to run projects reliably.

<details markdown="block">
  <summary>Why the tutorial shows both manual commands and the pipeline</summary>

Manual commands make the framework transparent:

- where files are written,
- which configuration sections matter,
- where validation checks happen.

The project pipeline is then easier to trust and debug because you already understand
its building blocks.
</details>

---

## What can go wrong if the workflow order is wrong?

Typical mistakes (and why this chapter exists):

### Running the project before preprocessing

Result:

- missing per-step observation CSVs
- DA steps cannot find required observation inputs

### Editing `assimilation_events` without rebuilding steps

Result:

- step windows no longer match event dates
- per-step preprocessing fails (correctly) with a fail-fast error

### Changing observation class mapping after generating summaries

Result:

- summaries may no longer represent the intended observation interpretation
- regenerate summaries and per-step obs files after class changes

{: .warning }
> Class mapping changes require regeneration of derived observation files. Do not continue with stale summaries.

---

## How the Rofental tutorial case fits this workflow

The tutorial uses the bundled `examples/rofental` setup as a central case study with:

- 100 m grid configuration
- ensemble size 10
- October-June project period
- raw SCF and wet-snow rasters included in the example

This setup is intended to be:

- realistic enough to show the full framework behavior,
- but still feasible to run on a normal computer.

{: .note }
> This baseline should also be the reference for tutorial screenshots/snippets to keep all pages consistent.

---

## What you should understand before moving on

Before continuing to the framework chapter, make sure these points are clear:

1. Raw rasters are **not** used directly by the DA step; they are summarized first.
2. The project pipeline consumes **per-step CSV inputs**, not raw observation imagery.
3. `assimilation_events` define the temporal structure of the DA project (steps).
4. The pipeline automates many tasks, but the outputs remain inspectable at each stage.

<details markdown="block">
  <summary>Quick self-check (optional)</summary>

If you can explain these two points in your own words, you are ready for the next chapter:

- Why are summary CSVs needed before the DA run?
- Why can changing `assimilation_events` affect both preprocessing and runtime behavior?
</details>

---

## Next step

Continue with [4. Framework]({{ site.baseurl }}{% link Tutorial/04-framework.md %}) for a deeper
look at:

- setup/project separation,
- step structure and temporal segmentation,
- ensemble members and the open loop,
- and the DA cycle inside a project run.
