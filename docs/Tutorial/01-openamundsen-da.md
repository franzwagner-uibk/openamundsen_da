---
layout: default
title: 1. openAMUNDSEN-DA
parent: Tutorial
nav_order: 1
permalink: /tutorial/openamundsen-da/
---

# 1. openAMUNDSEN-DA

This chapter introduces the framework and defines what this tutorial is trying to teach.

The tutorial is intentionally detailed and practical:

- you will preprocess observations,
- build DA-ready project inputs,
- run a full DA project,
- and inspect diagnostics and outputs in detail.

Goal: after this tutorial, you should be able to run the bundled example and adapt the workflow to your own project.

---

## Step-by-step flow on this page

{: .step }

> Read this page top-to-bottom once to build the mental model before starting hands-on commands.

Use this page as the tutorial entry point:

1. understand what openAMUNDSEN-DA is (and how it relates to openAMUNDSEN)
2. understand what the tutorial covers and what it does not cover
3. review the Rofental case-study baseline used throughout the tutorial
4. use the overview figure (`a)`-`e)`) as a roadmap for later chapters
5. continue with [2. Dependencies]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %})

This page is orientation-first. You do not need to run commands yet.

---

## openAMUNDSEN and openAMUNDSEN-DA

**openAMUNDSEN-DA** is a lightweight open-source framework for ensemble-based snow
modelling and particle-filter data assimilation built around **openAMUNDSEN**.

It combines:

- the distributed snow-hydrological model **openAMUNDSEN** ,
- ensemble forcing perturbations,
- observation preprocessing,
- particle-filter assimilation (e.g. SCF and wet snow),
- diagnostics, plotting, and DA output products.

In practice, many users run openAMUNDSEN-DA via the provided Docker image.

{: .note }

>openAMUNDSEN-DA is designed to be operationally practical (Docker-friendly) while still keeping intermediate products and diagnostics transparent for debugging and research workflows. Users who focus on contributing to this open source project or adapting the code for their own use should clone the [GitHub repository](https://github.com/franzwagner-uibk/openamundsen_da).

For more information about **openAMUNDSEN**:

- Documentation: <https://doc.openamundsen.org/>
- GitHub: <https://github.com/openamundsen/openamundsen>

---

## What this tutorial covers

This tutorial is a full end-to-end walkthrough using the bundled **Rofental** example
as the central case study.

You will cover:

- framework concepts (setup / project / step / member),
- observation preprocessing from **raw satellite rasters**,
- project execution using the DA pipeline,
- DA diagnostics (weights, ESS, plots),
- result inspection (tables, plots, DA grids),
- and adaptation of the workflow to a new project.

{: .note }

> The tutorial is intentionally not just a command list. It explains why each step exists and what to check before moving on.

<details markdown="block">
  <summary>What this tutorial does not try to do (on purpose)</summary>

- It is not a full DA theory textbook.
- It does not document every CLI option inline (reference pages cover that).
- It does not cover every observation product in one workflow.
- It does not replace the full configuration reference.

The focus is a reproducible, understandable workflow that users can reuse.

</details>

---

## Visual overview

This schematic combines the most important concepts of the framework in one figure:
framework architecture, particle-filter DA cycle, run modes, and the role of different
snow observations across the season.

![openAMUNDSEN-DA framework overview and workflow scheme]({{ site.baseurl }}/assets/images/tutorial/scheme_oa_da.png)

*Integrated overview of framework structure, particle-filter cycle, run modes, and snow-observation types.*


**(a) Schematic overview**:
  openAMUNDSEN model ensemble is propagated forward, compared to an **observation**, and updated from        a **prior** ensemble to a **posterior** ensemble.
   This is the iterative assimilation step in a project
   ([Tutorial workflow]({{ site.baseurl }}{% link Tutorial/03-workflow.md %}),
   [Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}),
   [DA methods]({{ site.baseurl }}{% link reference/da-methods.md %})).

**(b) Open-source software**:
   `openAMUNDSEN` is the fully distributed snow-hydrological model, and `openAMUNDSEN-DA` wraps and extends it with ensemble handling, observation preprocessing, assimilation logic, and diagnostics.
   ([Configuration guide]({{ site.baseurl }}{% link guides/configuration.md %}),
   [Project structure]({{ site.baseurl }}{% link project-structure.md %}),
   [Package structure]({{ site.baseurl }}{% link reference/package-structure.md %})).

**(c) Data assimilation based on a particle filter**
   Shows the particle-filter cycle used by the framework:
   **forcing perturbation -> prior ensemble -> importance weighting -> resampling -> rejuvenation -> next prior**.
   This panel maps directly to the project pipeline stages and to the output diagnostics you will inspect later (weights, ESS, posterior spread)
   ([Running the project]({{ site.baseurl }}{% link Tutorial/06-running-the-project.md %}),
   [Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}),
   [DA methods]({{ site.baseurl }}{% link reference/da-methods.md %})).

**(d) Spatial domain**
   Shows the two execution modes supported by the framework:
   **single-domain mode** (one model domain, used in the tutorial) and
   **sub-domain mode** (large-area decomposition into tiles/subdomains).
   The same DA concepts apply in both modes ([Workflow]({{ site.baseurl }}{% link workflow.md %}),
   [Sub-domain mode docs]({{ site.baseurl }}{% link guides/cli.md %}#oa-da-subdomain),
   [Project structure]({{ site.baseurl }}{% link project-structure.md %})).

**(e) Snow data**
   Shows why multiple observation types are useful and when they are most informative in a season:
   **snow depth**, **snow cover**, and **wet snow** contribute information at different times (early/high/late season).
   In the tutorial project, you will preprocess and use **snow cover (FSC)** and **wet snow** raster observations, and validate against station snow measurements
   ([Pre-processing]({{ site.baseurl }}{% link Tutorial/05-pre-processing.md %}),
   [Observations guide]({{ site.baseurl }}{% link guides/observations.md %}),
   [Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %})).

---

## Central case study: Rofental

The tutorial uses the bundled `examples/rofental` setup as the central case study.

Rofental setup:

- grid resolution: **100 m**
- ensemble size: **10**
- project period: **October to June**

This baseline is intended to stay feasible on a normal computer while still showing the
full preprocessing + DA + diagnostics workflow.

### Rofental setup inputs used throughout the tutorial

The tutorial chapters refer back to the same bundled Rofental setup inputs:

- **Forcing inputs** in `meteo/` (meteorological station time series + station metadata)
- **Snow observations for DA** in `obs/` (snow cover / FSC and wet-snow products)
- **Station snow measurements for evaluation/diagnostics** (used later in results checks)

![Rofental tutorial setup map (domain, forcing stations, and land cover layer)]({{ site.baseurl }}/assets/images/tutorial/rofental_setup_map.png)

*Rofental tutorial setup map showing the model domain, forcing stations, and land cover throughout the tutorial.*

You will inspect concrete setup data files (including `meteo/stations.csv` and one forcing station CSV) in [2. Dependencies]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %}) right after copying the bundled example into your workspace.

---

## Run modes (high-level)

openAMUNDSEN-DA supports two general execution modes.

### Single-domain mode

- one domain / one ROI
- one project pipeline run for that domain
- simplest mode for learning and most example workflows

### Sub-domain mode

- ROI is split into multiple subdomains
- subdomain preparation and orchestration are added on top of the core workflow
- useful for scaling and large-domain strategies

This tutorial uses **single-domain mode** so the learning path stays focused on the core framework concepts and preprocessing workflow.

---

## How to use this tutorial effectively

Recommended approach:

1. Follow the chapters in order
2. Run the commands, do not just read them
3. Inspect the intermediate files and logs
4. Use the diagnostics chapter to validate your run before changing configuration

{: .warning }

> Skipping preprocessing or changing `assimilation_events` without regenerating dependent files is a common source of confusing errors later in the workflow.

---

## Links to deeper documentation (use throughout the tutorial)

{: .references }

> Use these links as background/reference material while working through later chapters.

You will see cross-references to the detailed documentation pages. These are the most
important ones:

- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %})
- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %})
- [Observation Processing Guide]({{ site.baseurl }}{% link guides/observations.md %})
- [Workflow Guide]({{ site.baseurl }}{% link workflow.md %})
- [Advanced Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %})
- [Advanced Performance]({{ site.baseurl }}{% link advanced/performance.md %})

Tutorial pages explain the workflow. Reference pages explain the full option/configuration details.

---

## Next step

{: .references }

> Continue directly to the dependencies/setup chapter after this overview.

Continue with [2. Dependencies]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %}) to prepare the runtime environment (Docker, resource expectations, and command style used in this tutorial).
