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

{: .highlight }

> Goal: after this tutorial, you should be able to run the bundled example and adapt the workflow to your own project.

---

## Step-by-step flow on this page

Use this page as the tutorial entry point:

1. understand what openAMUNDSEN-DA is (and how it relates to openAMUNDSEN)
2. understand what the tutorial covers and what it does not cover
3. review the Rofental case-study baseline used throughout the tutorial
4. use the overview figure (`a)`-`e)`) as a roadmap for later chapters
5. continue with [2. Dependencies]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %})

{: .note }
> This page is orientation-first. You do not need to run commands yet.

---

## openAMUNDSEN and openAMUNDSEN-DA

**openAMUNDSEN-DA** is a lightweight open-source framework for ensemble-based snow
modelling and particle-filter data assimilation built around **openAMUNDSEN**.

It combines:

- the distributed snow-hydrological model **openAMUNDSEN** (core simulation),
- ensemble forcing perturbations,
- observation preprocessing,
- particle-filter assimilation (e.g. SCF and wet snow),
- diagnostics, plotting, and compact DA output products.

In practice, many users run openAMUNDSEN-DA via the provided Docker image.

{: .note }

> openAMUNDSEN-DA is designed to be operationally practical (Docker-friendly) while still keeping intermediate products and diagnostics transparent for debugging and research workflows.

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
- result inspection (tables, plots, compact DA grids),
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

## Visual overview (read this once before the hands-on chapters)

This schematic combines the most important concepts of the framework in one figure:
framework architecture, particle-filter DA cycle, run modes, and the role of different
snow observations across the season.

![openAMUNDSEN-DA framework overview and workflow scheme]({{ site.baseurl }}/assets/images/tutorial/scheme_oa_da.png)

_Integrated overview of framework structure, particle-filter cycle, run modes, and snow-observation types._

### How to read the schematic (`a` to `e`)

{: .note }

> This figure is a conceptual map. The next tutorial chapters show where each concept appears in the actual folder structure, YAML files, and CLI commands.

1. **(a) Schematic overview**
   Shows the central DA idea in one line:
   an **openAMUNDSEN model ensemble** is propagated forward, compared to an **observation**, and updated from a **prior** ensemble to a **posterior** ensemble.
   In practice, this is what happens inside each assimilation step in a project
   ([Tutorial workflow]({{ site.baseurl }}{% link Tutorial/03-workflow.md %}),
   [Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}),
   [DA methods]({{ site.baseurl }}{% link reference/da-methods.md %})).
2. **(b) Open-source software**
   Shows the software relationship:
   `openAMUNDSEN` is the fully distributed snow-hydrological model, and `openAMUNDSEN-DA` wraps and extends it with ensemble handling, observation preprocessing, assimilation logic, and diagnostics.
   ([Configuration guide]({{ site.baseurl }}{% link guides/configuration.md %}),
   [Project structure]({{ site.baseurl }}{% link project-structure.md %}),
   [Package structure]({{ site.baseurl }}{% link reference/package-structure.md %})).
3. **(c) Data assimilation based on a particle filter**
   Shows the particle-filter cycle used by the framework:
   **forcing perturbation -> prior ensemble -> importance weighting -> resampling -> rejuvenation -> next prior**.
   This panel maps directly to the project pipeline stages and to the output diagnostics you will inspect later (weights, ESS, posterior spread)
   ([Running the project]({{ site.baseurl }}{% link Tutorial/06-running-the-project.md %}),
   [Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}),
   [DA methods]({{ site.baseurl }}{% link reference/da-methods.md %})).
4. **(d) Spatial domain**
   Shows the two execution modes supported by the framework:
   **single-domain mode** (one model domain, used in the tutorial) and
   **sub-domain mode** (large-area decomposition into tiles/subdomains).
   The same DA concepts apply in both modes; only orchestration and file layout differ
   ([Workflow]({{ site.baseurl }}{% link workflow.md %}),
   [Sub-domain mode docs]({{ site.baseurl }}{% link guides/cli.md %}#oa-da-subdomain),
   [Project structure]({{ site.baseurl }}{% link project-structure.md %})).
5. **(e) Snow data**
   Shows why multiple observation types are useful and when they are most informative in a season:
   **snow depth**, **snow cover**, and **wet snow** contribute information at different times (early/high/late season).
   In the tutorial project, you will preprocess and use **snow cover (FSC)** and **wet snow** raster observations, and validate against station snow measurements
   ([Pre-processing]({{ site.baseurl }}{% link Tutorial/05-pre-processing.md %}),
   [Observations guide]({{ site.baseurl }}{% link guides/observations.md %}),
   [Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %})).

---

## Central case study: Rofental (tutorial baseline)

The tutorial uses the bundled `examples/rofental` setup as the central case study.

Current tutorial baseline:

- grid resolution: **100 m**
- ensemble size: **10**
- project period: **October to June** (snow season focus)

This baseline is intended to stay feasible on a normal computer while still showing the
full preprocessing + DA + diagnostics workflow.

{: .highlight }

> Tutorial screenshots, snippets, and expected outputs should be based on this baseline to keep the documentation consistent.

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

{: .note }

> This tutorial uses **single-domain mode** so the learning path stays focused on the core framework concepts and preprocessing workflow.

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

You will see cross-references to the detailed documentation pages. These are the most
important ones:

- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %})
- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %})
- [Observation Processing Guide]({{ site.baseurl }}{% link guides/observations.md %})
- [Workflow Guide]({{ site.baseurl }}{% link workflow.md %})
- [Advanced Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %})
- [Advanced Performance]({{ site.baseurl }}{% link advanced/performance.md %})

{: .highlight }

> Tutorial pages explain the workflow. Reference pages explain the full option/configuration details.

---

## Next step

Continue with [2. Dependencies]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %}) to prepare the runtime environment (Docker, resource expectations, and command style used in this tutorial).
