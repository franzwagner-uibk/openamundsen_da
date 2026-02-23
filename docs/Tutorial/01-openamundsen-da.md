---
layout: default
title: 1. openAMUNDSEN-DA
parent: Tutorial
nav_order: 1
permalink: /tutorial/openamundsen-da/
---

# 1. openAMUNDSEN-DA

This chapter introduces the framework and defines what this tutorial is trying to teach.

The tutorial is intentionally long-form and practical:

- you will preprocess observations,
- build DA-ready project inputs,
- run a full DA project,
- and inspect diagnostics and outputs in detail.

{: .highlight }
> Goal: after this tutorial, you should be able to run the bundled example and adapt the workflow to your own project.

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

- Documentation: [https://doc.openamundsen.org/](https://doc.openamundsen.org/)
- GitHub: [https://github.com/openamundsen/openamundsen](https://github.com/openamundsen/openamundsen)

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

