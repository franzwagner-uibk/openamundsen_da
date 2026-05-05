---
layout: default
title: Home
nav_order: 1
description: "openAMUNDSEN-DA - Data Assimilation Framework for openAMUNDSEN"
permalink: /
---

# openAMUNDSEN-DA

{: .fs-9 }

Data assimilation for snow observations in openAMUNDSEN
{: .fs-6 .fw-300 }

[How to Use]({{ '/tutorial/' | relative_url }}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Installation]({{ site.baseurl }}{% link installation.md %}){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/franzwagner-uibk/openamundsen_da){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What It Is

openAMUNDSEN-DA is an open-source framework for ensemble-based snow data assimilation with
[openAMUNDSEN](http://doc.openamundsen.org/). It combines model forcing perturbation,
observation preprocessing, particle-filter assimilation, diagnostics, and result export into
one reproducible workflow that runs on a normal workstation or in larger computing
environments.

The framework works with gridded snow-observation products such as snow-cover fraction and
wet-snow masks, and it can also use station observations for evaluation plots and diagnostics.

![openAMUNDSEN-DA framework overview and workflow scheme]({{ site.baseurl }}/assets/images/tutorial/scheme_oa_da.png)

*High-level overview of the framework: openAMUNDSEN provides the snow model, openAMUNDSEN-DA adds observation preprocessing, step-wise ensemble assimilation, and result diagnostics.*

## How It Works

In practical terms, the workflow has four stages:

1. prepare a setup with model grids, forcing, observation products, and one project YAML
2. preprocess observation rasters into project summaries and per-step observation CSVs
3. run the step-wise data assimilation pipeline
4. inspect diagnostics, plots, and compact NetCDF result products

If you only read one page before starting, this is the key idea:
openAMUNDSEN-DA does not assimilate raster products directly inside the model loop.
It first turns them into validated project summaries and per-step observation CSVs.
That separation keeps the workflow reproducible and easier to debug.

## What You Can Do With It

- run ensemble snow simulations with particle-filter data assimilation
- preprocess snow-cover and wet-snow observation products for assimilation
- generate or ingest observation uncertainty layers on a `0..100` scale
- inspect weights, ESS, performance metrics, plots, and compact output grids
- scale from a tutorial-sized single domain to larger sub-domain workflows

## Where To Start

Choose one path:

- [How to Use]({{ '/tutorial/' | relative_url }}): full guided walkthrough with the bundled Rofental example
- [Installation]({{ site.baseurl }}{% link installation.md %}): Docker-based setup and runtime basics
- [Workflow]({{ site.baseurl }}{% link workflow.md %}): conceptual overview of the data assimilation workflow
- [Configuration]({{ site.baseurl }}{% link guides/configuration.md %}): project and setup YAML reference
- [Observation Processing]({{ site.baseurl }}{% link guides/observations.md %}): product preprocessing and uncertainty handling
- [Sub-domain Runbook]({{ site.baseurl }}{% link guides/subdomain-runbook.md %}): end-to-end large-domain processing from Docker image pull to merged output

## Documentation Structure

- [How to Use]({{ '/tutorial/' | relative_url }})
- [Installation]({{ site.baseurl }}{% link installation.md %})
- [Project Structure]({{ site.baseurl }}{% link project-structure.md %})
- [Workflow]({{ site.baseurl }}{% link workflow.md %})
- [Configuration]({{ site.baseurl }}{% link guides/configuration.md %})
- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %})
- [Observation Processing]({{ site.baseurl }}{% link guides/observations.md %})
- [Sub-domain Runbook]({{ site.baseurl }}{% link guides/subdomain-runbook.md %})
- [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %})

## Current Status

This documentation is still under active development. Treat it as working technical
documentation for a moving code base, not as a frozen scientific reference.

## References

- Strasser, U., Warscher, M., Rottler, E., and Hanzer, F. (2024). openAMUNDSEN v1.0: an open-source snow-hydrological model for mountain regions. Geoscientific Model Development, 17, 6775-6797. https://doi.org/10.5194/gmd-17-6775-2024.
