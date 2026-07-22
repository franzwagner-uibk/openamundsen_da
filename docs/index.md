---
layout: default
title: Home
nav_order: 1
description: "Technical documentation for openAMUNDSEN-DA"
permalink: /
---

# openAMUNDSEN-DA

{: .fs-9 }

Assimilating satellite-based snow cover, wet snow extent and station observations
into openAMUNDSEN.
{: .fs-6 .fw-300 }

[How to Use]({{ '/tutorial/' | relative_url }}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Installation]({{ site.baseurl }}{% link installation.md %}){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/openamundsen/openamundsen-da){: .btn .fs-5 .mb-4 .mb-md-0 }

## Software scope

openAMUNDSEN-DA is an open-source data assimilation framework built around
[openAMUNDSEN](https://doc.openamundsen.org/). It preprocesses configured snow
observations, prepares deterministic assimilation sequences, executes an
ensemble particle-filter workflow and validates the resulting tables, grids,
plots, maps and report.

The documentation is deliberately technical. For the particle-filter
formulation, scientific motivation and further interpretation of the results,
see Wagner et al. (2026), the manuscript accompanying this software release.

![openAMUNDSEN-DA technical workflow from mounted setup inputs through observation summaries and prepared steps to validated results]({{ site.baseurl }}/assets/images/diagrams/openamundsen-da-workflow.svg)

*The host setup is mounted at `/data`. openAMUNDSEN-DA summarizes configured snow
observations, prepares inspectable event inputs and writes a validated result set.*

## Start here

- [Installation]({{ site.baseurl }}{% link installation.md %}) explains the Python
  package and wheel-based Docker image.
- [Input Data]({{ site.baseurl }}{% link guides/observations.md %}) defines required
  model, forcing and observation inputs without prescribing product-generation algorithms.
- [Configuration]({{ site.baseurl }}{% link guides/configuration.md %}) documents the
  strict setup, project and step ownership boundary.
- [Running]({{ site.baseurl }}{% link running.md %}) covers the supported single-domain
  and subdomain command sequences.
- [Output Data]({{ site.baseurl }}{% link output-data.md %}) defines the manifest,
  compact NetCDF, diagnostics and cleanup contract.
- [Example Data]({{ site.baseurl }}{% link example-data.md %}) describes the shipped
  Rofental and North Tyrol examples.
- [How to Use]({{ '/tutorial/' | relative_url }}) is the reviewed, continuous
  Rofental walkthrough.

## License and availability

The stable v0.9.0 Python package is available from
[PyPI](https://pypi.org/project/openamundsen-da/). The tested multi-architecture
container is available from
[GHCR](https://github.com/franzwagner-uibk/openamundsen_da/pkgs/container/openamundsen_da),
and release archives and evidence are available from
[GitHub Releases](https://github.com/openamundsen/openamundsen-da/releases).
The MIT License permits commercial use subject to its terms.
