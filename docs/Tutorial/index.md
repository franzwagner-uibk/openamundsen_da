---
layout: default
title: How to Use
nav_order: 8
has_children: true
has_toc: false
permalink: /tutorial/
---

# How to Use
{: .no_toc }

A guided walkthrough of openAMUNDSEN-DA using the existing reviewed Rofental
example. Work through it once from start to finish: understand the framework,
start the pinned runtime, inspect the example, preprocess observations, prepare
the project, run the model and inspect the outputs.

For the scientific background to the openAMUNDSEN-DA software publication, it
is recommended to read Wagner et al. (2026).

## Chapter Order
{: .no_toc }

- [1. Overview]({{ site.baseurl }}{% link Tutorial/01-openamundsen-da.md %}) defines the workflow and core terms.
- [2. Installation]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %}) copies the example from the pinned image and starts the container.
- [3. Example Data: Rofental]({{ site.baseurl }}{% link Tutorial/03-example-data-rofental.md %}) introduces the prepared inputs and project configuration.
- [4. Preprocess Observations]({{ site.baseurl }}{% link Tutorial/05-pre-processing.md %}) reproduces the fSCA and wet-snow summary tables.
- [5. Prepare the Project]({{ site.baseurl }}{% link Tutorial/05-prepare-project.md %}) creates deterministic steps and per-event inputs.
- [6. Running the Model]({{ site.baseurl }}{% link Tutorial/06-running-the-project.md %}) executes and validates the complete project.
- [7. Results and Diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}) reviews the manifest, plots, maps, tables and compact grid result.
- [8. Adapting to Your Own Project]({{ site.baseurl }}{% link Tutorial/08-adapting-to-your-own-project.md %}) transfers the workflow with controlled changes.
