---
layout: default
title: Tutorial
nav_order: 2
has_children: true
has_toc: false
---

# Tutorial
{: .no_toc }

## Overview

Step-by-step walkthrough to learn about openAMUNDSEN-DA. This tutorial guides you through the core capabilities of this data assimilation framework. openAMUNDSEN-DA is applied in a case study that will be processed step by step.
{: .fs-6 .fw-300 }

Use the docs search field (top navigation / sidebar area) to find information across the **entire documentation**, not only inside the tutorial. This is especially useful for jumping quickly to CLI options, configuration keys, and theory/reference pages while following the tutorial.

---

## Scope

The tutorial sequence covers:

- prerequisites and Docker runtime setup,
- workflow and framework concepts (setup/project/step/member, DA cycle),
- preparing the Rofental example (central case study),
- end-to-end observation preprocessing from raw raster inputs,
- running the project (granular command context + project pipeline),
- inspecting results, diagnostics, plots, tables, and output grids,
- adapting the example workflow to your own project.

The central case study is the bundled `examples/rofental` setup.

Use the chapter order from 1 to 8.

---

## Tutorial Chapters
{: .no_toc }

- [1. openAMUNDSEN-DA]({{ site.baseurl }}{% link Tutorial/01-openamundsen-da.md %})
- [2. Dependencies]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %})
- [3. Workflow]({{ site.baseurl }}{% link Tutorial/03-workflow.md %})
- [4. Framework]({{ site.baseurl }}{% link Tutorial/04-framework.md %})
- [5. Pre-processing]({{ site.baseurl }}{% link Tutorial/05-pre-processing.md %})
- [6. Running the project]({{ site.baseurl }}{% link Tutorial/06-running-the-project.md %})
- [7. Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %})
- [8. Adapting to your own project]({{ site.baseurl }}{% link Tutorial/08-adapting-to-your-own-project.md %})
