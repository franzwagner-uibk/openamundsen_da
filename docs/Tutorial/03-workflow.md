---
layout: default
title: 3. Workflow
parent: Tutorial
nav_order: 3
permalink: /tutorial/workflow/
---

# 3. Workflow

This tutorial follows the standard openAMUNDSEN-DA project workflow:

1. Prepare a setup folder (copy a tested example).
2. Build the project skeleton (`step_*` folders from assimilation events).
3. Prepare per-step observation CSVs from project summaries.
4. Run the full project pipeline (open loop + ensemble + DA).
5. Review diagnostics and result products.

Conceptually:

- **Setup**: openAMUNDSEN-level configuration and input data (`setup.yml`, `grids`, `meteo`, `env`, `obs`)
- **Project**: DA-specific configuration and time span (`project_YYYY_YYYY.yml`)
- **Step**: one assimilation window
- **Member**: one ensemble realization plus `open_loop` baseline

The next tutorial chapters execute this workflow directly on the Rofental example.
