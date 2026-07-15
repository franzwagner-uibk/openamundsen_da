---
layout: default
title: Package Structure
parent: Reference
nav_order: 1
---

# Package Structure

High-level map of the Python package.

## Top-level packages
- `openamundsen_da.core` - config merge, env helpers, runner/launcher, prior forcing, constants.
- `openamundsen_da.pipeline` - project orchestration (`project`), project skeleton scaffolding (`project_skeleton`), cleanup (`cleanup`).
- `openamundsen_da.observer` - observation preprocessing and summary tools.
- `openamundsen_da.methods` - assimilation methods, H(x), wet-snow tools, visualization.
- `openamundsen_da.io` - path and file discovery helpers.
- `openamundsen_da.util` - stats, time series, validation, landcover, ROI, parallel helpers.
- `openamundsen_da.subdomain` - internal staged DA and plain-model subdomain preparation, execution, mosaic, rendering and status modules exposed through `openamundsen-da subdomains ...`.

## CLI entry points
`pyproject.toml` declares one `openamundsen-da` command with nested workflow commands.

## Conventions
- Setup YAML (`<setup-name>.yml`, template fallback `setup.yml`) contains setup-wide openAMUNDSEN settings.
- Project YAML (`<project-name>.yml`, fallback `project.yml`) contains data assimilation settings and project time span.
- Steps live under `projects/project_*/steps/step_*`.
- Members live under `.../ensembles/{prior,posterior}/member_*`.


