---
layout: default
title: 1. Overview
parent: How to Use
nav_order: 1
permalink: /tutorial/openamundsen-da/
---

# 1. Overview

This chapter gives the minimum conceptual background needed to use openAMUNDSEN-DA
without scattering the same explanations across later pages.

openAMUNDSEN-DA prepares snow observations and uses them to update an ensemble
of openAMUNDSEN model realizations in a step-wise data assimilation workflow.
The tutorial uses the single-domain Rofental example; subdomain execution is an
advanced workflow. It focuses on station snow-depth, fractional snow covered area
(fSCA) and wet-snow observations.

## The Workflow In One Paragraph

A setup contains the openAMUNDSEN model configuration, input grids, forcing, and
observation files. Inside that setup, a project YAML defines the time period,
observation mappings, data assimilation events, and data assimilation settings.
The shipped Rofental example already includes baseline fSCA (`scf`) and wet-snow summary
tables under `obs/summaries/`, while station observations and station data assimilation metadata
live under `obs/stations/`. Observation rasters can still be reprocessed into
project summaries and per-step observation CSVs. The project pipeline then runs
the ensemble step by step, assimilates the prepared observations, and writes
diagnostics and result products.

## Setup And Project Layout

The schematic below shows where the main files live inside one setup and one
project workspace. It separates setup-level inputs from the project-specific
configuration, step folders, and outputs.

![Setup and project layout showing setup-level inputs, project configuration, step folders, member folders, results, and plots]({{ site.baseurl }}/assets/images/diagrams/setup-project-structure-annotated.svg)

_Shared layout of a setup root and one project workspace. Optional sub-domain
folders are omitted for clarity._

## The Core Terms

These four terms are the important ones for the tutorial:

- `setup`: the top-level case directory with model inputs, observations, and projects
- `project`: one data assimilation configuration inside a setup
- `step`: one time window between two data assimilation events
- `member`: one ensemble realization inside a step

Inside the ensemble workflow, three more terms matter:

- `open_loop`: the unperturbed reference simulation
- `prior`: the ensemble state before assimilating the current observation
- `posterior`: the ensemble state after weighting/resampling the current observation

## How Observations Enter The Framework

The framework does not read observation rasters directly during the assimilation step.
Instead it uses a two-stage preprocessing workflow:

1. observation rasters are summarized to project-level tables such as `scf_summary.csv`
2. `openamundsen-da prepare` aligns the summaries with `assimilation_events` and
   writes one-row per-step observation CSVs under `steps/*/obs/`

That separation is deliberate. It keeps preprocessing explicit, makes event alignment
transparent, and gives you files that can be inspected before the model run starts.

Continue with [2. Installation]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %}).
