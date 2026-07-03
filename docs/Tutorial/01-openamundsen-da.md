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

openAMUNDSEN-DA is built around a simple idea: use snow-observation products to update an
ensemble of openAMUNDSEN model realizations in a step-wise data assimilation workflow.
The tutorial uses the single-domain Rofental example and focuses on a mixed workflow with
station snow-depth, fractional snow covered area (fSCA), and wet-snow observations.

## The Workflow In One Paragraph

A setup contains the openAMUNDSEN model configuration, input grids, forcing, and
observation files. Inside that setup, a project YAML defines the time period,
observation mappings, data assimilation events, and data assimilation settings.
The shipped Rofental example already includes baseline fSCA (`scf`) and wet-snow summary
tables under `obs/summaries/`, while station observations and station DA metadata
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
2. these summaries are aligned with the configured `assimilation_events` and written as
   one-row per-step observation CSVs under `steps/*/obs/`

That separation is deliberate. It keeps preprocessing explicit, makes event alignment
transparent, and gives you files that can be inspected before the model run starts.

## What The Tutorial Covers

The tutorial is organized as a straight workflow:

1. install and start the Docker-based runtime environment
2. inspect the bundled Rofental example data
3. inspect or regenerate the shipped observation summaries and uncertainty inputs
4. run the project pipeline
5. inspect diagnostics and outputs
6. adapt the example workflow to another project

## Execution Mode In This Tutorial

openAMUNDSEN-DA supports both single-domain and sub-domain workflows, but this tutorial
uses the single-domain path only. That keeps the first run focused on the core
preprocessing and ensemble-assimilation workflow.

## What To Remember Before Continuing

If you keep only three points from this chapter, they should be these:

- the setup YAML stays pure openAMUNDSEN configuration; data assimilation settings live in the project YAML
- observation rasters are preprocessed before the project pipeline runs
- the project pipeline works step by step and consumes prepared per-step observation CSVs

Continue with [2. Installation]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %}).
