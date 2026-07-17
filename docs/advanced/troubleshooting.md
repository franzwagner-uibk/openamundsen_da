---
layout: default
title: Troubleshooting
parent: Advanced
nav_order: 4
---

# Troubleshooting

## Docker is unavailable

Run `docker run --rm hello-world`. On Linux, start the Docker service and make
sure the current login session has access to the Docker group. On Windows or
macOS, start Docker Desktop and verify its CPU, memory and disk allocation.

## Files are owned by root

On Linux, add `--user "$(id -u):$(id -g)"` to `docker run` when writing mounted
outputs. Repair an existing copied setup on the host with `chown`, then avoid
running the tutorial copy command through `sudo`.

## Configuration fails before execution

Read the complete error; validation reports the configuration path. Check that
the setup and project use canonical names, setup-relative paths stay inside the
setup, `output_data.grids.format` is supported and every observation product has
explicit format, classes, tag and summary configuration.

## Preparation cannot find an event

Confirm that each `assimilation_events` date exists in the corresponding summary
CSV and uses the same product tag. Rerun observation preprocessing only when the
source product/config changed, then rerun `openamundsen-da prepare --overwrite`.

## A run will not resume

Resume is intentionally hash-safe. If configuration or inputs differ from the
manifest, move the previous project to an archive and prepare a fresh project.
Do not edit a completed project in place.

## Outputs are incomplete

Treat the run manifest and log as authoritative. A project is not successful if
compact grids, benchmark outputs or configured renders fail validation. Preserve
restart state and fix the reported stage before interpreting partial outputs.

For command options, use the [CLI guide]({{ site.baseurl }}{% link reference/cli.md %}).
