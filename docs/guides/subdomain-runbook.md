---
layout: default
title: Sub-domain Runbook
parent: Guides
nav_order: 6
---

# End-to-end Sub-domain Runbook
{: .no_toc }

Run a large openAMUNDSEN-DA setup as independent sub-domains and merge the grid output.
{: .fs-6 .fw-300 }

Sub-domain mode is a regional decomposition workflow: each polygon is prepared and
assimilated as an independent project before compact grid outputs are merged. It
is not a particle-filter localization scheme.

<details markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## What This Produces

For a data assimilation project, the sub-domain workflow:

1. reads one normal setup directory
2. splits it with non-overlapping polygons from `env/subdomains.gpkg`
3. creates one independent project per sub-domain
4. runs all sub-domains in parallel
5. merges compact gridded DA outputs into one hard-mosaic NetCDF

The final merged DA grid product is:

```text
<setup>/projects/<project>/results/grids/da_output_grids.nc
```

Sub-domain summary tables are written to:

```text
<setup>/projects/<project>/results/subdomain_*.csv
```

Point outputs and point plots remain inside each sub-domain project. They are not merged at the project root.

## Requirements

Install Docker on the machine that will run the workflow. The commands below use Bash syntax and mount the host setup directory at `/data` inside the container.

Pull the image:

```bash
docker pull ghcr.io/franzwagner-uibk/openamundsen_da:latest
```

## Host Setup Layout

A complete setup directory is required on the host, for example:

```text
large_setup/
  large_setup.yml
  env/
    subdomains.gpkg
  grids/
    dem_<domain>_<resolution>.asc
    lc_<domain>_<resolution>.asc
    roi_<domain>_<resolution>.asc
    ...
  meteo/
    stations.csv
    <station>.csv
    ...
  obs/
    ...
  projects/
    project_YYYY-YYYY/
      project_YYYY-YYYY.yml
```

The setup YAML remains a plain openAMUNDSEN configuration. The project YAML contains the data assimilation configuration, including `start_date`, `end_date`, and `data_assimilation.assimilation_events`.

The regions file must contain at least two non-overlapping polygons:

```text
large_setup/env/subdomains.gpkg
```

Each feature must have the canonical `id` field. Keep identifiers short and stable, for example `sd_01`, `sd_02` and `sd_03`.

## Pre-run Checklist

- `subdomains.gpkg` uses the same CRS as the setup domain or can be reprojected correctly.
- polygons cover the complete setup ROI without gaps inside the area that should be simulated.
- polygons do not overlap, except for tiny slivers tolerated by the CLI.
- the project YAML is marked for sub-domain mode:

```yaml
run_mode: subdomain
```

- configured assimilation events are available in the observation summaries for the local sub-domains, or `data_assimilation.subdomain_event_filter` is enabled so unavailable local events can be dropped explicitly.
- the machine has enough disk space for intermediate per-sub-domain projects.
- `--max-workers` is no larger than the CPU cores available to Docker.

## Staged DA Run

Set the host setup path and project name:

```bash
SETUP_HOST=/absolute/path/to/large_setup
PROJECT_NAME=project_YYYY-YYYY
IMAGE=ghcr.io/franzwagner-uibk/openamundsen_da:latest
```

Run the four explicit stages in order:

```bash
docker run \
  -v "${SETUP_HOST}:/data" \
  "${IMAGE}" \
  openamundsen-da subdomains prepare "/data/projects/${PROJECT_NAME}"

docker run \
  -v "${SETUP_HOST}:/data" \
  "${IMAGE}" \
  openamundsen-da subdomains run "/data/projects/${PROJECT_NAME}" \
    --max-workers 8 \
    --inner-max-workers 3

docker run \
  -v "${SETUP_HOST}:/data" \
  "${IMAGE}" \
  openamundsen-da subdomains merge "/data/projects/${PROJECT_NAME}"

docker run \
  -v "${SETUP_HOST}:/data" \
  "${IMAGE}" \
  openamundsen-da subdomains render "/data/projects/${PROJECT_NAME}" \
    --max-workers 8
```

This assumes the regions file is available at `/data/env/subdomains.gpkg`. Pass `--regions /data/path/to/regions.gpkg` to the prepare stage when needed.

The supported sequence is:

```text
prepare -> run -> merge -> render
```

The separate stages are the resumability interface. Inspect `<project>/subdomains/subdomain_manifest.json` after any interruption; it records the latest state of `prepare`, `run`, `merge`, `render` and compact cleanup. Rerun only the failed or interrupted command. Merge replaces each completed mosaic atomically and render does not remove compact-retention inputs until all maps and the project report validate.

## Optional Flags

Add these only when the default behavior is not appropriate:

- `--regions /data/path/to/regions.gpkg`: use a regions file outside `/data/env/subdomains.gpkg`.
- `--max-workers <n>`: limit parallel sub-domain workers.
- `--inner-max-workers <n>`: limit parallel member workers inside each DA sub-domain.
- `--station-buffer-km <km>`: select meteo and station-observation inputs from each sub-domain plus this buffer distance.
- `--overwrite`: replace existing prepared or successful outputs.
- `--rm`: remove the stopped container after the command finishes.
- `--user "$(id -u):$(id -g)"`: on Linux, write mounted outputs as the current host user.

Docker resource flags such as `--cpus <n>` and `--memory <size>` can be added before the image name when the run should be constrained by Docker.

## Output Locations

After a successful DA merge, inspect:

```text
large_setup/projects/project_YYYY-YYYY/subdomain_run.log
large_setup/projects/project_YYYY-YYYY/subdomains/subdomain_manifest.json
large_setup/projects/project_YYYY-YYYY/results/grids/da_output_grids.nc
```

After strict rendering, also inspect:

```text
large_setup/projects/project_YYYY-YYYY/results/subdomain_overview.csv
large_setup/projects/project_YYYY-YYYY/results/subdomain_assimilation_stats.csv
large_setup/projects/project_YYYY-YYYY/results/subdomain_assimilation_aggregate.csv
large_setup/projects/project_YYYY-YYYY/results/subdomain_dropped_events.csv
```

`subdomain_dropped_events.csv` is written when event filtering is enabled. It records the project-level candidate events that were not assimilated in a specific sub-domain, including the filter reason and threshold.

Each sub-domain keeps its own project under:

```text
large_setup/projects/project_YYYY-YYYY/subdomains/<subdomain_id>/
```

The merged NetCDF contains compact DA grid variables such as:

```text
open_loop_<var>
ens_mean_<var>
ens_std_<var>
ens_min_<var>
ens_max_<var>
increment_<var>
analysis_mean_<var>
analysis_increment_<var>
```

`analysis_*` variables are present when event weights are available.

The merge is a hard mosaic. It does not interpolate, blend, smooth boundaries, or exchange particles across sub-domains. Visible breaks at sub-domain boundaries can therefore be expected.

## Plain openAMUNDSEN Model-only Run

For a split and merged plain openAMUNDSEN model run without data assimilation, use the nested model branch. The setup does not need `projects/` or `obs/`, but the setup YAML must define `start_date`, `end_date`, domain settings, grid and meteo input directories, and desired grid outputs.

```bash
SETUP_HOST=/absolute/path/to/large_setup
IMAGE=ghcr.io/franzwagner-uibk/openamundsen_da:latest

docker run \
  -v "${SETUP_HOST}:/data" \
  "${IMAGE}" \
  openamundsen-da subdomains model prepare /data

docker run \
  -v "${SETUP_HOST}:/data" \
  "${IMAGE}" \
  openamundsen-da subdomains model run /data --max-workers 24

docker run \
  -v "${SETUP_HOST}:/data" \
  "${IMAGE}" \
  openamundsen-da subdomains model merge /data
```

The merged model grid outputs are written to:

```text
large_setup/subdomains/model/results/grids/
```

Per-subdomain model outputs and diagnostics remain under:

```text
large_setup/subdomains/model/<subdomain_id>/results/
large_setup/subdomains/model/<subdomain_id>/run.log
large_setup/subdomains/model/<subdomain_id>/run_manifest.json
```

Model mode also uses a hard mosaic and only merges configured grid outputs under each subdomain `results/grids/` directory. `output_data.grids.format` selects NetCDF or GeoTIFF deterministically. Mixed formats, unsupported ASCII or memory output and stale ambiguous artifacts are rejected.

## Common Problems

If preparation fails with overlap or uncovered-pixel errors, fix the regions file first; command-line fallbacks do not hide invalid partitions.

If a sub-domain run fails because observations are missing, check that the configured `assimilation_events` are present in the sub-domain observation summaries. Without `data_assimilation.subdomain_event_filter`, sub-domain mode fails fast when a local sub-domain does not have the required events. With filtering enabled, inspect `subdomain_dropped_events.csv` to verify which candidate events were skipped locally.

If the machine runs out of memory, reduce `--max-workers` or `--inner-max-workers` and rerun the failed stage.

If host files become root-owned after the run on Linux, repair ownership:

```bash
sudo chown -R "$USER":"$USER" "$SETUP_HOST"
chmod -R u+rwX "$SETUP_HOST"
```
