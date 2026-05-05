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

Install Docker on the machine that will run the workflow. The commands below use Bash syntax and write files as the current host user with `--user "$(id -u):$(id -g)"`. On Windows PowerShell, replace that argument with a suitable user mapping or omit it and repair file ownership afterwards if needed.

Pull the image:

```bash
docker pull ghcr.io/franzwagner-uibk/openamundsen_da:latest
```

If the GHCR package is private in your environment, log in first:

```bash
echo "$GHCR_PAT_RO" | docker login ghcr.io -u <github-user> --password-stdin
docker pull ghcr.io/franzwagner-uibk/openamundsen_da:latest
```

Use a dated or SHA tag instead of `latest` when the run must be exactly reproducible:

```bash
docker pull ghcr.io/franzwagner-uibk/openamundsen_da:<tag>
```

## Host Setup Layout

The colleague needs one complete setup directory on the host, for example:

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

By default, each feature must have an `id` field. Use `--id-field <field>` if the identifier column has another name. Keep identifiers short and stable, for example `sd_01`, `sd_02`, `sd_03`.

## Pre-run Checklist

- `subdomains.gpkg` uses the same CRS as the setup domain or can be reprojected correctly.
- polygons cover the complete setup ROI without gaps inside the area that should be simulated.
- polygons do not overlap, except for tiny slivers tolerated by the CLI.
- the project YAML is marked for sub-domain mode:

```yaml
run_mode: subdomain
```

- configured assimilation events are available in the observation summaries for the local sub-domains.
- the machine has enough disk space for intermediate per-sub-domain projects.
- `--max-workers` is no larger than the CPU cores available to Docker.

## One-shot DA Run

Set the host setup path and project name:

```bash
SETUP_HOST=/absolute/path/to/large_setup
PROJECT_NAME=project_YYYY-YYYY
IMAGE=ghcr.io/franzwagner-uibk/openamundsen_da:latest
```

Run the full sub-domain data assimilation pipeline:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${SETUP_HOST}:/data" \
  --cpus 24 \
  --memory 64g \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/xdg \
  -e MPLCONFIGDIR=/tmp/mpl \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 \
  -e GPD_USE_PYOGRIO=0 \
  "${IMAGE}" \
  oa-da-subdomain pipeline \
    --setup-dir /data \
    --project-dir "/data/projects/${PROJECT_NAME}" \
    --regions /data/env/subdomains.gpkg \
    --id-field id \
    --max-workers 24 \
    --inner-max-workers 2 \
    --overwrite
```

Adjust `--cpus`, `--memory`, `--max-workers`, and `--inner-max-workers` to the machine. For large domains, start conservatively. A useful first setting is one outer worker per physical core group and `--inner-max-workers 1` or `2`.

The pipeline runs:

```text
prepare -> run -> report -> merge -> plot
```

If plotting inputs are incomplete, map plotting is best effort and the pipeline continues after writing the merged grid output.

## Staged DA Run

Use staged commands when debugging geometry, observations, or a failing sub-domain. The staged sequence below prepares, runs, and merges the grids. The `subdomain_*.csv` report tables are produced by the one-shot pipeline.

Prepare sub-domain projects:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${SETUP_HOST}:/data" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/xdg \
  -e MPLCONFIGDIR=/tmp/mpl \
  -e GPD_USE_PYOGRIO=0 \
  "${IMAGE}" \
  oa-da-subdomain prepare \
    --setup-dir /data \
    --project-dir "/data/projects/${PROJECT_NAME}" \
    --regions /data/env/subdomains.gpkg \
    --id-field id \
    --overwrite
```

Run all prepared sub-domains:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${SETUP_HOST}:/data" \
  --cpus 24 \
  --memory 64g \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/xdg \
  -e MPLCONFIGDIR=/tmp/mpl \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 \
  -e GPD_USE_PYOGRIO=0 \
  "${IMAGE}" \
  oa-da-subdomain run \
    --project-dir "/data/projects/${PROJECT_NAME}" \
    --max-workers 24 \
    --inner-max-workers 2
```

Merge grids:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${SETUP_HOST}:/data" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/xdg \
  -e MPLCONFIGDIR=/tmp/mpl \
  -e GPD_USE_PYOGRIO=0 \
  "${IMAGE}" \
  oa-da-subdomain merge \
    --project-dir "/data/projects/${PROJECT_NAME}"
```

Optionally render station comparison plots:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${SETUP_HOST}:/data" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/xdg \
  -e MPLCONFIGDIR=/tmp/mpl \
  -e GPD_USE_PYOGRIO=0 \
  "${IMAGE}" \
  oa-da-subdomain plot \
    --project-dir "/data/projects/${PROJECT_NAME}"
```

To run or merge only selected sub-domains, add for example:

```bash
--subdomains sd_01 sd_02
```

## Output Locations

After a successful DA merge, inspect:

```text
large_setup/projects/project_YYYY-YYYY/subdomain_run.log
large_setup/projects/project_YYYY-YYYY/subdomains/subdomain_manifest.json
large_setup/projects/project_YYYY-YYYY/results/grids/da_output_grids.nc
```

After the one-shot pipeline, also inspect:

```text
large_setup/projects/project_YYYY-YYYY/results/subdomain_overview.csv
large_setup/projects/project_YYYY-YYYY/results/subdomain_assimilation_stats.csv
large_setup/projects/project_YYYY-YYYY/results/subdomain_assimilation_aggregate.csv
```

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

The merge is a hard mosaic. It does not interpolate, blend, or smooth boundaries. Visible breaks at sub-domain boundaries can therefore be expected.

## Plain openAMUNDSEN Model-only Run

If the colleague only wants to split and merge a plain openAMUNDSEN model run without data assimilation, use the `model-*` commands. The setup does not need `projects/` or `obs/`, but the setup YAML must define `start_date`, `end_date`, domain settings, grid and meteo input directories, and desired grid outputs.

```bash
SETUP_HOST=/absolute/path/to/large_setup
IMAGE=ghcr.io/franzwagner-uibk/openamundsen_da:latest

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${SETUP_HOST}:/data" \
  --cpus 24 \
  --memory 64g \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/xdg \
  -e MPLCONFIGDIR=/tmp/mpl \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 \
  -e GPD_USE_PYOGRIO=0 \
  "${IMAGE}" \
  oa-da-subdomain model-pipeline \
    --setup-dir /data \
    --regions /data/env/subdomains.gpkg \
    --id-field id \
    --max-workers 24 \
    --overwrite
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

Model mode also uses a hard mosaic and only merges matching grid outputs under each sub-domain `results/grids/` directory.

## Common Problems

If preparation fails with overlap or uncovered-pixel errors, fix the regions file first. For tiny geometry slivers, the DA commands expose `--overlap-area-tol-m2` and `--sliver-fix-m`, but those options should not hide real overlaps.

If a sub-domain run fails because observations are missing, check that the configured `assimilation_events` are present in the sub-domain observation summaries. Sub-domain mode fails fast when a local sub-domain does not have the required events.

If the machine runs out of memory, reduce `--max-workers`, reduce `--inner-max-workers`, or rerun a subset with `--subdomains`.

If host files become root-owned after the run on Linux, repair ownership:

```bash
sudo chown -R "$USER":"$USER" "$SETUP_HOST"
chmod -R u+rwX "$SETUP_HOST"
```
