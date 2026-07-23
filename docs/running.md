---
layout: default
title: Running the model
parent: Documentation
nav_order: 4
---

# Running the model

All supported workflows use the single `openamundsen-da` command tree. Commands
accept a project directory, not separate setup/config guesses; the setup is the
parent of the project's `projects/` directory.

## Single-domain project

Preprocess configured raster observations when their summaries need to be
created or refreshed, then prepare and run the project:

```bash
openamundsen-da observations snow-cover /data/setup/projects/project_2022_2023
openamundsen-da observations wet-snow /data/setup/projects/project_2022_2023
openamundsen-da prepare /data/setup/projects/project_2022_2023
openamundsen-da run /data/setup/projects/project_2022_2023 --max-workers <MAX_WORKERS>
```

`prepare` creates the deterministic step structure and per-step observation
inputs. `run` requires a successful preparation manifest. It resumes only when
all hashed inputs match; completed projects are immutable and returned as reused.

Performance monitoring is always enabled and writes
`results/plots/perf/project_perf_metrics.csv` and `project_perf.png`. Set numeric
library thread counts to 1 when using multiple process workers to avoid
oversubscription. Replace `<MAX_WORKERS>` with a value no larger than the CPUs
available to the process; do not type the angle brackets. A project with an
ensemble of size `N` has at most `N + 1` useful propagation workers, including
the open loop. Reduce the value when memory is the limiting resource.

Regenerate configured plots, maps and the project report from a compatible
completed project with:

```bash
openamundsen-da render /data/setup/projects/project_2022_2023 --max-workers <MAX_WORKERS>
```

## Docker execution

Mount the setup at `/data` and run the exact release image:

```bash
docker run --rm \
  -v "/absolute/path/to/setup:/data" \
  --cpus <CPU_COUNT> \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 \
  {{ site.data.release.image }} \
  openamundsen-da run /data/projects/project_2022_2023 --max-workers <MAX_WORKERS>
```

Replace `<CPU_COUNT>` and `<MAX_WORKERS>` before execution. The worker limit
must not exceed the CPUs exposed to the container.

On Windows and macOS, assign enough CPU, memory and disk to Docker Desktop. On
Linux, add `--user "$(id -u):$(id -g)"` when mounted outputs should use the
current host ownership.

## Data assimilation subdomains

Large data assimilation projects use four explicit, resumable stages:

```bash
openamundsen-da subdomains prepare PROJECT_DIR
openamundsen-da subdomains run PROJECT_DIR \
  --max-workers <MAX_WORKERS> \
  --inner-max-workers <INNER_MAX_WORKERS>
openamundsen-da subdomains merge PROJECT_DIR
openamundsen-da subdomains render PROJECT_DIR --max-workers <MAX_WORKERS>
```

Each subdomain is an independent data assimilation problem. This is regional
decomposition, not particle-filter localization. See the
[Subdomain Runbook]({{ site.baseurl }}{% link guides/subdomain-runbook.md %}).

## Plain-model subdomains

The intentionally separate model branch tiles one ordinary openAMUNDSEN
simulation without projects, ensembles or assimilation:

```bash
openamundsen-da subdomains model prepare SETUP_DIR
openamundsen-da subdomains model run SETUP_DIR --max-workers <MAX_WORKERS>
openamundsen-da subdomains model merge SETUP_DIR
```

Use the [CLI guide]({{ site.baseurl }}{% link reference/cli.md %})
for all options and [Output data]({{ site.baseurl }}{% link output-data.md %})
for completion and cleanup semantics.
