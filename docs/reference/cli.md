---
layout: default
title: Command-Line Interface
parent: Reference
nav_order: 1
---

# Command-Line Interface

This page is generated from `openamundsen_da.cli.build_parser`. It documents the
single supported command tree installed by the `openamundsen-da` package.

Run `python scripts/docs/render_cli_reference.py` after changing the parser.
The documentation gate fails if this file is stale.

## `openamundsen-da`

```text
usage: openamundsen-da [-h] [--version]
                       {observations,prepare,run,render,clean,subdomains} ...

Prepare, run and inspect openAMUNDSEN data-assimilation projects.

positional arguments:
  {observations,prepare,run,render,clean,subdomains}
    observations        Preprocess configured observation products
    prepare             Prepare deterministic project steps and observations
    run                 Run a prepared single-domain project
    render              Regenerate configured project outputs
    clean               Preview safe heavy restart-artifact cleanup
    subdomains          Run explicit subdomain workflows

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

### `openamundsen-da observations`

```text
usage: openamundsen-da observations [-h] {snow-cover,wet-snow} ...

positional arguments:
  {snow-cover,wet-snow}
    snow-cover          Preprocess configured snow-cover observations
    wet-snow            Preprocess configured wet-snow observations

options:
  -h, --help            show this help message and exit
```

#### `openamundsen-da observations snow-cover`

```text
usage: openamundsen-da observations snow-cover [-h] [--json] [--overwrite]
                                               PROJECT_DIR

Preprocess configured snow-cover observations

positional arguments:
  PROJECT_DIR

options:
  -h, --help   show this help message and exit
  --json       Emit one machine-readable JSON envelope
  --overwrite
```

#### `openamundsen-da observations wet-snow`

```text
usage: openamundsen-da observations wet-snow [-h] [--json] [--overwrite]
                                             PROJECT_DIR

Preprocess configured wet-snow observations

positional arguments:
  PROJECT_DIR

options:
  -h, --help   show this help message and exit
  --json       Emit one machine-readable JSON envelope
  --overwrite
```

### `openamundsen-da prepare`

```text
usage: openamundsen-da prepare [-h] [--json] [--overwrite] PROJECT_DIR

Prepare deterministic project steps and observations

positional arguments:
  PROJECT_DIR

options:
  -h, --help   show this help message and exit
  --json       Emit one machine-readable JSON envelope
  --overwrite
```

### `openamundsen-da run`

```text
usage: openamundsen-da run [-h] [--json] [--max-workers MAX_WORKERS]
                           PROJECT_DIR

Run a prepared single-domain project

positional arguments:
  PROJECT_DIR

options:
  -h, --help            show this help message and exit
  --json                Emit one machine-readable JSON envelope
  --max-workers MAX_WORKERS
```

### `openamundsen-da render`

```text
usage: openamundsen-da render [-h] [--json] [--max-workers MAX_WORKERS]
                              PROJECT_DIR

Regenerate configured project outputs

positional arguments:
  PROJECT_DIR

options:
  -h, --help            show this help message and exit
  --json                Emit one machine-readable JSON envelope
  --max-workers MAX_WORKERS
```

### `openamundsen-da clean`

```text
usage: openamundsen-da clean [-h] [--json] [--apply] PROJECT_DIR

Preview safe heavy restart-artifact cleanup

positional arguments:
  PROJECT_DIR

options:
  -h, --help   show this help message and exit
  --json       Emit one machine-readable JSON envelope
  --apply      Apply the previewed cleanup
```

### `openamundsen-da subdomains`

```text
usage: openamundsen-da subdomains [-h] {prepare,run,merge,render,model} ...

positional arguments:
  {prepare,run,merge,render,model}
    prepare             Prepare DA subdomains for a project
    run                 Run prepared DA subdomains
    merge               Merge compact DA subdomain outputs
    render              Render merged DA subdomain outputs
    model               Tile one plain openAMUNDSEN simulation

options:
  -h, --help            show this help message and exit
```

#### `openamundsen-da subdomains prepare`

```text
usage: openamundsen-da subdomains prepare [-h] [--json] [--regions REGIONS]
                                          [--station-buffer-km STATION_BUFFER_KM]
                                          [--grid-buffer-m GRID_BUFFER_M]
                                          [--overwrite]
                                          PROJECT_DIR

Prepare DA subdomains for a project

positional arguments:
  PROJECT_DIR

options:
  -h, --help            show this help message and exit
  --json                Emit one machine-readable JSON envelope
  --regions REGIONS
  --station-buffer-km STATION_BUFFER_KM
  --grid-buffer-m GRID_BUFFER_M
  --overwrite
```

#### `openamundsen-da subdomains run`

```text
usage: openamundsen-da subdomains run [-h] [--json]
                                      [--max-workers MAX_WORKERS]
                                      [--inner-max-workers INNER_MAX_WORKERS]
                                      [--overwrite]
                                      PROJECT_DIR

Run prepared DA subdomains

positional arguments:
  PROJECT_DIR

options:
  -h, --help            show this help message and exit
  --json                Emit one machine-readable JSON envelope
  --max-workers MAX_WORKERS
  --inner-max-workers INNER_MAX_WORKERS
  --overwrite
```

#### `openamundsen-da subdomains merge`

```text
usage: openamundsen-da subdomains merge [-h] [--json]
                                        [--coverage-sliver-tol-px COVERAGE_SLIVER_TOL_PX]
                                        [--out-dir OUT_DIR]
                                        PROJECT_DIR

Merge compact DA subdomain outputs

positional arguments:
  PROJECT_DIR

options:
  -h, --help            show this help message and exit
  --json                Emit one machine-readable JSON envelope
  --coverage-sliver-tol-px COVERAGE_SLIVER_TOL_PX
  --out-dir OUT_DIR
```

#### `openamundsen-da subdomains render`

```text
usage: openamundsen-da subdomains render [-h] [--json]
                                         [--max-workers MAX_WORKERS]
                                         PROJECT_DIR

Render merged DA subdomain outputs

positional arguments:
  PROJECT_DIR

options:
  -h, --help            show this help message and exit
  --json                Emit one machine-readable JSON envelope
  --max-workers MAX_WORKERS
```

#### `openamundsen-da subdomains model`

```text
usage: openamundsen-da subdomains model [-h] {prepare,run,merge} ...

positional arguments:
  {prepare,run,merge}
    prepare            Prepare plain-model subdomains
    run                Run plain-model subdomains
    merge              Merge plain-model subdomain outputs

options:
  -h, --help           show this help message and exit
```

##### `openamundsen-da subdomains model prepare`

```text
usage: openamundsen-da subdomains model prepare [-h] [--regions REGIONS]
                                                [--station-buffer-km STATION_BUFFER_KM]
                                                [--grid-buffer-m GRID_BUFFER_M]
                                                [--overwrite] [--json]
                                                SETUP_DIR

positional arguments:
  SETUP_DIR

options:
  -h, --help            show this help message and exit
  --regions REGIONS
  --station-buffer-km STATION_BUFFER_KM
  --grid-buffer-m GRID_BUFFER_M
  --overwrite
  --json                Emit one machine-readable JSON envelope
```

##### `openamundsen-da subdomains model run`

```text
usage: openamundsen-da subdomains model run [-h] [--max-workers MAX_WORKERS]
                                            [--overwrite] [--json]
                                            SETUP_DIR

positional arguments:
  SETUP_DIR

options:
  -h, --help            show this help message and exit
  --max-workers MAX_WORKERS
  --overwrite
  --json                Emit one machine-readable JSON envelope
```

##### `openamundsen-da subdomains model merge`

```text
usage: openamundsen-da subdomains model merge [-h]
                                              [--coverage-sliver-tol-px COVERAGE_SLIVER_TOL_PX]
                                              [--out-dir OUT_DIR] [--json]
                                              SETUP_DIR

positional arguments:
  SETUP_DIR

options:
  -h, --help            show this help message and exit
  --coverage-sliver-tol-px COVERAGE_SLIVER_TOL_PX
  --out-dir OUT_DIR
  --json                Emit one machine-readable JSON envelope
```
