---
layout: default
title: Installation
nav_order: 2
---

# Installation

openAMUNDSEN-DA is distributed as a Python package and as a wheel-based,
multi-architecture Docker image. The image includes openAMUNDSEN, GDAL/PROJ and
the plotting stack and is the recommended path for the complete workflow.

## Requirements

- Docker Desktop on Windows 10/11 with WSL2 or on macOS, or Docker Engine on Linux
- an x86-64 or Apple-silicon/ARM64 host
- enough local storage for one setup plus member outputs

A small wiring test can run with 4 CPU cores and 16 GB RAM. The tutorial ES30
project is more comfortable with 8 or more cores, 32 GB RAM and at least 50 GB
of free disk. Higher resolution, more members and full member-grid retention can
increase storage substantially; monitor `results/plots/perf/` during the run.

Verify Docker before downloading the image:

```bash
docker run --rm hello-world
```

## Docker image

Pull the stable image reviewed by this documentation:

```bash
docker pull {{ site.data.release.image }}
```

Verify the installed version and architecture:

```bash
docker run --rm {{ site.data.release.image }} openamundsen-da --version
```

The image contains the shipped examples under `/workspace/examples`. Copy an
example to a host directory before running it so that configuration and outputs
persist outside the container. The [How to Use installation chapter]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %})
shows that sequence in full.

## Python package

Python 3.11, 3.12, 3.13 and 3.14 are supported:

```bash
python -m pip install openamundsen-da=={{ site.data.release.version }}
```

The stable distribution is published on
[PyPI](https://pypi.org/project/openamundsen-da/). Use the reviewed container
when you need the complete end-to-end runtime with its geospatial system
libraries.

The Python package does not replace system-level geospatial libraries on every
platform. Use the container when you need the tested end-to-end runtime.

## Developer checkout

Clone the repository, build the strict distribution and install the tested wheel:

```bash
git clone https://github.com/openamundsen/openamundsen-da.git
cd openamundsen-da
python -m pip install build twine
bash scripts/ci/build_distribution.sh
python -m pip install dist/openamundsen_da-*.whl
```

`compose.yml` executes the non-editable wheel installed in the image and mounts
only setup data and cache. Add `compose.dev.yml` explicitly when developing with
a source overlay.

## Next steps

Continue with [Input Data]({{ site.baseurl }}{% link guides/observations.md %}),
[Configuration]({{ site.baseurl }}{% link guides/configuration.md %}) or the
[How to Use tutorial]({{ '/tutorial/' | relative_url }}). Installation problems
are collected under [Advanced troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}).
