# openAMUNDSEN-DA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21519388.svg)](https://doi.org/10.5281/zenodo.21519388)

openAMUNDSEN-DA is the data assimilation layer for the distributed snow and
hydrological model [openAMUNDSEN](https://github.com/openamundsen/openamundsen).
It prepares configured snow observations, executes sequential ensemble updates
and writes reproducible diagnostics and compact gridded results.

The project is a v0.9 research preview. A scientific manuscript describing the
openAMUNDSEN-DA framework and its Rofental application is in preparation. This
repository documents the software interface and operational workflow.

## Documentation

- [Installation](https://doc-da.openamundsen.org/installation.html)
- [Input data](https://doc-da.openamundsen.org/guides/observations.html)
- [Configuration](https://doc-da.openamundsen.org/guides/configuration.html)
- [Running the model](https://doc-da.openamundsen.org/running.html)
- [Output data](https://doc-da.openamundsen.org/output-data.html)
- [Example data sets](https://doc-da.openamundsen.org/example-data.html)
- [How to Use](https://doc-da.openamundsen.org/tutorial/)
- [CLI reference](https://doc-da.openamundsen.org/reference/cli.html)

## Install

Install the Python package on Python 3.11–3.14:

```bash
python -m pip install openamundsen-da
```

For the complete geospatial runtime, use the multi-architecture container:

```bash
docker pull ghcr.io/openamundsen/openamundsen-da:0.9.3
```

Use exact versions or image digests for reproducible work. The moving `edge`
image tracks the latest green `main` commit and is not a release.

## Command-line workflow

The installed distribution exposes one command, `openamundsen-da`:

```text
openamundsen-da observations snow-cover PROJECT_DIR
openamundsen-da observations wet-snow PROJECT_DIR
openamundsen-da prepare PROJECT_DIR
openamundsen-da run PROJECT_DIR --max-workers 24
openamundsen-da render PROJECT_DIR --max-workers 24
openamundsen-da clean PROJECT_DIR
```

`clean` is preview-only unless `--apply` is supplied. A successful single-domain
run already removes package-owned restart state after all configured outputs pass
validation; member grids remain available.

Large domains use explicit data assimilation stages:

```text
openamundsen-da subdomains prepare PROJECT_DIR
openamundsen-da subdomains run PROJECT_DIR
openamundsen-da subdomains merge PROJECT_DIR
openamundsen-da subdomains render PROJECT_DIR
```

The separate `openamundsen-da subdomains model prepare|run|merge` branch tiles
one ordinary openAMUNDSEN simulation without projects or assimilation.

## Python interface

```python
from openamundsen_da import prepare_project, run_project

project = "/data/rofental/projects/project_2022_2023"
prepare_project(project)
result = run_project(project, max_workers=24)
print(result.manifest_path)
```

The supported Python surface contains the six top-level workflow operations
documented in the [API reference](https://doc-da.openamundsen.org/reference/api.html).
Subdomain modules and lower-level scientific routines are internal interfaces.

## Configuration boundary

- `<setup-name>.yml` is pure openAMUNDSEN configuration and shared setup data.
- `<project-name>.yml` owns the time span, observations and data assimilation.
- generated step YAML files own one assimilation window and step-local model overrides.

Model grid inputs are selected exclusively by `output_data.grids.format` and
support grid-layout NetCDF or deterministic georeferenced GeoTIFF. The compact
data assimilation result is always `results/grids/da_output_grids.nc`.

## Development

Build and validate the distribution, then build the same wheel-based image used
for release:

```bash
python -m pip install build twine
bash scripts/ci/build_distribution.sh
docker build -t openamundsen-da:local .
```

Run tests through the repository wrappers:

```bash
bash scripts/ci/run_lint.sh
bash scripts/ci/run_unit_tests.sh
bash scripts/ci/run_integration_tests.sh
bash scripts/ci/run_integration_tests_subdomain.sh
```

See [tests/README.md](tests/README.md) for the exact validation contracts.

## License

openAMUNDSEN-DA is released under the [MIT License](LICENSE), including
commercial use subject to the license terms.
