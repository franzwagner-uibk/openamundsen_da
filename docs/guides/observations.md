---
layout: default
title: Input Data
nav_order: 3
---

# Input Data

openAMUNDSEN-DA combines a standard openAMUNDSEN setup with explicit observation
products and one data assimilation project configuration.

![Setup, project and step configuration ownership in openAMUNDSEN-DA]({{ site.baseurl }}/assets/images/diagrams/setup-project-configuration.png)

*The setup contains shared openAMUNDSEN inputs. One project defines the data
assimilation period, observations and events; preparation materializes the
step-specific inputs.*

## openAMUNDSEN inputs

The setup root contains the standard openAMUNDSEN configuration, static grids and
meteorological forcing. Follow the upstream [openAMUNDSEN input-data documentation](https://doc.openamundsen.org/doc/input.html)
for model-specific variables and formats.

openAMUNDSEN-DA additionally requires:

```text
<setup>/
  <setup-name>.yml
  env/roi.gpkg
  grids/
  meteo/
  obs/
  projects/<project-name>/<project-name>.yml
```

`output_data.grids.format` in the setup YAML selects the only accepted model-grid
reader. Supported inputs are grid-layout NetCDF and deterministic georeferenced
GeoTIFF. ASCII output, in-memory output, NetCDF `roi_pixel` layout, mixed formats
and discovery fallbacks are rejected.

## Observation products

The project YAML declares every observation family, directory, file format,
product tag, summary path and class mapping. Paths are setup-relative and may not
escape the setup root.

### Snow cover

Snow-cover input is GeoTIFF or NetCDF with an explicit class mapping. The
preprocessor masks the ROI and configured exclusions and writes
`scf_summary.csv`. When uncertainty-aware processing is enabled, NetCDF variables
or GeoTIFF `<stem>_uncertainty.tif` sidecars must be present as configured.

### Wet snow

Wet-snow input is GeoTIFF or NetCDF with explicit wet, valid and excluded class
codes. The preprocessor writes `wet_snow_summary.csv` and, when configured, wet
snow line diagnostics and profiles.

### Station observations

Station snow depth and snow water equivalent use one CSV per station under the
configured station directory. `stations_da_metadata.csv` declares station roles
and uncertainty settings. Station observations are read directly during project
preparation; they do not have a raster-summary stage.

## Observation preprocessing

Run only the preprocessors required by the configured project:

```bash
openamundsen-da observations snow-cover PROJECT_DIR
openamundsen-da observations wet-snow PROJECT_DIR
```

`PROJECT_DIR` is `<setup>/projects/<project>`. Both commands read paths and class
mappings from the project YAML. `--overwrite` replaces an existing summary;
without it, an existing output is not silently changed. `--json` emits the same
typed result as one machine-readable envelope.

Inspect summary dates, spatial support, invalid/cloud fractions and uncertainty
statistics before selecting `assimilation_events`. `openamundsen-da prepare`
then verifies that each configured event has the required observation and writes
deterministic per-step inputs.

See [Configuration]({{ site.baseurl }}{% link guides/configuration.md %}) for the
YAML schema.
