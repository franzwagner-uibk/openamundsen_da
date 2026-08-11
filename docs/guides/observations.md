---
layout: default
title: Input data
parent: Documentation
nav_order: 2
---

# Input data

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

## Snow observation products

The project YAML declares every observation family, directory, file format,
product tag, summary path and class mapping. Paths are setup-relative and may not
escape the setup root.

### Snow cover

Snow-cover input is GeoTIFF or NetCDF with an explicit class mapping. The
preprocessor masks the ROI and configured exclusions and writes
`scf_summary.csv`. When uncertainty-aware processing is enabled, NetCDF variables
or GeoTIFF `<stem>_uncertainty.tif` sidecars must be present as configured.

Satellite products can reference a tracked acquisition manifest with columns
`product`, `source`, `product_identity`, `acquisition_time`, `time_source` and
`time_quality`. Acquisition time is resolved from the CF time coordinate,
raster metadata, sidecar metadata, a configured filename parser, then this
manifest. If none is available, preparation warns and records UTC midnight with
`time_quality=fallback_midnight`.

### Wet snow

Wet-snow input is GeoTIFF or NetCDF with explicit wet, valid and excluded class
codes. The preprocessor writes `wet_snow_summary.csv` and, when configured, wet
snow line diagnostics and profiles.

### Station observations

Station snow depth and snow water equivalent use one CSV per station under the
configured station directory. `stations_da_metadata.csv` declares station roles
and uncertainty settings. Its `station_id`, `x` and `y` columns also support
spatial subdomain selection when no legacy `stations_snow_depth.csv` is present.
Station observations are read directly during project preparation; they do not
have a raster-summary stage. A station event is supported only by an enabled DA
station with a same-ID series and one finite, nonnegative observation within
half the setup timestep of the model assimilation timestamp. Ties are invalid;
a value elsewhere on the same date is not support.

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

Prepared satellite rows record both the timezone-aware UTC observation time and
the matched naive model-clock time. A date with several scenes requires the
event's `observation_time` selector. A unique scene may still use the date-only
event form.

See [Configuration]({{ site.baseurl }}{% link guides/configuration.md %}) for the
YAML schema.
