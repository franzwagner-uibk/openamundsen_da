---
published: false
---

# Output Completeness and Diagnostic Plot Contracts

Status: approved by the user on 2026-08-11.

## Scope

This change strengthens four existing contracts without adding configuration
keys or changing scientific calculations: compact grid completeness, point
observation presentation, data assimilation weight readability and performance
monitor presentation. The shipped Rofental configuration remains the
single-domain regression baseline. No storage-lifecycle behavior, subdomain
event selection, release or version change is included.

## Compact Grid Completeness

The run preflight compares the output names declared at
`setup.output_data.grids.variables[*].name` with the source names declared at
`project.data_assimilation.output.grids.variables[*].var` or `name`. Every
explicitly requested compact source must be produced by openAMUNDSEN. Missing
sources are one aggregated, fail-fast configuration error that identifies both
configuration paths and every missing name. Maintained snow setups request
`snow.depth` as `snowdepth_daily` and `snow.swe` as `swe_daily`.

The compact builder applies the same strictness to generated artifacts. When a
project explicitly configures compact variables, the open-loop and every
member NetCDF must contain all of them with compatible dimensions and shapes.
The completed compact dataset must contain every requested metric-variable
pair. Errors aggregate all missing variables or metrics and name the affected
files. Legacy projects that omit the compact-variable list retain their
existing all-available-output behavior.

The project writer validates each step and its completed project NetCDF. The
subdomain merge validates every leaf compact NetCDF and the merged NetCDF
against the top-level project contract, preventing a consistently incomplete
set of leaves from passing merge validation.

## Point Observation Presentation

Standalone point-result panels continue to render valid ensemble and open-loop
data when no same-ID station observation exists. Their station-observation
line and legend item remain conditional on finite observation data, and their
title uses `openAMUNDSEN ensemble and open loop` when no observation is drawn.
When a finite same-ID series exists, the current observation line, markers,
legend wording and station-observation title are preserved. ROI aggregate
titles remain unchanged.

## Weight Plot Readability

Each event panel derives a symmetric residual range from the finite residuals
actually plotted in that panel plus every finite sigma/reference line. A
deterministic nice-number ceiling and small edge margin keep all marks visible
without allowing another event to widen the panel. Snow-depth residuals remain
in meters, SWE residuals in millimeters and fractional residuals dimensionless.

Sigma keys with at most five entries preserve the established vertical layout.
Larger keys remain inside the residual panel and wrap deterministically into a
compact multirow grid, preventing them from overprinting titles or neighboring
panels. This leaves the smaller Rofental key visually unchanged.

## Performance Monitor Presentation

The performance plot retains its dimensions and CSV schema. When CPU
temperature is available, layout is computed after an initial canvas draw:
the project-size tick labels and axis-label extents determine the temperature
spine offset, while the right subplot margin expands only as much as those
rendered extents require. This keeps the two right axes separate for project
sizes from single-digit gigabytes through four-digit gigabytes without
reserving large-run whitespace for small runs.

RAM and project-size values in the header use one decimal place. Temperature
labels and the header consistently use `°C`. Exact recursive project-size
scans default to 150 seconds instead of 300 seconds; CPU/RAM sampling, plot
refresh, paths and the no-temperature path remain unchanged. The recursive
scan cost is documented, but storage monitoring or termination is outside this
change.

## Acceptance

Focused tests cover aggregated preflight failures, member and final compact
NetCDF completeness, legacy compatibility, point panels with and without
observations, narrow and wide residual data, wrapped sigma keys, dynamic
right-axis separation for one-, three- and four-digit project sizes,
one-decimal headers, degree notation and the 150-second default. The existing
Rofental plotting tests and compact-grid tests remain green.
