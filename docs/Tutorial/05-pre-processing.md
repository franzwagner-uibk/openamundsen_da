---
layout: default
title: 4. Preprocessing
parent: How to Use
nav_order: 4
permalink: /tutorial/pre-processing/
---

# 4. Preprocessing

This chapter turns the bundled observation raster products into the explicit files
that the project pipeline consumes. The shipped Rofental example already includes
baseline `scf_summary.csv` and `wet_snow_summary.csv` files so users can start
running the pipeline immediately. In the tutorial, we still walk through the
preprocessing steps so you can reproduce or modify those summaries yourself.
Preprocessing has three stages: summarize snow-cover and wet-snow rasters to
project-level CSVs, build the project step skeleton from `assimilation_events`,
and then generate one-row per-step observation CSVs under `steps/*/obs/`.

From this point on, the tutorial assumes you are inside the running tutorial
container shell at `/data/rofental`.

## Inputs and configuration

The Rofental example provides three observation groups: `obs/snowcover/` for
snow-cover fraction rasters, `obs/wetsnow/` for wet-snow masks, and
`obs/stations/` for station observations plus station-specific DA metadata in
`stations_da_metadata.csv`. Baseline SCF and wet-snow summaries are shipped under
`obs/summaries/project_2022_2023/`. All preprocessing commands are driven by the
project YAML. The most important parts are `obs.snowcover`, `obs.wetsnow`,
`data_assimilation.station`, `data_assimilation.landcover_mask`, and
`data_assimilation.assimilation_events`. If class mappings, product tags, or
required station metadata are missing, preprocessing or the later project run
fails instead of guessing.

The same YAML-driven setup map used earlier in the tutorial is a useful reference
while checking ROI coverage, station locations, and the static land-cover context
that later influences preprocessing:

![Rofental tutorial setup map (ROI, stations, and land-cover context)]({{ site.baseurl }}/assets/images/tutorial/rofental_setup_map.png)

Selected sections from
`/data/rofental/projects/project_2022_2023/project_2022_2023.yml`:

```yaml
start_date: "2022-10-01"
end_date: "2023-06-30"

obs:
  stations:
    dir: obs/stations
  snowcover:
    dir: obs/snowcover
    product_tag: SNOWCOVER
    classes:
      cloud: [205]
      water: [210]
      nodata: [255]
  wetsnow:
    dir: obs/wetsnow
    product_tag: WETSNOW
    classes:
      wet: [110]
      valid: [110, 125, 200, 210]
      exclude: [200, 210]

data_assimilation:
  station:
    default_station_uncertainty_pct: 25
    min_station_uncertainty_pct: 10
    single_station_factor: 2.0
  assimilation_events:
    - date: "2022-11-24"
      variable: station_hs
    - date: "2022-12-22"
      variable: station_hs
    - date: "2023-01-21"
      variable: station_hs
    - date: "2023-02-21"
      variable: station_hs
    - date: "2023-03-22"
      variable: station_hs
    - date: "2023-04-29"
      variable: wet_snow
      product: WETSNOW
    - date: "2023-05-03"
      variable: station_hs
    - date: "2023-05-23"
      variable: wet_snow
      product: WETSNOW
    - date: "2023-05-18"
      variable: scf
      product: SNOWCOVER
    - date: "2023-05-26"
      variable: scf
      product: SNOWCOVER
  landcover_mask:
    classes_to_exclude: [...]
  uncertainty:
    scf:
      enabled: true
      assimilation:
        sigma_mode: uncertainty_layer
        aggregate_metric: unc_mean
    wet_snow:
      enabled: true
      assimilation:
        sigma_mode: formula
        aggregate_metric: unc_mean
```

This tutorial uses uncertainty as provided input data rather than generating it
from scratch. The bundled example already contains the required uncertainty
rasters next to the GeoTIFF observations. In the current project config, SCF uses
`sigma_mode: uncertainty_layer`, while wet snow uses `sigma_mode: formula` with
the uncertainty rasters still available for inspection and experimentation.
openAMUNDSEN-DA expects uncertainty values on a `0..100` scale and treats missing
or invalid uncertainty inputs as an error when uncertainty-aware preprocessing is
enabled.

Conceptual background and best-practice rules are summarized in
[Workflow: Observation Uncertainty]({{ site.baseurl }}{% link workflow.md %}#observation-uncertainty).

The figure below shows what one tutorial uncertainty layer is meant to represent:
a continuous field over valid FSC pixels, shaped by the baseline FSC uncertainty
and local penalties such as land cover. Clouds remain missing observations rather
than turning into high-uncertainty pixels, and later preprocessing aggregates
valid-pixel metrics such as `unc_mean` into the summary tables.

![Rofental SCF uncertainty example with land-cover component and local zoom]({{ site.baseurl }}/assets/images/tutorial/rofental_uncertainty.png)

## Step 1: Summarize snow-cover rasters to `scf_summary.csv`

Run `oa-da-snowcover` to summarize the FSC rasters in `obs/snowcover/` into a
project-level table. This command reads ROI masking, land-cover exclusions,
product tags, and class mappings from the project YAML and writes
`/data/rofental/obs/summaries/project_2022_2023/scf_summary.csv`. In practice,
this file is your quality-control table for SCF date selection and the later input
to `oa-da-scf`.

This command depends mainly on `obs.snowcover.dir`,
`obs.snowcover.product_tag`, `obs.snowcover.classes.*`, and
`data_assimilation.landcover_mask.*` in
`/data/rofental/projects/project_2022_2023/project_2022_2023.yml`.

**🟢 Run this command:**

```bash
oa-da-snowcover \
  --input-dir /data/rofental/obs/snowcover \
  --project-label project_2022_2023 \
  --setup-dir /data/rofental \
  --overwrite
```

Before you trust the resulting summary, confirm that the expected acquisition dates
are present, cloud fraction is acceptable for your use case, valid support is
large enough, and the `scf` values look plausible for the season. Expect columns
for date, spatial support, `scf`, `cloud_fraction`, and `source`. When
uncertainty is enabled, the summary also contains `unc_mean`, `unc_min`,
`unc_max`, and `unc_n_valid`.

Reference snippet from
`/data/rofental/obs/summaries/project_2022_2023/scf_summary.csv`:

| date | region_id | n_valid | n_snow | scf | cloud_fraction | source |
| --- | --- | --- | --- | --- | --- | --- |
| 2022-10-03 |  | 88455 | 46001 | 0.52 | 0.00 | s2_fsc_snowflake_rofental_2022_10_03.tif |
| 2022-10-05 |  | 78631 | 23973 | 0.30 | 0.00 | s2_fsc_snowflake_rofental_2022_10_05.tif |
| 2022-10-08 |  | 33976 | 1078 | 0.03 | 0.00 | s2_fsc_snowflake_rofental_2022_10_08.tif |
| 2022-10-13 |  | 12530 | 306 | 0.02 | 0.00 | s2_fsc_snowflake_rofental_2022_10_13.tif |
| 2022-10-18 |  | 72228 | 2255 | 0.03 | 0.00 | s2_fsc_snowflake_rofental_2022_10_18.tif |

## Step 2: Summarize wet-snow rasters to `wet_snow_summary.csv`

Run `oa-da-wetsnow` to build the wet-snow equivalent summary table under
`/data/rofental/obs/summaries/project_2022_2023/wet_snow_summary.csv`. This step
uses `obs.wetsnow.dir`, `obs.wetsnow.product_tag`, `obs.wetsnow.classes.*`, and
the same land-cover exclusions from the project YAML. The output makes wet-snow
date coverage explicit and later feeds `oa-da-wetsnow-project`.

**🟢 Run this command:**

```bash
oa-da-wetsnow \
  --input-dir /data/rofental/obs/wetsnow \
  --project-label project_2022_2023 \
  --setup-dir /data/rofental \
  --overwrite
```

Use the summary to confirm that the dates and fractions make sense before adding or
editing wet-snow events in `assimilation_events`. Expect `wet_snow_fraction`,
support counts, and `source`; with uncertainty enabled, expect the same `unc_*`
columns that appear in the SCF summary.

Reference snippet from
`/data/rofental/obs/summaries/project_2022_2023/wet_snow_summary.csv`:

| date | region_id | wet_snow_fraction | n_valid | n_wet | source |
| --- | --- | --- | --- | --- | --- |
| 2023-03-12 |  | 0.02 | 156982 | 3453 | WSM_S1A_SAR_track117_2023_03_12_17_07_24.tif |
| 2023-03-16 |  | 0.03 | 158953 | 4667 | WSM_S1A_SAR_track168_2023_03_16_05_27_37.tif |
| 2023-03-24 |  | 0.47 | 156982 | 73218 | WSM_S1A_SAR_track117_2023_03_24_17_07_24.tif |
| 2023-03-28 |  | 0.27 | 158953 | 42301 | WSM_S1A_SAR_track168_2023_03_28_05_27_38.tif |
| 2023-04-05 |  | 0.06 | 156982 | 9750 | WSM_S1A_SAR_track117_2023_04_05_17_07_24.tif |

## Step 3: Build the project step skeleton

`oa-da-project-skeleton` translates `start_date`, `end_date`, and
`data_assimilation.assimilation_events` into the `step_*` folder structure under
the project directory. Those folders define the time windows that later receive the
per-step observation CSVs and the data assimilation outputs.

**🟢 Run this command:**

```bash
oa-da-project-skeleton \
  --setup-dir /data/rofental \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --overwrite \
  --log-level INFO
```

The first folders in the tutorial project look like this:

```text
step_00_init
step_01_20221122-20221222
step_02_20221222-20230122
step_03_20230122-20230221
step_04_20230221-20230322
step_05_20230322-20230426
step_06_20230426-20230503
step_07_20230503-20230518
step_08_20230518-20230523
step_09_20230523-20230526
step_10_20230526-20230630
```

Read that structure as a direct translation of the project period plus the
configured assimilation dates. If you edit `assimilation_events`, rebuild the
step skeleton before regenerating per-step observation CSVs.

{: .warning }
> Rerun `oa-da-project-skeleton --overwrite` every time you change
> `assimilation_events`. Step windows and observation dates must stay aligned.

## Step 4: Create per-step SCF observation CSVs

`oa-da-scf` matches rows from `scf_summary.csv` to the configured SCF events and
writes one-row `obs_scf_*.csv` files into the matching `steps/*/obs/` folders.
This is more than a file copy: it validates that the event date exists in the
summary, belongs to the expected step window, and matches the configured SCF
product tag.

**🟢 Run this command:**

```bash
oa-da-scf \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --summary-csv /data/rofental/obs/summaries/project_2022_2023/scf_summary.csv \
  --overwrite \
  --log-level INFO
```

After the command, expect one SCF observation CSV per configured SCF event under
`/data/rofental/projects/project_2022_2023/steps/*/obs/`. A typical file is
`/data/rofental/projects/project_2022_2023/steps/step_00_init/obs/obs_scf_SNOWCOVER_20230101.csv`.
This one-row file is the actual SCF input consumed later during data assimilation.
When uncertainty is enabled and matching layers exist, the generated file also
contains `unc_mean`, `unc_min`, `unc_max`, and `unc_n_valid`.

Reference snippet from a generated SCF observation file:

| date | n_valid | n_snow | scf | cloud_fraction | source |
| --- | --- | --- | --- | --- | --- |
| 2023-01-01 | 106750 | 106750 | 1.00 | 0.00 | s2_fsc_snowflake_rofental_2023_01_01.tif |

## Step 5: Create per-step wet-snow observation CSVs

`oa-da-wetsnow-project` applies the same alignment logic to wet-snow events. It
matches rows from `wet_snow_summary.csv` to the configured wet-snow
`assimilation_events` and writes one-row `obs_wet_snow_*.csv` files into the
corresponding `steps/*/obs/` folders.

**🟢 Run this command:**

```bash
oa-da-wetsnow-project \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --summary-csv /data/rofental/obs/summaries/project_2022_2023/wet_snow_summary.csv \
  --overwrite \
  --log-level INFO
```

Station snow-depth assimilation needs no separate preprocessing command. It reads
`obs/stations/*.csv` together with `obs/stations/stations_da_metadata.csv`
directly during the project run, so once the step skeleton exists the station HS
events are ready.

After this step, the project has an explicit step structure, project-level SCF and
wet-snow summaries, and per-step observation inputs for every configured
fraction assimilation event. At that point the data assimilation run becomes
reproducible: the exact observation artifact consumed by each SCF or wet-snow step
is visible on disk and easy to inspect, while station HS uses the setup-level
station files and metadata you already reviewed above.

![Preprocessing observation flow diagram]({{ site.baseurl }}/assets/images/tutorial/diagrams/preprocessing-observation-flow.svg)

_Flow from input SCF and wet-snow rasters to project summary CSVs, then to the
per-step observation CSVs consumed by each data assimilation step._

Reference snippet from a generated wet-snow observation file:

| date | wet_snow_fraction | n_valid | n_wet | source |
| --- | --- | --- | --- | --- |
| 2023-05-11 | 0.89 | 156982 | 139793 | WSM_S1A_SAR_track117_2023_05_11_17_07_26.tif |

## When preprocessing fails

Most preprocessing failures come from four causes. A selected assimilation date may
be missing from `scf_summary.csv` or `wet_snow_summary.csv`, which means the event
list and the available observation dates do not match. The event `product` can
also disagree with `obs.snowcover.product_tag` or `obs.wetsnow.product_tag`. If
step windows were built from an older event list, rerun
`oa-da-project-skeleton --overwrite` and then regenerate the per-step observation
CSVs. Finally, if support is unexpectedly low or empty, inspect the land-cover
grid and the classes excluded by `data_assimilation.landcover_mask`.

For product-specific preprocessing details, see the
[Observation Processing Guide]({{ site.baseurl }}{% link guides/observations.md %}).
For debugging failed runs, see
[Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}).

## What you should have before the project run

{: .checks }
> Before moving on to the project run, verify that these outputs exist:
> - `obs/summaries/project_2022_2023/scf_summary.csv`
> - `obs/summaries/project_2022_2023/wet_snow_summary.csv`
> - `projects/project_2022_2023/steps/step_*`
> - `steps/.../obs/obs_scf_<PRODUCT>_YYYYMMDD.csv`
> - `steps/.../obs/obs_wet_snow_<PRODUCT>_YYYYMMDD.csv`

These are the direct observation inputs consumed by the data assimilation project
run in the next chapter.
