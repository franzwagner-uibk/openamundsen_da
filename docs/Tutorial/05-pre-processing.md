---
layout: default
title: 5. Pre-processing
parent: Tutorial
nav_order: 5
permalink: /tutorial/pre-processing/
---

# 5. Pre-processing

This chapter covers the **full observation preprocessing workflow** for the tutorial
case study (`Rofental`) starting from the shipped raster observations in the example
setup.

The goal is to make the preprocessing logic transparent and reproducible:

1. summarize raw SCF rasters to `scf_summary.csv`,
2. summarize raw wet-snow rasters to `wet_snow_summary.csv`,
3. build the project step skeleton from `assimilation_events`,
4. create per-step one-row observation CSVs used by the DA pipeline.

The tutorial workflow starts from the shipped **raw observation rasters**. Summary CSVs
and per-step observation CSVs are generated in this chapter so users learn the full
preprocessing workflow end to end.

This is the key bridge between satellite products and the DA framework: raw rasters are transformed into validated, step-aligned CSV inputs.

---

## Step-by-step flow on this page

{: .step }
> Follow this chapter in order; each preprocessing step creates inputs needed by the next one.

Run the chapter in this order:

1. check raw inputs and preprocessing-relevant project YAML keys (orientation)
2. run the **five required preprocessing commands** (mandatory)
3. compare generated files with the shown snippets/paths (recommended)
4. use optional CLI checks only if something looks wrong

**Mandatory commands only (copy-paste path):** `oa-da-snowcover` -> `oa-da-wetsnow` -> `project_skeleton` -> `oa-da-scf` -> `oa-da-wetsnow-project`

---

## Before you start

This chapter assumes you already:

- started the interactive tutorial container shell from [2. Dependencies]({{ site.baseurl }}{% link Tutorial/02-dependencies.md %}),
- copied the bundled Rofental example to your mounted workspace (`/data/rofental`).

{: .references }
> - [Framework]({{ site.baseurl }}{% link Tutorial/04-framework.md %}) for setup/project/step concepts
> - [Observation Processing Guide]({{ site.baseurl }}{% link guides/observations.md %}) for product details and CLI options
> - [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) for all preprocessing commands

<details markdown="block">
  <summary>Why this chapter comes before the project run</summary>

The project pipeline expects per-step observation CSVs in each `step_*/obs/` folder.
Those files do not exist until you:

1. summarize raw rasters (`oa-da-snowcover`, `oa-da-wetsnow`),
2. generate the step skeleton (`project_skeleton`),
3. align summaries to assimilation events (`oa-da-scf`, `oa-da-wetsnow-project`).
</details>

All command blocks below are executed **inside the running tutorial container shell**. Once the shell is started in chapter 2, the commands are identical on Linux, macOS, and Windows (WSL/PowerShell users type them inside the container).

Most file checks are shown as **paths + snippets** to reduce command noise. If you are following the tutorial for the first time, focus on the five preprocessing commands and compare your generated files with the shown references.

---

## Inspect the raw observation inputs (Rofental example)

The Rofental example contains:

- `obs/snowcover/` - Sentinel-2 FSC rasters (GeoTIFF)
- `obs/wetsnow/` - Sentinel-1 wet-snow mask rasters (GeoTIFF)
- `obs/stations/` - station snow observations for validation plots

What to expect:

- many FSC rasters across the season (more than the final assimilation dates),
- wet-snow rasters from Sentinel-1 tracks,
- at least the two tutorial validation stations (`latschbloder`, `proviantdepot`).

Reference snippet (Rofental example filenames):

```text
/data/rofental/obs/snowcover/
  s2_fsc_snowflake_rofental_2022_10_03.tif
  s2_fsc_snowflake_rofental_2022_10_05.tif
  s2_fsc_snowflake_rofental_2022_10_08.tif
  ...

/data/rofental/obs/wetsnow/
  WSM_S1A_SAR_track117_2023_03_12_17_07_24.tif
  WSM_S1A_SAR_track117_2023_03_24_17_07_24.tif
  WSM_S1A_SAR_track168_2023_03_28_05_27_38.tif
  ...
```

Reference CSV snippet (station snow observations)

File path: `/data/rofental/obs/stations/proviantdepot.csv`

| time | snow_depth | swe |
| --- | --- | --- |
| 2022-10-01 00:00:00 | 0.09849999999999999 | 15.116667 |
| 2022-10-01 01:00:00 | 0.09883333333333333 | 14.783333 |
| 2022-10-01 02:00:00 | 0.09949999999999999 | 14.783333 |

Why this matters:

- the tutorial project assimilates only a subset of dates,
- but date selection is driven by the **summary tables** derived from these raw rasters.

<details markdown="block">
  <summary>Optional CLI check (list raw observation files)</summary>

```bash
echo "Snow-cover rasters:";
ls -1 /data/rofental/obs/snowcover | head -10;
echo;
echo "Wet-snow rasters:";
ls -1 /data/rofental/obs/wetsnow | head -10;
echo;
echo "Station observation files:";
ls -1 /data/rofental/obs/stations
```


</details>

---

## Inspect the project configuration that drives preprocessing

The preprocessing commands read product tags, classes, and paths from the project YAML.

Relevant sections in the tutorial project:

- `obs.snowcover` (SCF classes + product tag),
- `obs.wetsnow` (wet-snow classes + product tag),
- `data_assimilation.landcover_mask`,
- `data_assimilation.assimilation_events`.

The preprocessors are configuration-driven and intentionally fail-fast. If required
class mappings or product tags are missing, preprocessing aborts instead of guessing.

Reference YAML snippet (project config driving preprocessing, selected sections)

File path: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`

```yaml
start_date: "2022-10-01"
end_date: "2023-06-30"

obs:
  stations:
    dir: obs/stations # station CSVs used for validation plots (SWE obs expected in mm)
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
  prior_forcing:
    ensemble_size: 10
  wet_snow:
    classification_threshold_percent: 0.5
```

{: .warning }
> **Required before continuing:** open `/data/rofental/projects/project_2022_2023/project_2022_2023.yml` and verify that these tutorial-baseline values are present before preprocessing.
>
> In particular, confirm the wet-snow class mapping:
> - `wet: [110]`
> - `valid: [110, 125, 200, 210]`
> - `exclude: [200, 210]`
>
> If these keys differ (for example in an older local copy of the example), update the file and then continue with the preprocessing commands below.

Edit the file in your preferred editor on the host machine (it is inside your mounted tutorial workspace) or from inside the container. The tutorial path inside the container is:
`/data/rofental/projects/project_2022_2023/project_2022_2023.yml`

{: .warning }
> If you change class mappings or product tags, regenerate summaries before creating per-step observation CSVs.

<details markdown="block">
  <summary>Optional CLI check (inspect project YAML in the container)</summary>

```bash
sed -n "1,220p" /data/rofental/projects/project_2022_2023/project_2022_2023.yml
```

</details>

---

## (Optional but recommended) Clean previously generated preprocessing outputs

If you are rerunning this chapter, remove previously generated summary and per-step
observation files first so the workflow stays reproducible.

```bash
rm -f /data/rofental/obs/summaries/project_2022_2023/scf_summary.csv
rm -f /data/rofental/obs/summaries/project_2022_2023/wet_snow_summary.csv
find /data/rofental/projects/project_2022_2023/steps -type f \
  \\( -name "obs_scf_*.csv" -o -name "obs_wet_snow_*.csv" \\) -delete 2>/dev/null || true
```


Expected state after cleanup (if you ran it):

- `/data/rofental/obs/summaries/project_2022_2023/scf_summary.csv` removed (or not present yet)
- `/data/rofental/obs/summaries/project_2022_2023/wet_snow_summary.csv` removed (or not present yet)
- old per-step `obs_scf_*.csv` / `obs_wet_snow_*.csv` files removed from `steps/*/obs/`

---

## Step 1: Summarize snow-cover rasters to `scf_summary.csv`

`oa-da-snowcover` reads raw FSC rasters from `--input-dir`, applies ROI/masking and class mapping from `--setup-dir` + project YAML, writes the project summary under `obs/summaries/<project_label>/`, and `--overwrite` replaces an existing summary file.

Purpose in this tutorial:

- quality control / date selection,
- per-step observation file generation (`oa-da-scf`),
- reproducible assimilation date matching.

```bash
oa-da-snowcover \
--input-dir /data/rofental/obs/snowcover \
--project-label project_2022_2023 \
--setup-dir /data/rofental \
--overwrite
```


**Configuration used (project YAML)**  
File path: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`  
Relevant keys: `obs.snowcover.dir`, `obs.snowcover.product_tag`, `obs.snowcover.classes.*`, `data_assimilation.landcover_mask.*`

**Where this command writes by default**  
`oa-da-snowcover` writes to `obs/summaries/<project_label>/scf_summary.csv` unless you override the output root. In this tutorial: `/data/rofental/obs/summaries/project_2022_2023/scf_summary.csv`.

What this does:

- reads rasters from `obs/snowcover/`,
- clips to the tutorial ROI,
- applies land-cover exclusions (from project config),
- interprets FSC/cloud/water/nodata classes from `obs.snowcover.classes`,
- writes `scf_summary.csv`.

Expected output file after running the command:

- `/data/rofental/obs/summaries/project_2022_2023/scf_summary.csv`

Expected content (columns may evolve slightly over time):

- date
- ROI or region identifier
- valid pixel count / snow pixel count
- SCF fraction
- cloud fraction
- source/product metadata

Use this summary table to inspect availability and quality before choosing or changing
assimilation dates in `assimilation_events`.

<details markdown="block">
  <summary>What to review in <code>scf_summary.csv</code> before trusting your event dates</summary>

Typical checks:

- the date exists and matches the expected acquisition day,
- cloud fraction is acceptable for your use case,
- valid pixel count is large enough to support assimilation,
- the SCF value is plausible for the season and ROI.
</details>

Reference CSV snippet (SCF summary)

File path: `/data/rofental/obs/summaries/project_2022_2023/scf_summary.csv`

| date | region_id | n_valid | n_snow | scf | cloud_fraction | source |
| --- | --- | --- | --- | --- | --- | --- |
| 2022-10-03 |  | 88455 | 46001 | 0.5200462382002148 | 0.0 | s2_fsc_snowflake_rofental_2022_10_03.tif |
| 2022-10-05 |  | 78631 | 23973 | 0.30487988198038946 | 0.0 | s2_fsc_snowflake_rofental_2022_10_05.tif |
| 2022-10-08 |  | 33976 | 1078 | 0.03173769719802213 | 0.0 | s2_fsc_snowflake_rofental_2022_10_08.tif |
| 2022-10-13 |  | 12530 | 306 | 0.024387869114126097 | 0.0 | s2_fsc_snowflake_rofental_2022_10_13.tif |
| 2022-10-18 |  | 72228 | 2255 | 0.031222656033671154 | 0.0 | s2_fsc_snowflake_rofental_2022_10_18.tif |

Exact values depend on ROI, masking, and class mapping. The column structure and value ranges should look similar.

<details markdown="block">
  <summary>Optional CLI check (confirm SCF summary file in the container)</summary>

```bash
ls -lh /data/rofental/obs/summaries/project_2022_2023/scf_summary.csv
```

</details>

How to read this SCF summary snippet:

- `n_valid` is the ROI support after masking/class filtering,
- `n_snow` is the snow-class support used to compute `scf`,
- `scf` is the fraction used later for SCF DA events,
- `cloud_fraction` helps you judge whether a date is suitable for assimilation.

---

## Step 2: Summarize wet-snow rasters to `wet_snow_summary.csv`

`oa-da-wetsnow` reads raw wet-snow rasters from `--input-dir`, uses wet/valid/exclude class mappings from the project config (`--setup-dir`), writes `wet_snow_summary.csv` under `obs/summaries/<project_label>/`, and `--overwrite` replaces an existing file.

Purpose in this tutorial:

- quality control of wet-snow date coverage
- per-step wet-snow observation file generation (`oa-da-wetsnow-project`)
- reproducible wet-snow event/date matching

```bash
oa-da-wetsnow \
--input-dir /data/rofental/obs/wetsnow \
--project-label project_2022_2023 \
--setup-dir /data/rofental \
--overwrite
```

**Configuration used (project YAML)**  
File path: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`  
Relevant keys: `obs.wetsnow.dir`, `obs.wetsnow.product_tag`, `obs.wetsnow.classes.*`, `data_assimilation.landcover_mask.*`

**Where this command writes by default**  
`oa-da-wetsnow` writes to `obs/summaries/<project_label>/wet_snow_summary.csv` unless you override the output root. In this tutorial: `/data/rofental/obs/summaries/project_2022_2023/wet_snow_summary.csv`.


What this does:

- reads wet-snow rasters from `obs/wetsnow/`,
- applies wet/valid/exclude class mapping from `obs.wetsnow.classes`,
- clips to the ROI and applies land-cover exclusions,
- writes `wet_snow_summary.csv`.

Expected output file after running the command:

- `/data/rofental/obs/summaries/project_2022_2023/wet_snow_summary.csv`

Why this matters:

- wet-snow acquisitions are sparser and track-dependent,
- the summary makes date coverage explicit and drives the per-step wet-snow obs files.

Reference CSV snippet (wet-snow summary)

File path: `/data/rofental/obs/summaries/project_2022_2023/wet_snow_summary.csv`

| date | region_id | wet_snow_fraction | n_valid | n_wet | source |
| --- | --- | --- | --- | --- | --- |
| 2023-03-12 |  | 0.022 | 156982 | 3453 | WSM_S1A_SAR_track117_2023_03_12_17_07_24.tif |
| 2023-03-16 |  | 0.0294 | 158953 | 4667 | WSM_S1A_SAR_track168_2023_03_16_05_27_37.tif |
| 2023-03-24 |  | 0.4664 | 156982 | 73218 | WSM_S1A_SAR_track117_2023_03_24_17_07_24.tif |
| 2023-03-28 |  | 0.2661 | 158953 | 42301 | WSM_S1A_SAR_track168_2023_03_28_05_27_38.tif |
| 2023-04-05 |  | 0.0621 | 156982 | 9750 | WSM_S1A_SAR_track117_2023_04_05_17_07_24.tif |

Wet-snow coverage is typically sparser than SCF. Sparse wet-snow dates are expected and should not be treated as a preprocessing error by default.

<details markdown="block">
  <summary>Optional CLI check (confirm wet-snow summary file in the container)</summary>

```bash
ls -lh /data/rofental/obs/summaries/project_2022_2023/wet_snow_summary.csv
```

</details>

How to read this wet-snow summary snippet:

- `wet_snow_fraction` is the ROI fraction interpreted as wet snow,
- `n_valid` and `n_wet` show the spatial support behind the fraction,
- `source` often encodes the Sentinel-1 track and acquisition timestamp (useful when comparing dates/tracks).

---

## Step 3: Build the project step skeleton (`step_*` folders)

`project_skeleton` reads `start_date`, `end_date`, and `assimilation_events` from the project YAML, generates `step_*` folders under `--project-dir`, uses `--setup-dir` for context, and `--overwrite` rebuilds the step structure after event edits.

This step builds the project step structure from:

- `start_date`, `end_date`
- `data_assimilation.assimilation_events`

```bash
python -m openamundsen_da.pipeline.project_skeleton \
  --setup-dir /data/rofental \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --overwrite \
  --log-level INFO
```

**Configuration used (project YAML)**  
File path: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`  
Relevant keys: `start_date`, `end_date`, `data_assimilation.assimilation_events[*].date|variable|product`


What to expect:

- `step_00_init`
- one step per assimilation window boundary
- the final step (after the last DA event) has no assimilation

Reference snippet (first step folders in the tutorial project):

```text
step_00_init
step_01_20221003-20221025
step_02_20221025-20221119
step_03_20221119-20230101
step_04_20230101-20230309
step_05_20230309-20230328
step_06_20230328-20230405
step_07_20230405-20230416
...
```

Read this step-list snippet as a direct translation of the configured project period plus `assimilation_events`; it is the structure that later receives the per-step observation CSVs and DA outputs.

{: .warning }
> If you edit `assimilation_events`, rerun `project_skeleton` before regenerating per-step
> observation CSVs. Step windows and event dates must match exactly.

This fail-fast behavior prevents silent mismatches between event dates and step windows.

{: .references }
> - [Framework]({{ site.baseurl }}{% link Tutorial/04-framework.md %}) for why `assimilation_events` define the step structure
> - [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) for editing `assimilation_events`

---

## Step 4: Create per-step SCF observation CSVs (`oa-da-scf`)

`oa-da-scf` matches SCF rows from `--summary-csv` to SCF `assimilation_events` in `--project-dir`, writes one-row `obs_scf_*.csv` files into `steps/*/obs/`, and `--overwrite` regenerates them after configuration or date changes.

This step maps rows from `scf_summary.csv` to the configured SCF assimilation dates and writes one-row observation CSVs into the corresponding `step_*/obs/` folders.

```bash
oa-da-scf \
--project-dir /data/rofental/projects/project_2022_2023 \
--summary-csv /data/rofental/obs/summaries/project_2022_2023/scf_summary.csv \
--overwrite \
--log-level INFO
```

**Configuration used (project YAML)**  
File path: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`  
Relevant keys: `data_assimilation.assimilation_events[*]` (SCF events), `obs.snowcover.product_tag`


Expected outputs after running the command:

- one SCF observation CSV per configured SCF DA event
- files written under: `/data/rofental/projects/project_2022_2023/steps/*/obs/obs_scf_*.csv`

What this command validates (fail-fast):

- SCF events exist in `assimilation_events`
- each SCF event date exists in `scf_summary.csv`
- each SCF event date belongs to the associated step window
- product tag mapping is configured correctly

<details markdown="block">
  <summary>Why <code>oa-da-scf</code> is more than a file copy</summary>

`oa-da-scf` verifies consistency between:

- project configuration (`assimilation_events`, product tags),
- summary table dates,
- and the generated step windows.

This is an important quality gate in the workflow.
</details>

Reference CSV snippet (generated per-step SCF observation)

File path: `/data/rofental/projects/project_2022_2023/steps/step_00_init/obs/obs_scf_SNOWCOVER_20221003.csv`

| date | n_valid | n_snow | scf | cloud_fraction | source |
| --- | --- | --- | --- | --- | --- |
| 2022-10-03 | 88455 | 46001 | 0.5200462382002148 | 0.0 | s2_fsc_snowflake_rofental_2022_10_03.tif |

This one-row file is what the DA step consumes directly for the SCF event, so it is the key place to verify date, product tag alignment (via filename), and the summary values passed into the run.

---

## Step 5: Create per-step wet-snow observation CSVs (`oa-da-wetsnow-project`)

`oa-da-wetsnow-project` applies the same alignment logic for wet-snow events: it reads `--summary-csv`, matches rows to wet-snow `assimilation_events` in `--project-dir`, writes `obs_wet_snow_*.csv` into `steps/*/obs/`, and `--overwrite` refreshes existing files.

This step applies the same alignment logic for wet-snow assimilation dates.

```bash
oa-da-wetsnow-project \
--project-dir /data/rofental/projects/project_2022_2023 \
--summary-csv /data/rofental/obs/summaries/project_2022_2023/wet_snow_summary.csv \
--overwrite \
--log-level INFO
```

**Configuration used (project YAML)**  
File path: `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`  
Relevant keys: `data_assimilation.assimilation_events[*]` (wet-snow events), `obs.wetsnow.product_tag`


Expected outputs after running the command:

- one wet-snow observation CSV per configured wet-snow DA event
- files written under: `/data/rofental/projects/project_2022_2023/steps/*/obs/obs_wet_snow_*.csv`

At this point, the project is ready for execution:

- step structure exists,
- SCF and wet-snow summaries exist,
- per-step observation CSVs exist,
- project config and setup config are aligned.

At this point the DA run becomes reproducible: the observation inputs consumed by each step are explicit and inspectable.

{: .step }
> Preprocessing observation flow (conceptual)

![Preprocessing observation flow diagram]({{ site.baseurl }}/assets/images/tutorial/diagrams/preprocessing-observation-flow.svg)

_Flow from raw SCF/wet-snow rasters to project summary CSVs, then to per-step observation CSVs consumed by each DA step._

Reference CSV snippet (generated per-step wet-snow observation)

File path: `/data/rofental/projects/project_2022_2023/steps/step_05_20230309-20230328/obs/obs_wet_snow_WETSNOW_20230328.csv`

| date | wet_snow_fraction | n_valid | n_wet | source |
| --- | --- | --- | --- | --- |
| 2023-03-28 | 0.2661 | 158953 | 42301 | WSM_S1A_SAR_track168_2023_03_28_05_27_38.tif |

This wet-snow per-step file plays the same role as the SCF per-step CSV: it is the explicit DA input artifact for that event date and should match the selected summary row.

## Visual sanity teaser (what this preprocessing enables)

If preprocessing succeeded, the later results plot `fraction_timeseries.png` can show the
expected SCF and wet-snow observation markers at the configured event dates.

![Fraction time series (Rofental tutorial reference run)]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_ens10/fraction_timeseries.png)

_Preview only: detailed interpretation of this plot comes in [7. Results and diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %})._

What this preview confirms conceptually:

- SCF and wet-snow observations exist in the project timeline,
- multiple DA events are visible across the season,
- the preprocessing outputs are not just intermediate files: they directly shape the DA workflow.

---

## Common preprocessing failure modes (and what they mean)

{: .warning }
> Use these diagnostics when preprocessing outputs are missing or inconsistent.

<details markdown="block">
  <summary>Expand troubleshooting list</summary>

### 1. Missing summary row for an assimilation date

Meaning:

- `assimilation_events` contains a date that is not present in `scf_summary.csv` or `wet_snow_summary.csv`.

Typical fix:

- update `assimilation_events`, or
- regenerate the summary from the correct raw data.

### 2. Product tag mismatch

Meaning:

- event `product` does not match the configured product tag in project YAML (`obs.snowcover.product_tag` / `obs.wetsnow.product_tag`).

Typical fix:

- correct the project YAML (preferred),
- avoid hardcoded fallbacks in code.

### 3. Step/date mismatch after editing events

Meaning:

- step windows were created from an older `assimilation_events` list.

Typical fix:

- rerun `project_skeleton --overwrite`,
- rerun `oa-da-scf` and `oa-da-wetsnow-project`.

### 4. Too many masked pixels / empty ROI support

Meaning:

- land-cover exclusions remove too much of the ROI (or all of it).

Typical fix:

- check the land-cover grid,
- review `data_assimilation.landcover_mask.classes_to_exclude`.

</details>

{: .references }
> - [Observation Processing Guide]({{ site.baseurl }}{% link guides/observations.md %}) for product-specific preprocessing details
> - [Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) for debugging failed preprocessing runs

---

## What we created in this chapter

{: .checks }
> Verify these outputs before starting the project run in the next chapter.

Key outputs:

- project summaries:
  - `obs/summaries/project_2022_2023/scf_summary.csv`
  - `obs/summaries/project_2022_2023/wet_snow_summary.csv`
- project step skeleton:
  - `projects/project_2022_2023/steps/step_*`
- per-step assimilation inputs:
  - `steps/.../obs/obs_scf_<PRODUCT>_YYYYMMDD.csv`
  - `steps/.../obs/obs_wet_snow_<PRODUCT>_YYYYMMDD.csv`

These are the direct observation inputs consumed by the DA project run.

---

## Next step

{: .references }
> Continue to the project execution chapter after the preprocessing outputs look correct.

Continue with [6. Running the project]({{ site.baseurl }}{% link Tutorial/06-running-the-project.md %}) to:

- run a full DA project with the project pipeline,
- understand the relationship between granular commands and the pipeline,
- and validate that the run completed successfully.
