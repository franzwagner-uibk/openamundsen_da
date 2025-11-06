# Running openamundsen_da from PowerShell

This guide shows common PowerShell commands to run the ensemble launcher and where to find logs/results on Windows.

## Quick Start

```powershell (ore miniconda prompt)
# Activate your environment first
conda activate openamundsen

# Define paths (adjust to your project layout)
$proj = "C:\Daten\PhD\openamundsen_da\examples\test-project"
$seas = "$proj\\propagation\\season_2017-2018"
$step = "$seas\\step_00_init"

# Launch prior ensemble with 16 workers at INFO level
& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
  --max-workers 16 `
  --log-level   INFO `
  --overwrite
```

- Parent progress appears in the console (start/finish per member + summary).
- Each worker writes its own log file to `<member_dir>\\logs\\member.log`.

## Satellite SCF Observation

Single-image, single-region SCF extraction with the new observer:

```powershell
# Activate env first (conda activate openamundsen)
$step   = "C:\Daten\PhD\openamundsen_da\examples\test-project\propagation\season_2017-2018\step_00_init"
$raster = "C:\data\modis\NDSI_Snow_Cover_20250315.tif"
$region = "C:\data\modis\region.gpkg"

oa-da-scf --raster $raster --region $region --step-dir $step
```

Input expectations:

- Raster: MODIS/Terra MOD10A1 Collection 6/6.1 exported to GeoTIFF, single band
  `NDSI_Snow_Cover`, values scaled 0..100, nodata already applied.
- AOI: exactly one polygon feature with field `region_id`, same CRS as the raster.
- Output: `<step>\obs\obs_scf_MOD10A1_YYYYMMDD.csv` with columns `date,region_id,scf`.
- Threshold override via step YAML (`step_00.yml`):

```yaml
scf:
  ndsi_threshold: 35
  region_id_field: region_id
```

- To write elsewhere: `oa-da-scf --raster ... --region ... --output C:\tmp\myscf.csv`.
- The tool fails if the AOI lacks `region_id`, contains multiple features, or no valid pixels remain after masking.

## MOD10A1 Preprocess

Batch-convert MODIS/Terra MOD10A1 Collection 6/6.1 HDF files into
`NDSI_Snow_Cover` GeoTIFFs ready for the SCF script:

```powershell
C:\Users\franz\miniconda3\condabin\conda.bat run -n gistools --no-capture-output `
  pip install -e C:\Daten\PhD\openamundsen_da

$input = "C:\Daten\PhD\openamundsen_da\examples\test-project\obs\MOD10A1_61_HDF"
$proj  = "C:\Daten\PhD\openamundsen_da\examples\test-project"
$aoi   = "$proj\env\GMBA_Inventory_L8_15422.gpkg"

C:\Users\franz\miniconda3\condabin\conda.bat run -n gistools --no-capture-output `
python -m openamundsen_da.observer.mod10a1_preprocess `
  --input-dir "C:\Daten\PhD\openamundsen_da\examples\test-project\obs\MOD10A1_61_HDF" `
  --project-dir "C:\Daten\PhD\openamundsen_da\examples\test-project" `
  --season-label season_2017-2018 `
  --aoi "C:\Daten\PhD\openamundsen_da\examples\test-project\env\GMBA_Inventory_L8_15422.gpkg" `
  --target-epsg 25832 `
  --resolution 500 `
  --max-cloud-fraction 0.1 `
  --overwrite
```

- Output folder: `$proj\obs\season_2017-2018\NDSI_Snow_Cover_YYYYMMDD.tif` (flat per season).
- Each accepted scene also yields `NDSI_Snow_Cover_YYYYMMDD_class.tif` (0=invalid, 1=no snow, 2=snow).
- The script maintains `$proj\obs\season_2017-2018\scf_summary.csv` with columns `date,region_id,scf,cloud_fraction,source`.
- The tool keeps only the `NDSI_Snow_Cover` layer, reprojects to `--target-epsg`,
  and clips to the AOI (bounding box by default, use `--no-envelope` for exact cutline).
- Use `--resolution` (e.g. `500`) to set output pixel size in meters; omit for native.
- Existing GeoTIFFs are skipped unless `--overwrite` is provided.
- Use `--max-cloud-fraction` (e.g. `0.1`) to reject scenes where more than 10% of usable pixels are
  flagged as cloudy (`NDSI_Snow_Cover == 200`).
- Adjust `--ndsi-threshold` if you need a different snow classification limit (default 40).

## Show Help

```powershell
python -m openamundsen_da.core.launch --help
```

## Use a Global Results Root

By default, each member writes results to `<member_dir>\\results`. To collect results under a single root:

```powershell
$resultsRoot = "D:\\oa_runs\\2025-11-04"
python -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
  --results-root $resultsRoot `
  --max-workers 6 `
  --log-level   INFO
```

Member results end up in `$resultsRoot\\<member_name>`.

## Increase Verbosity or Run Single-Threaded

```powershell
# More verbose logs (written to per-member log files)
python -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
  --max-workers 1 `
  --log-level   DEBUG
```

## Overwrite Existing Results

```powershell
python -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
--overwrite
```

## Build Prior Forcing Ensemble (Standalone)

The prior ensemble builder creates an open-loop forcing set and N perturbed members
for a specific step. It reads dates (inclusive) from the step YAML and prior parameters
from `project.yml` under `data_assimilation.prior_forcing`.

Required keys in `project.yml` (example):

```yaml
data_assimilation:
  prior_forcing:
    ensemble_size: 15
    random_seed: 42
    sigma_t: 0.5 # additive temperature stddev
    mu_p: 0.0 # log-space mean for precip factor
    sigma_p: 0.2 # log-space stddev for precip factor
```

Step YAML must define the time window:

```yaml
start_date: 2017-10-01 00:00:00
end_date: 2018-09-30 23:59:59
```

Run the builder directly (PowerShell):

```powershell
# Paths
$proj = "C:\Daten\PhD\openamundsen_da\examples\test-project"
$seas = "$proj\propagation\season_2017-2018"
$step = "$seas\step_00_init"
$meteo = "$proj\meteo"   # long-span original meteo (stations.csv + station CSVs)

& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.core.prior_forcing `
  --input-meteo-dir $meteo `
  --project-dir     $proj `
  --step-dir        $step `
  --overwrite
```

Output structure (created under the step):

```text
<step>\ensembles\prior\
  open_loop\
    meteo\   # date-filtered, unperturbed CSVs + stations.csv
    results\
  member_001\
    meteo\   # perturbed CSVs + stations.csv (T additive; P multiplicative)
    results\
    INFO.txt
  member_002\
    ...
```

Notes:

- CSV schema is strict: column `date` is required; `temp` and `precip` are optional.
- If a station file has `precip`, it must not contain negative values; otherwise the run aborts.
- Temperature and precipitation perturbations are constant per member across all stations/timesteps.
- Stations without `precip` receive temperature-only perturbations.

## Inspect Per‑Member Logs

```powershell
# List all member.log files under the ensemble
Get-ChildItem -Recurse -Filter member.log "$step\\ensembles\\prior" | Select-Object FullName

# Tail a specific member log
$log = "$step\\ensembles\\prior\\member_0001\\logs\\member.log"
Get-Content $log -Tail 50 -Wait
```

## Notes

- Quote paths containing spaces: `"C:\\path with spaces\\..."`.
- GDAL/PROJ and threading env are auto‑applied from `project.yml` when present; you usually don’t need to set them manually.
- `--log-level` controls both the parent process and the openAMUNDSEN logger inside workers.

## Helpers

# convert shp to gpkg

& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m osgeo.ogr2ogr -f GPKG "C:\path\to\output.gpkg" "C:\path\to\input.shp"

```powershell
# Paths

$inshp = "C:\\Daten\\PhD\\02-Daten\\15422\\misc\\GMBA_Inventory_L8_15422.shp"
$outgpkg = "C:\\Daten\\PhD\\openamundsen_da\\examples\\test-project\\env\\GMBA_Inventory_L8_15422.gpkg"

"C:\Users\franz\miniconda3\envs\openamundsen\Library\bin\ogr2ogr.exe" -f GPKG "$outgpkg" "$inshp"
```
