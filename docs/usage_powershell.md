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
  --step-dir        $step
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
