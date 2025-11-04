# Running openamundsen_da from PowerShell

This guide shows common PowerShell commands to run the ensemble launcher and where to find logs/results on Windows.

## Quick Start

```powershell
# Activate your environment first
conda activate openamundsen

# Define paths (adjust to your project layout)
$proj = "C:\\data\\oa_project"
$seas = "$proj\\seasons\\2017_2018"
$step = "$seas\\steps\\step_01"

# Launch prior ensemble with 4 workers at INFO level
python -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
  --max-workers 4 `
  --log-level   INFO
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
