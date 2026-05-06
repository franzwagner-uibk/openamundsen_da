# Testing and CI Runbook

This folder documents the current regression-testing setup for `openamundsen_da`.

## Recommended Workflow (Single Developer, Private Repo)

1. Start from `main` and create a feature branch.
2. Implement changes on the feature branch.
3. Push branch to GitHub.
4. Wait for `CI and Publish` -> `Unit and Integration Tests` to finish green.
5. Merge branch into `main` only if green.
6. After merge, verify `main` pipeline is green (`tests` + `publish`).
7. Delete merged feature branch.

Important: because the repository is private and you do not use paid GitHub plan features for protected rulesets, this process is enforced by discipline, not hard merge blocking.

## What Is Automated

Workflow file: `.github/workflows/ci.yml`

- Triggered on:
  - Pull requests to `main` (except docs-only changes)
  - Pushes to any branch (except docs-only changes)
  - Manual dispatch (`workflow_dispatch`)
- Job `Ruff Lint`:
  - Runs fast fatal lint checks before tests
  - Script: `scripts/ci/run_lint.sh`
- Job `Unit and Integration Tests`:
  - Runs on self-hosted runner labels: `self-hosted, linux, x64, oa-da`
  - Builds a CI Docker image from current commit
  - Runs unit tests with `pytest` via `scripts/ci/run_unit_tests.sh`
  - Runs full single-domain example integration test via `scripts/ci/run_integration_tests.sh`
  - Runs trimmed sub-domain integration test via `scripts/ci/run_integration_tests_subdomain.sh`
  - Runs trimmed plain openAMUNDSEN model sub-domain integration test via `scripts/ci/run_integration_tests_model_subdomain.sh`
  - Uploads integration artifacts on failure (log + example setup outputs)
- Job `Build and Push GHCR Image`:
  - Runs only on push to `main`
  - Depends on successful `tests` job
  - Logs in to GHCR using `GHCR_PAT`
  - Builds and pushes image tags:
    - `main-YYYYMMDD`
    - short SHA
    - `latest`

## What You Must Do Manually

- Follow feature-branch workflow (do not develop directly on `main`).
- Monitor Actions results after push/merge.
- Keep the self-hosted runner online and healthy.
- Keep GitHub secret `GHCR_PAT` valid.
- Tune integration validation rules if expected output/warnings change.

## Current Test Stack

### Rough CI test steps (execution order)

1. `Ruff Lint` job runs (`scripts/ci/run_lint.sh`) and checks fatal lint classes.
2. `Unit and Integration Tests` job starts only after lint is green.
3. CI image is built from current commit.
4. Unit tests are executed with `pytest` (`scripts/ci/run_unit_tests.sh`).
5. Integration run clones `examples/rofental` into a temporary workspace.
6. Setup skeleton is generated for the shipped `project_2022_2023`.
7. SCF and wet-snow per-step observation CSVs are prepared.
8. Full example pipeline is executed (`oa-da-project` equivalent module call).
9. Integration validator checks logs, outputs, plots, scientific benchmark outputs, and weight sanity.
10. Example-specific diagnostics such as station HS weights and setup weights overview are checked.
11. Sub-domain integration validator checks manifest status and project-level results outputs.
12. Model sub-domain integration validator checks model manifest status and merged grid output.
13. If integration fails, log and example outputs are uploaded as CI artifacts.
14. On push to `main` only: publish job builds and pushes GHCR image.

### Unit tests

Location: `tests/unit/`

Current coverage areas include:
- config merging behavior
- data assimilation events parsing/handling
- land-cover path and resolution behavior
- setup skeleton behavior
- statistics helper functions
- assimilation requirement validation prechecks

Runner command wrapper: `scripts/ci/run_unit_tests.sh`

Framework/tooling config:
- unit test dependencies and lint tool extras: `pyproject.toml` (`[project.optional-dependencies].test`)

### What is tested (rough checklist)

- Lint gate:
  - fatal syntax/logic lint classes (`E9`, `F63`, `F7`, `F82`)
- Unit logic:
  - config merge behavior
  - assimilation event parsing
  - land-cover path handling
  - setup skeleton basics
  - statistics helpers
  - assimilation requirement validation prechecks
- Integration scenario behavior:
  - full `examples/rofental` orchestration on a temp clone
  - station HS, SCF, and wet-snow assimilation events from the shipped example
  - ensemble propagation, assimilation, resampling/rejuvenation path
  - plot generation path
- Integration output contracts:
  - required per-step SCF and wet-snow obs CSVs
  - required station HS diagnostics and weights
  - required SCF and wet-snow weights CSVs
  - required benchmark manifest, long-form tables, wide summary tables, summary, and core timeline plots
  - required model result artifacts (`point_*.csv`, `*.nc`)
  - required ROI mean SWE / snow-depth member CSVs
  - required plot outputs (forcing, results, assimilation, setup ESS timeline, setup weights overview including numbered continuation pages when the overview spans multiple A4-length pages)
  - weight CSV numeric sanity (valid range and sum to 1.0)
- Integration log contracts:
  - fail on fatal log patterns
  - fail on severe warnings
  - allow explicitly whitelisted benign warnings

### Integration regression test (full example setup)

Runner script: `scripts/ci/run_integration_tests.sh`

What it does:
- clones `examples/rofental` into a temp directory
- uses the shipped `project_2022_2023` configuration directly
- generates setup skeleton
- distributes SCF and wet-snow observations
- runs full setup pipeline
- validates logs and outputs with `scripts/ci/validate_trimmed_project.py`

Validation focuses on:
- no fatal log patterns (`ERROR`, `CRITICAL`, `Traceback`, `Exception`)
- no severe warnings (with explicit allow-list for known benign optional-observation warnings)
- expected outputs exist and are non-empty:
  - per-step SCF obs files
  - per-step wet-snow obs files
  - station HS diagnostics and weights
  - SCF weights CSVs
  - wet-snow weights CSVs
  - member SCF point time series
  - forcing plots, setup result plots, setup ESS timeline, setup weights overview, numbered setup weights continuation pages when present, and assimilation plots
  - benchmark outputs under `results/benchmark/` and the headline skill figure `results/plots/assim/scores/performance_scores.png`
  - project maps under `results/maps/` when `maps.yml` is present
  - shipped semi-independent benchmark view for `station_swe`
  - persistent point outputs (`point_*.csv`)
  - compact data assimilation grid output (`results/grids/da_output_grids.nc`)
- minimal weight sanity (weights exist, numeric, sum to `1.0`)

### Integration regression test (trimmed sub-domain)

Runner script: `scripts/ci/run_integration_tests_subdomain.sh`

What it does:
- clones `examples/subdomains` into a temp directory
- writes a trimmed project config (`project_ci_2022_2023`) under the sub-domain setup
- runs full sub-domain pipeline (`oa-da-subdomain pipeline`) with:
  - setup: `/data/subdomains` (the copied sub-domain setup root)
  - project: `/data/subdomains/projects/project_ci_2022_2023`
  - regions: `/data/subdomains/env/subdomains.gpkg` (8 avalanche-report subdomains)
  - station buffer: `10 km`
- validates logs and outputs with `scripts/ci/validate_trimmed_subdomain.py`

Validation focuses on:
- no fatal log patterns (`ERROR`, `CRITICAL`, `Traceback`, `Exception`)
- manifest exists and all sub-domains report `status=success`
- each sub-domain keeps non-empty point outputs (`point_*.csv`) in member results
- compact data assimilation grid output exists (`projects/<project>/results/grids/da_output_grids.nc`)
- project-level sub-domain reports exist (`projects/<project>/results/subdomain_*.csv`)

Failure artifacts:
- integration log and example setup outputs are copied to CI artifact directory when the run fails
- artifact upload is defined in `.github/workflows/ci.yml`

### Integration regression test (trimmed model sub-domain)

Runner script: `scripts/ci/run_integration_tests_model_subdomain.sh`

What it does:
- clones `examples/subdomains` into a temp directory
- shortens setup-level `start_date`/`end_date` in the temp copy
- runs the plain openAMUNDSEN model sub-domain pipeline (`oa-da-subdomain model-pipeline`) with:
  - setup: `/data/subdomains` (the copied sub-domain setup root)
  - regions: `/data/subdomains/env/subdomains.gpkg` (8 avalanche-report subdomains)
  - station buffer: `10 km`
- validates logs and outputs with `scripts/ci/validate_trimmed_model_subdomain.py`

Validation focuses on:
- no fatal log patterns (`ERROR`, `CRITICAL`, `Traceback`, `Exception`)
- manifest exists and all sub-domains report `status=success`
- each sub-domain has a successful `run_manifest.json`
- each sub-domain writes at least one model grid output under `subdomains/model/<id>/results/grids/`
- merged model grid output exists under `subdomains/model/results/grids/`
- generated model sub-domain folders do not contain DA `projects/` directories

Failure artifacts:
- integration log and model sub-domain outputs are copied to CI artifact directory when the run fails
- artifact upload is defined in `.github/workflows/ci.yml`

### Lint gate

Runner script: `scripts/ci/run_lint.sh`

What it checks:
- ruff fatal classes only (`E9`, `F63`, `F7`, `F82`) as a fast quality gate

## Self-Hosted Runner Setup (Ubuntu Test Machine)

Current intended environment:
- Ubuntu Linux
- Docker installed and daemon running
- Runner installed as a systemd service
- Runner labels include `oa-da` to match CI config
- Runner user must be able to run Docker commands

Required network direction:
- Outbound HTTPS from test machine to GitHub/GHCR endpoints
- No inbound exposure from internet is required for standard runner operation

## GitHub Repository Settings

Required:
- Actions enabled
- Secret `GHCR_PAT` present (package write access for GHCR publish job)

Recommended for this project:
- Keep `main` as release branch
- Use PR + green CI discipline before merge
- Avoid direct pushes to `main` in daily work

Note on private repo + free tier:
- ruleset enforcement and strict branch protection options may be limited
- therefore, process discipline is the effective quality gate

## Where Test Setup Is Configured

Main locations:
- CI workflow/jobs and sequencing: `.github/workflows/ci.yml`
- unit test runner command: `scripts/ci/run_unit_tests.sh`
- integration run recipes:
  - single-domain full example: `scripts/ci/run_integration_tests.sh`
  - sub-domain trimmed: `scripts/ci/run_integration_tests_subdomain.sh`
  - model sub-domain trimmed: `scripts/ci/run_integration_tests_model_subdomain.sh`
- integration validation logic:
  - single-domain: `scripts/ci/validate_trimmed_project.py`
  - sub-domain: `scripts/ci/validate_trimmed_subdomain.py`
  - model sub-domain: `scripts/ci/validate_trimmed_model_subdomain.py`
- lint command: `scripts/ci/run_lint.sh`
- test/lint optional dependencies: `pyproject.toml`

Single-domain example configuration details:
- source project copied for CI: `examples/rofental`
- source sub-domain setup copied for CI: `examples/subdomains`
- the full shipped example project is exercised in:
  - `scripts/ci/run_integration_tests.sh` (single-domain)
- trimmed dates, assimilation events, and ensemble sizes are still hard-coded in:
  - `scripts/ci/run_integration_tests_subdomain.sh` (sub-domain)
- max workers for CI integration runs are set in `.github/workflows/ci.yml` via:
  - `OA_DA_TEST_MAX_WORKERS` (single-domain, current value: `8`)
  - `OA_DA_SUBDOMAIN_TEST_MAX_WORKERS` / `OA_DA_SUBDOMAIN_TEST_INNER_WORKERS` (sub-domain, current values: `8` / `4`)
  - `OA_DA_MODEL_SUBDOMAIN_TEST_MAX_WORKERS` (model sub-domain, current value: `8`)
- feature branches are validated through pull requests to avoid duplicate branch-push and PR runs on the self-hosted runner.
- workflow concurrency is grouped by branch name, so new PR updates cancel stale in-progress checks for older commits.

If you want to change the CI test setup:
- edit the shipped example project or the relevant script directly:
  - `examples/rofental/projects/project_2022_2023/project_2022_2023.yml`
  - `scripts/ci/run_integration_tests.sh`
  - `scripts/ci/run_integration_tests_subdomain.sh`

## Failure Modes and Fast Checks

If CI fails, check in this order:

1. Runner availability
- runner shown as online/idle in GitHub settings
- systemd service active on test machine

2. Docker availability
- Docker daemon running
- runner user has Docker permissions

3. Test-stage failure class
- unit test assertion failure
- integration validation failure (log/output checks)
- environment/runtime issue (paths, resources, permissions)

4. Publish-stage failures (only on `main`)
- `GHCR_PAT` missing/expired/insufficient permissions
- GHCR login or push denied

## Local Reproduction (optional)

From repository root:
- run unit test wrapper: `bash scripts/ci/run_unit_tests.sh`
- run single-domain integration wrapper: `bash scripts/ci/run_integration_tests.sh`
- run sub-domain integration wrapper: `bash scripts/ci/run_integration_tests_subdomain.sh`

Use same scripts as CI to avoid drift between local and server behavior.
