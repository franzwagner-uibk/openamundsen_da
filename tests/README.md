# Testing and CI Runbook

This folder documents the current regression-testing setup for `openamundsen_da`.

## Recommended Workflow (Single Developer, Private Repo)

1. Start from `main` and create a feature branch.
2. Implement changes on the feature branch.
3. Push branch to GitHub.
4. Wait for all required `CI` jobs to finish green.
5. Merge branch into `main` only if green.
6. After merge, verify the trusted integration and `publish-edge` jobs are green.
7. Delete merged feature branch.

Important: the repository is public. Keep every CI check green before merging;
until required-check branch protection is configured, this process is enforced
by review discipline rather than a hard merge block.

## What Is Automated

Workflow file: `.github/workflows/ci.yml`

- Triggered on:
  - Pull requests to `main`
  - Pushes to `main`
  - Manual dispatch (`workflow_dispatch`)
- GitHub-hosted gates:
  - fatal and stale-code Ruff checks
  - strict wheel and sdist validation with SHA-256 checksums
  - no-dependency installed-wheel smoke tests on Linux, macOS and Windows with Python 3.11 through 3.14
  - curated CLI, navigation/link, frozen tutorial-asset and Jekyll documentation checks
  - pull-request dependency review
  - `linux/amd64` and `linux/arm64` release-image bootstrap checks and a critical-vulnerability image scan
- Job `Trusted Lenovo P8 integration`:
  - Runs on self-hosted runner labels: `self-hosted, linux, x64, oa-da`
  - Downloads the wheel that passed the package and portable gates
  - Builds a release-layout image that installs that wheel non-editably
  - Runs unit tests with `pytest` via `scripts/ci/run_unit_tests.sh`
  - Installs the tested wheel outside the checkout via `scripts/ci/run_wheel_smoke.sh`; verifies that only `openamundsen-da` is installed and exercises nested help and JSON errors
  - Runs full single-domain example integration test via `scripts/ci/run_integration_tests.sh`
  - Runs trimmed sub-domain integration test via `scripts/ci/run_integration_tests_subdomain.sh`
  - Runs trimmed plain openAMUNDSEN model sub-domain integration test via `scripts/ci/run_integration_tests_model_subdomain.sh`
  - Uploads integration artifacts on failure (log + example setup outputs)
- Job `Publish immutable commit and edge images`:
  - Runs only on push to `main`
  - Depends on all trusted tests
  - Uses the scoped repository `GITHUB_TOKEN`
  - Builds and pushes multi-architecture `sha-<full-commit>` and `edge` tags
  - Generates BuildKit provenance/SBOM metadata and a GitHub attestation
  - Never updates `latest`; only a stable release does that

Workflow file `.github/workflows/deploy-docs.yml` validates and builds the
documentation before deploying it to GitHub Pages. The separate
`.github/workflows/deploy-cloudflare.yml` workflow has no push trigger. It can
only update the protected Cloudflare Pages fallback through an intentional
manual dispatch.

Workflow file `.github/workflows/release.yml` runs for exact stable and RC tags.
It repeats the portable matrix and trusted Lenovo P8 rehearsal before OIDC
Trusted Publishing to TestPyPI or approval-gated PyPI, then publishes the
multi-architecture release image and GitHub prerelease/release.

## What You Must Do Manually

- Follow feature-branch workflow (do not develop directly on `main`).
- Monitor Actions results after push/merge.
- Keep the self-hosted runner online and healthy.
- Keep the `testpypi` and approval-gated `pypi` GitHub environments and their
  corresponding Trusted Publisher registrations valid.
- After a verified release or documentation cutover, manually dispatch the
  Cloudflare workflow from the same release tag or reviewed commit. Never use
  it as an automatic parallel deployment.
- Tune integration validation rules if expected output/warnings change.

## Current Test Stack

### Rough CI test steps (execution order)

1. Lint, package, documentation and architecture jobs start on GitHub-hosted runners.
2. The built wheel is reused by the 12-job portable matrix.
3. The Lenovo P8 job starts only after those gates are green.
4. A release-layout image is built from the tested wheel.
5. Unit tests are executed with `pytest` (`scripts/ci/run_unit_tests.sh`).
6. Integration run clones `examples/rofental` into a temporary workspace.
7. SCF and wet-snow per-step observation CSVs are prepared.
8. The full example is executed through `openamundsen-da run`.
9. Integration validators check logs, outputs, plots, scientific benchmark outputs, weights and subdomain manifests.
10. If integration fails, log and example outputs are uploaded as CI artifacts.
11. On push to `main`, the multi-architecture `edge` and immutable commit images publish after all gates.

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

### Documentation contract

`tests/unit/test_docs_contract.py` and `scripts/ci/validate_docs.py` keep the
published documentation aligned with the installed interface. The gate rejects
incomplete curated CLI guidance, invalid navigation parents, broken Jekyll links,
missing local assets, removed review/history pages, legacy installed commands
and any changed, unreferenced or extra PNG in the 13-image frozen Rofental
tutorial selection. CI then builds the production Jekyll site and confirms that
repository-only guidance and the superseded unreferenced science schematic are
not published.

### What is tested (rough checklist)

- Lint gate:
  - fatal syntax/logic lint classes (`E9`, `F63`, `F7`, `F82`)
  - stale-code checks for unused imports (`F401`), unused local variables (`F841`) and commented-out Python (`ERA001`)
- Unit logic:
  - config merge behavior
  - assimilation event parsing
  - land-cover path handling
  - setup skeleton basics
  - statistics helpers
  - assimilation requirement validation prechecks
- Integration project behavior:
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
- runs full project pipeline
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
  - compact data assimilation grid output (`results/grids/da_output_grids.nc`) with expected compressed integer storage encodings for DA-owned snow grids
  - generated meteo CSV precision and retained member grid storage dtypes
- minimal weight sanity (weights exist, numeric, sum to `1.0`)

### Integration regression test (trimmed sub-domain)

Runner script: `scripts/ci/run_integration_tests_subdomain.sh`

What it does:
- clones `examples/subdomains` into a temp directory
- writes a trimmed project config (`project_ci_2022_2023`) under the sub-domain setup
- runs the explicit data-assimilation subdomain stages
  (`openamundsen-da subdomains prepare`, `run` and `merge`) with:
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
- runs the explicit plain openAMUNDSEN model subdomain stages
  (`openamundsen-da subdomains model prepare`, `run` and `merge`) with:
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
- ruff fatal classes (`E9`, `F63`, `F7`, `F82`)
- stale-code classes enforced for releases (`F401`, `F841`, `ERA001`)

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
- GitHub environments `testpypi` and `pypi`
- A required reviewer on the stable `pypi` environment
- Matching TestPyPI and PyPI Trusted Publisher registrations for
  `.github/workflows/release.yml`

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

4. Publish-stage failures
- repository `GITHUB_TOKEN` lacks package or attestation permission
- GHCR login or push denied
- Trusted Publisher workflow/environment fields do not match
- stable `pypi` environment is waiting for reviewer approval

## Local Reproduction (optional)

From repository root:
- build and validate the distributions: `bash scripts/ci/build_distribution.sh`
- run unit test wrapper: `bash scripts/ci/run_unit_tests.sh`
- run single-domain integration wrapper: `bash scripts/ci/run_integration_tests.sh`
- run sub-domain integration wrapper: `bash scripts/ci/run_integration_tests_subdomain.sh`

Use same scripts as CI to avoid drift between local and server behavior.

## Rofental ES30 Scientific Fingerprint

The release-only comparison harness records scientific content without depending
on PNG chunks or NetCDF history metadata. Its selection contract is
`tests/baselines/rofental_es30_science_spec.json`.

Capture a completed canonical run:

```bash
python scripts/release/science_fingerprint.py capture \
  /path/to/rofental \
  tests/baselines/rofental_es30_science_spec.json \
  /path/to/rofental_es30_fingerprint.json \
  --metadata git_commit=<full-commit> \
  --metadata image_digest=<sha256-digest>
```

Compare a candidate capture to the committed baseline:

```bash
python scripts/release/science_fingerprint.py compare \
  tests/baselines/rofental_es30_science_fingerprint.json \
  /path/to/candidate_fingerprint.json
```

The capture compares semantic YAML content; CSV columns, row order and values;
decoded NetCDF dimensions, coordinates and arrays; and decoded RGBA image
pixels. Generated member configuration files are excluded because their
absolute runtime paths are provenance rather than scientific output.
Performance monitor outputs are deliberately excluded because they describe the
host run, not the scientific result. A changed fingerprint blocks release until
the difference is explained and explicitly approved.

### Exact manuscript setup

The manuscript figures and reported statistics come from the selected
`original8_p074_wsla100_fsca005` Rofental run. That run used a prepared
precipitation factor of `0.74`, the corrected 100 m snow redistribution grid and
an earlier input snapshot than the current shipped example. The five
byte-distinct inputs are preserved under
`tests/baselines/rofental_es30_manuscript_inputs/` with SHA-256 checksums.
Four differ only in serialization; the selected fSCA summary also differs in
values and therefore controls the DA7/DA8 simulation-stage observation record.
Materialize a clean reproduction setup with:

```bash
python scripts/release/materialize_manuscript_setup.py \
  /path/to/rofental_manuscript_es30
```

The command copies the current shipped setup, preserving its strict project
contract and current `maps.yml`/`plots.yml`, then overlays only the frozen
scientific inputs. Run the exact setup through the integration path with:

```bash
OA_DA_TEST_SETUP_SOURCE=/path/to/rofental_manuscript_es30 \
OA_DA_TEST_MAX_WORKERS=24 \
bash scripts/ci/run_integration_tests.sh
```

First validate the completed selected simulation before applying any later
analysis inputs:

```bash
python scripts/release/validate_manuscript_reference.py \
  /path/to/completed/rofental_manuscript_es30 \
  --stage simulation
```

The manuscript assets were rendered later with an updated fSCA summary. The
assimilation weights and model outputs remained those of the selected run, but
the benchmark tables, plots, maps and report were regenerated. Reproduce that
explicit second stage with:

```bash
python scripts/release/refresh_manuscript_outputs.py \
  /path/to/completed/rofental_manuscript_es30 \
  --manuscript-root /path/to/openAMUNDSEN-DA \
  --max-workers 24 \
  --apply
```

The refresh command refuses to mutate a run until the simulation-stage contract
passes. It regenerates the canonical public outputs under `results/`, then
replaces `results/paper/` with the developer-only manuscript profile. That
profile contains exactly the title-free setup weights overview and DA maps 6
and 8; normal project rendering does not create publication mirrors. Its final
publication-stage validation checks the selected-run
provenance, all case-study parameters, the eight event ESS and resampling
decisions, quoted benchmark values, generated figures, exact manuscript asset
hashes and the corresponding literals in `template.tex`. The selected
simulation and publication-analysis science contracts are stored separately in
`tests/baselines/rofental_es30_manuscript_simulation_fingerprint.json` and
`tests/baselines/rofental_es30_manuscript_science_fingerprint.json`. The
publication figure contract remains
`tests/baselines/rofental_es30_manuscript_assets.json`. It keeps the manuscript
files byte-exact and records one accepted fresh-render variant of Figure 04,
whose benchmark reductions differ only at machine precision and affect 1434
antialiased pixels in the CRPSS panel.

Tutorial images are governed separately by
`tests/baselines/rofental_es30_tutorial_assets.json`. Documentation validation
and tutorial staging use this dedicated contract, so accepting a runtime-specific
tutorial image cannot alter the manuscript figure records.

Preview manuscript asset staging before changing the separate manuscript
repository:

```bash
python scripts/release/stage_publication_assets.py \
  /path/to/completed/rofental_manuscript_es30 \
  --target manuscript \
  --destination /path/to/openAMUNDSEN-DA/assets
```

Preview the corresponding tutorial image selection with:

```bash
python scripts/release/stage_publication_assets.py \
  /path/to/completed/rofental_manuscript_es30 \
  --target tutorial \
  --destination docs/assets/images/tutorial/rofental_2022_2023_es30
```

The staging command validates source hashes and decoded RGBA pixels, lists
every selected copy and leaves all unlisted destination files untouched.
`PRESERVE` means that a source is an explicitly recorded cross-platform render
variant, or a machine-specific performance plot, while the canonical
destination remains byte-exact. Such a source is never copied over the
canonical asset. The command does not edit LaTeX. Use `--apply` only after
reviewing the preview and, for manuscript assets, receiving explicit approval
for the proposed replacements.

The validator records the intentional metadata distinction between the
physical Proviantdepot altitude reported by the manuscript and upstream point
example (2737 m) and the 2659 m altitude stored in the regional forcing table
used by the selected run. The selected-run contract also records, without
blocking validation, the author-approved manuscript wording that describes a
40% forcing reduction while the frozen input snapshot uses a factor of 0.74.
