# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add opt-in project-map markers that distinguish forcing, snow-observation,
  co-located and holdout stations, with holdouts using a smaller black `x`
  rendered above other station markers, plus optional subdomain ID labels.
- Add fixed disk-admission limits, shared-filesystem subdomain reservations for
  accumulated forcing/grid/point growth, rolling checkpoints and atomic merge,
  resumable low-disk status and a versioned cleanup ledger for restart-safe
  compact retention.
- Add a retained `results/storage/storage_reservation.json` audit ledger and a
  coordinator-owned incremental admission path. Full estimation now occurs at
  preflight and lifecycle transitions, while ordinary step boundaries consume
  producer byte summaries and perform one filesystem usage check.
- Add compressed all-member point and consumed-forcing NetCDFs plus retained
  satellite-event map support for compact projects, with mean-collapsed step
  overlaps, retained-value validation and leaf summaries kept for rerendering.

### Changed
- Interpolate exactly two symmetric station HS or SWE observations at their
  model-time midpoint when both lie inside the half-timestep window and are no
  more than 24 hours apart, while retaining strict matching for other ties and
  model outputs.
- Allow subdomain station selection from coordinates in
  `stations_da_metadata.csv`, require same-ID observations and model points for
  every active DA or benchmark station before propagation.
- Keep particle rejuvenation within the project runner's worker limit.
- Generate perturbed forcing only for each consuming step's exact time window.
- Make compact retention remove validated member forcing, point, grid and
  restart artifacts while full retention preserves them for reanalysis.
- Resume failed subdomain leaves non-destructively unless overwrite is
  requested explicitly.
- Require readable successor checkpoints before compact restart cleanup, bind
  interrupted cleanup retries to exact source, consumer and producer-manifest
  generations and durably validate compact grids, time series and DA map
  support before raw-grid deletion.
- Make storage admission account for each forcing file independently, layered
  and default point columns, explicit map support, full atomic overwrite
  temporaries and completed parent-finalization stages.
- Finalize and consumer-validate each successful compact subdomain leaf before
  admitting the next bounded leaf wave, then retain only its compact analysis
  and parent merge/render support. After compact forcing and stable render
  completion evidence validate, remove replaceable step forcing PNGs as a
  ledger-backed derived-artifact class; project forcing plots rerender from
  compact NetCDF.
  The audited prepared-Euregio admission envelope now fits a clean 3.6 TB
  filesystem, but an end-to-end production run remains the acceptance gate.
- Version cleanup generations explicitly so a validated overwrite supersedes
  historical consumer hashes without deleting their audit records. Validate
  raw SCF/wet-snow render support by rebuilding it under full retention, reserve
  full checkpoint replacement coexistence during overwrite and include a
  calibrated retained diagnostics/log/metadata allowance in active and queued
  budgets.
- Bind planned cleanup generations to exact source, consumer and producer
  inventories. Refuse overwrite/resume identity mixing, revalidate every batch
  before generation completion and require the public single-domain run API to
  accept the completed retention ledger before reporting success.
- Require explicitly configured compact grid sources in openAMUNDSEN output
  before propagation and validate every requested variable and metric in
  member, project and merged subdomain NetCDFs.
- Sample exact project-directory size every 150 seconds by default while
  leaving CPU, RAM and plot-refresh sampling unchanged.
- Match station observations consistently in validation, assimilation and
  analysis benchmarking to the unique nearest value within half the model
  timestep, with timezone-aware comparison and exact model-output timestamps.
- Treat every project's `assimilation_events` as final: reject the removed
  in-core subdomain event filter and derive cross-leaf rendering support from
  the accepted leaf project YAMLs.

### Fixed
- Parse mixed ISO date-only and date-time rows in point-output CSVs strictly so
  compact point export does not fail after otherwise successful propagation.
- Keep coordinated compact-storage admission conservative when rolling or
  final cleanup in one subdomain removes package-owned checkpoints, forcing,
  point, grid or forcing-plot artifacts while another subdomain recomputes the
  shared-filesystem reserve.
- Resolve posterior checkpoint pointers to their actual prior-member producer
  manifests before compact rolling cleanup, while continuing to reject missing,
  malformed or external checkpoint provenance before deletion.
- Keep point-result titles consistent with the observations actually plotted
  and wrap dense station uncertainty keys inside the panel while preserving the
  shared symmetric residual range per observable.
- Keep the two right-hand performance axes readable from small projects through
  multi-terabyte runs and use one-decimal storage/RAM summaries and `°C` labels.
- Allow final subdomain products to mask events absent from a leaf project while
  still rejecting configured-but-missing support and top-level events unsupported
  by every leaf.

## [0.9.4] - 2026-07-29

### Changed
- Use recursive importance weights in the particle filter and preserve the
  weighted analysis distribution until resampling actually occurs.
- Use keyed, event-specific process-noise perturbations with strict seed,
  provenance and resume contracts.
- Match satellite acquisitions to the nearest model timestep and select the
  interpolated uppermost wet-snow transition.
- Publish the approved Rofental configuration, tutorial assets and validation
  baselines for the corrected method.

### Fixed
- Prevent skipped resampling from discarding prior particle weights.
- Prevent rejuvenation from repeating identical perturbations at every event.
- Correct wet-snow-line support coverage and exclude presentation diagnostics
  from scientific resume ancestry.

## [0.9.3] - 2026-07-24

### Changed
- Credit Franz Wagner, Erwin Rottler and Ulrich Strasser as software creators
  in manuscript order, including their ORCIDs and shared affiliation.
- Move the restricted North Tyrol subdomain example and runbook to a pinned
  private maintainer fixture while retaining the public subdomain API, CLI,
  documentation and mandatory trusted integration coverage.
- Rename the 153 shipped Rofental snow-cover observation files to the neutral
  `s2_fsc_rofental_*` pattern without changing their raster contents.
- Improve the tutorial introduction, Docker setup, host-path guidance,
  `region_id` explanation and station-siting terminology, and remove the
  ambiguous Rofental SWE reference interpretation.
- Update stable package and container references to `0.9.3`.

### Fixed
- Round only `unc_min` and `unc_max` in generated `scf_summary.csv` files to
  three decimal places while preserving `unc_mean` and filtering decisions.

## [0.9.2] - 2026-07-23

### Changed
- Add machine-readable software citation metadata and validate it in CI and
  release workflows before publication.
- Replace references to an unpublished manuscript with a neutral
  manuscript-in-preparation note in the project overview and documentation.
- Update stable package and container references to `0.9.2`.

## [0.9.1] - 2026-07-23

### Changed
- Use `openamundsen/openamundsen-da` as the canonical repository and public
  `ghcr.io/openamundsen/openamundsen-da` as the canonical container package.
- Deploy current documentation to GitHub Pages at
  `https://doc-da.openamundsen.org/`; the existing Cloudflare Pages deployment
  remains a manual-only protected fallback.
- Group the shared user documentation under a `Documentation` section and align
  its visible page names with openAMUNDSEN without changing public URLs.
- Add a locked native Python wheel integration on the Lenovo P8 that processes
  the Rofental example through the public Python API outside Docker.
- Gate releases on both the native Python integration and the existing trusted
  Docker integrations before publishing packages or containers.

### Fixed
- Support contour path effects with Matplotlib 3.10 and later while retaining
  compatibility with older Matplotlib contour collections.
- Exclude native CI dependency constraints from source distributions.

## [0.9.0] - 2026-07-19

### Changed
- RC8 treats the exact root `CHANGELOG.md` as documentation-only in CI while
  retaining full CI for every uncertain or mixed change.
- Derive the package and runtime version from Git tags with `setuptools-scm`.
- Declare the supported Python range as 3.11 through 3.14.
- Expanded the CI lint gate to include unused imports, unused local variables and commented-out Python code.
- Updated package metadata with README, license expression, authors, project URLs and release classifiers.

### Removed
- Removed pre-v1 transitional aliases `cli_setup_daily`, `cli_model_setup`, `_read_resampling_from_setup` and the `openamundsen_da.methods.pf.plot_weights` re-export.
- Removed the unused `oa-da-resample --project-dir` compatibility option; resampling configuration is inferred from `--step-dir` through the project YAML.
- Removed migration-era cleanup of historical benchmark, map and project plot output locations.
- Removed the unregistered Copernicus HRWSI downloader and its unused `boto3` dependency after caller tracing.

### Fixed
- RC8 reduces performance-plot time-label density while preserving every tick,
  grid position, telemetry sample and scientific output.
- Removed UTF-8 BOMs and CRLF line endings from tracked release-facing text/config files.
- Excluded generated `docs/_site/` output from Docker build context.
- Corrected stale docs around Cloudflare Pages deployment, setup/project config ownership and resampling configuration ownership.

### Pre-release 2025

Initial pre-release of openamundsen-da - Data Assimilation Framework for openAMUNDSEN.

#### Features
- Ensemble generation with meteorological forcing perturbations
- Particle filter data assimilation for snow cover observations
- Support for MODIS MOD10A1 snow cover fraction
- Support for Sentinel-2 fractional snow cover (via Snowflake/Theia)
- Support for Sentinel-1 wet snow detection
- Forward operators (H(x)) for snow depth and SWE
- Systematic resampling with ESS monitoring
- Rejuvenation between assimilation cycles
- Docker-based deployment
- Comprehensive CLI tools for observation processing and ensemble management
- Performance monitoring and visualization suite

#### Documentation
- Installation and configuration guides
- Command-line interface reference
- Workflow and experiment tutorials
- API and package structure documentation
- Troubleshooting and performance tuning guides

---

[Unreleased]: https://github.com/openamundsen/openamundsen-da/compare/v0.9.4...HEAD
[0.9.4]: https://github.com/openamundsen/openamundsen-da/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/openamundsen/openamundsen-da/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/openamundsen/openamundsen-da/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/openamundsen/openamundsen-da/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/openamundsen/openamundsen-da/compare/v0.9.0rc8...v0.9.0
