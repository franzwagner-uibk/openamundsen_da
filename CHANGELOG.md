# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.3] - 2026-07-24

### Changed
- Credit Franz Wagner, Erwin Rottler and Ulrich Strasser as software creators
  in manuscript order, including their ORCIDs and shared affiliation.
- Update stable package and container references to `0.9.3`.

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

[Unreleased]: https://github.com/openamundsen/openamundsen-da/compare/v0.9.3...HEAD
[0.9.3]: https://github.com/openamundsen/openamundsen-da/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/openamundsen/openamundsen-da/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/openamundsen/openamundsen-da/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/openamundsen/openamundsen-da/compare/v0.9.0rc8...v0.9.0
