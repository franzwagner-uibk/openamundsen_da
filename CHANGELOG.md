# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-06-15

### Changed
- Marked the package metadata and runtime `__version__` as `1.0.0`.
- Expanded the CI lint gate to include unused imports, unused local variables and commented-out Python code.
- Updated package metadata with README, license expression, authors, project URLs and release classifiers.

### Removed
- Removed pre-v1 transitional aliases `cli_setup_daily`, `cli_model_setup`, `_read_resampling_from_setup` and the `openamundsen_da.methods.pf.plot_weights` re-export.
- Removed the unused `oa-da-resample --project-dir` compatibility option; resampling configuration is inferred from `--step-dir` through the project YAML.
- Removed migration-era cleanup of historical benchmark, map and project plot output locations.

### Fixed
- Removed UTF-8 BOMs and CRLF line endings from tracked release-facing text/config files.
- Excluded generated `docs/_site/` output from Docker build context.
- Corrected stale docs around Cloudflare Pages deployment, setup/project config ownership and resampling configuration ownership.

### Pre-release 2025

Initial pre-release of openamundsen_da - Data Assimilation Framework for openAMUNDSEN.

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

[Unreleased]: https://github.com/franzwagner-uibk/openamundsen_da/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/franzwagner-uibk/openamundsen_da/releases/tag/v1.0.0
