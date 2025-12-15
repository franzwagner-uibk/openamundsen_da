# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/franzwagner-uibk/openamundsen_da/compare/v0.1.0...HEAD
