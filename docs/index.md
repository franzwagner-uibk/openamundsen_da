---
layout: default
title: Home
nav_order: 1
description: "openamundsen_da - Data Assimilation Framework for openAMUNDSEN"
permalink: /
---

# openamundsen_da

{: .fs-9 }

Data Assimilation Framework for openAMUNDSEN
{: .fs-6 .fw-300 }

[Get started](#getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/franzwagner-uibk/openamundsen_da){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## DISCLAIMER

**This documentation may contain errors and incorrect statements.**

- this documentation is **work in progress** and only **internaly** available to the _Fram3S_ project team
- there is no scientific publication for the **openamundsen_da** framwework yet
- **do not use information from this documentation for critical work**
- **this version of the documentation is lacking scientific references**
- this documentation is **not complete**. More da algorithms and observation interfaces will be added later

## Overview

**openamundsen_da** is a lightweight toolkit for building and running [openAMUNDSEN](http://doc.openamundsen.org/) ensembles with particle filter data assimilation. It enables seasonal snow cover prediction by assimilating satellite snow cover and wet snow observations.

openAMUNDSEN is an open-source, fully distributed snow-hydrological model designed for mountain regions (Strasser et al., 2024). This framework extends openAMUNDSEN with ensemble data assimilation capabilities for improved snow cover forecasting.

### Key Features

- **Prior forcing builder** for meteorological perturbations (temperature ±σ<sub>T</sub>, precipitation ×σ<sub>P</sub>)
- **Parallel ensemble launcher** with warm-start capability
- **MODIS MOD10A1 preprocessing** (HDF → GeoTIFF, QA masking, reprojection)
- **Sentinel-2 FSC extraction** via Snowflake (**TODO** _Source_) product
- **Sentinel-1 wet snow classification**
- **H(x) forward operators** for model-to-observation space mapping
- **Particle filter implementation** (systematic resampling, ESS monitoring)
- **Rejuvenation and state propagation** between assimilation cycles
- **Comprehensive visualization suite** for forcing, results, and diagnostics
- **Performance monitoring** (CPU, RAM, disk usage, ETA estimation)

### Requirements

- Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- GDAL/PROJ (via Conda)
- Python ≥3.10

---

## Getting Started

{: .note }

> This framework is designed to work with Docker for easy deployment and reproducibility.

### Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/franzwagner-uibk/openamundsen_da.git
   cd openamundsen_da
   ```

2. **Build the Docker image**

   ```bash
   docker build -t oa-da .
   ```

3. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your paths
   ```

4. **Set up your project**

   - Copy the template from `templates/project/` to your project directory
   - Configure `project.yml`, `season.yml`, and prepare your data

5. **Run a seasonal experiment**
   ```bash
   docker compose run --rm oa \
     python -m openamundsen_da.pipeline.season \
     --project-dir /data \
     --season-dir /data/propagation/season_2019-2020
   ```

See the [Installation Guide]({{ site.baseurl }}{% link installation.md %}) for detailed setup instructions.

---

## Documentation Structure

### Core Documentation

- [Installation]({{ site.baseurl }}{% link installation.md %}) - Setup and configuration
- [Project Structure]({{ site.baseurl }}{% link project-structure.md %}) - Understanding the directory layout
- [Workflow]({{ site.baseurl }}{% link workflow.md %}) - Data assimilation workflow overview

### User Guides

- [Configuration]({{ site.baseurl }}{% link guides/configuration.md %}) - YAML configuration reference
- [Command-Line Interface]({{ site.baseurl }}{% link guides/cli.md %}) - CLI commands reference
- [Observation Processing]({{ site.baseurl }}{% link guides/observations.md %}) - Working with satellite data
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments.md %}) - Step-by-step experiment setup

### Technical Reference

- [Package Structure]({{ site.baseurl }}{% link reference/package-structure.md %}) - Module organization
- [API Reference]({{ site.baseurl }}{% link reference/api.md %}) - Python API documentation
- [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) - Particle filter implementation

### Advanced Topics

- [Performance Tuning]({{ site.baseurl }}{% link advanced/performance.md %}) - Optimization strategies
- [Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) - Common issues and solutions

---

## License

**TODO** _License information_

---

## Citation

If you use this software in your research, please cite both this framework and openAMUNDSEN:

```bibtex
**TODO** citation guide
```

---

## References

**openAMUNDSEN Documentation**: [doc.openamundsen.org](http://doc.openamundsen.org/)

**openAMUNDSEN GitHub**: [github.com/openamundsen/openamundsen](https://github.com/openamundsen/openamundsen)

Strasser, U., Warscher, M., Rottler, E., and Hanzer, F.: openAMUNDSEN v1.0: an open-source snow-hydrological model for mountain regions, Geosci. Model Dev., 17, 6775–6797, [https://doi.org/10.5194/gmd-17-6775-2024](https://doi.org/10.5194/gmd-17-6775-2024), 2024.
