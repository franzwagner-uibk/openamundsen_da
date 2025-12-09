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

## Overview

**openamundsen_da** is a lightweight toolkit for building and running openAMUNDSEN ensembles with particle filter data assimilation. It enables seasonal snow cover prediction by assimilating satellite snow cover fraction (SCF) observations from MODIS and Sentinel-2, and wet snow detection from Sentinel-1.

### Key Features

- **Prior forcing builder** with meteorological perturbations (temperature ±σ<sub>T</sub>, precipitation ×σ<sub>P</sub>)
- **Parallel ensemble launcher** with warm-start capability
- **MODIS MOD10A1 preprocessing** (HDF → GeoTIFF, QA masking, reprojection)
- **Sentinel-2 FSC extraction** via Snowflake product
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

This project is distributed under the MIT License. See `LICENSE` file for more information.

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{openamundsen_da,
  author = {Your Name},
  title = {openamundsen_da: Data Assimilation Framework for openAMUNDSEN},
  year = {2024},
  url = {https://github.com/franzwagner-uibk/openamundsen_da}
}
```
