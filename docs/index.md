---
layout: default
title: Home
nav_order: 1
description: "openAMUNDSEN-DA - Data Assimilation Framework for openAMUNDSEN"
permalink: /
---

# openAMUNDSEN-DA

{: .fs-9 }

Data Assimilation Framework for openAMUNDSEN
{: .fs-6 .fw-300 }

[Get started](https://openamundsen-da.pages.dev/installation#prerequisites){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/franzwagner-uibk/openamundsen_da){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## DISCLAIMER

**This documentation may contain errors and incorrect statements.**

- this documentation is **work in progress** and only **internaly** available to the _Fram3S_ project team
- there is no scientific publication for the **openAMUNDSEN-DA** framwework yet
- **do not use information from this documentation for critical work**
- **this version of the documentation is lacking scientific references**
- this documentation is **not complete**. More da algorithms and observation interfaces will be added later

## Overview

**openAMUNDSEN-DA** is a lightweight toolkit for building and running [openAMUNDSEN](http://doc.openamundsen.org/) ensembles with particle filter data assimilation. It enables setup-based snow cover prediction by assimilating satellite snow cover and wet snow observations.

openAMUNDSEN is an open-source, fully distributed snow-hydrological model designed for mountain regions (Strasser et al., 2024). This framework extends openAMUNDSEN with ensemble data assimilation capabilities for improved snow cover forecasting.

### Key Features

- **Prior forcing builder** for meteorological perturbations (temperature Â±Ïƒ<sub>T</sub>, precipitation Ã—Ïƒ<sub>P</sub>)
- **Parallel ensemble launcher** with warm-start capability
  - **Snow-cover preprocessing** (GeoTIFF/NetCDF; includes MODIS after HDF conversion with project.yml class mapping)
- **Sentinel-2 FSC extraction** via Snowflake product (Barella et al., 2022)
- **Sentinel-1 wet snow classification** (Nagler et al., 2016)
- **H(x) forward operators** for model-to-observation space mapping
- **Particle filter implementation** (systematic resampling, ESS monitoring)
- **Rejuvenation and state propagation** between assimilation cycles
- **Comprehensive visualization suite** for forcing, results, and diagnostics
- **Performance monitoring** (CPU, RAM, disk usage, ETA estimation)

### Requirements

- Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- GDAL/PROJ (via Conda)
- Python â‰¥3.10

---

## Getting Started

{: .note }

> This framework is designed to work with Docker for easy deployment and reproducibility.

### Quick Start (Rofental example)

Run the bundled Rofental setup shipped with the docker image:

[Open Quick Start](https://openamundsen-da.pages.dev/installation#prerequisites){: .btn .btn-primary .fs-4 }

## Documentation Structure

### Core Documentation

- [Installation]({{ site.baseurl }}{% link installation.md %}) - Setup and configuration
- [Project Structure]({{ site.baseurl }}{% link project-structure.md %}) - Understanding the directory layout
- [Workflow]({{ site.baseurl }}{% link workflow.md %}) - Data assimilation workflow overview

### User Guides

- [Configuration]({{ site.baseurl }}{% link guides/configuration.md %}) - YAML configuration reference
- [Command-Line Interface]({{ site.baseurl }}{% link guides/cli.md %}) - CLI commands reference
- [Observation Processing]({{ site.baseurl }}{% link guides/observations.md %}) - Working with satellite data
- [Running Experiments]({{ '/guides/experiments/' | relative_url }}) - Set up your own project and run custom setups

### Technical Reference

- [Package Structure]({{ site.baseurl }}{% link reference/package-structure.md %}) - Module organization
- [API Reference]({{ site.baseurl }}{% link reference/api.md %}) - Python API documentation
- [Data Assimilation Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) - Particle filter implementation

## License

**TODO** _License information_

---

## Citation

If you use this software in your research, please cite openAMUNDSEN and (where relevant) the satellite observation products used in your experiment (see References).

---

## References

- Strasser, U., Warscher, M., Rottler, E., and Hanzer, F. (2024). openAMUNDSEN v1.0: an open-source snow-hydrological model for mountain regions. Geoscientific Model Development, 17, 6775-6797. https://doi.org/10.5194/gmd-17-6775-2024.
- Barella, R., Marin, C., Gianinetto, M., and Notarnicola, C. (2022). A novel approach to high resolution snow cover fraction retrieval in mountainous regions. IGARSS 2022 - IEEE International Geoscience and Remote Sensing Symposium, 3856-3859. https://doi.org/10.1109/IGARSS46834.2022.9884177.
- Nagler, T., Rott, H., Ripper, E., Bippus, G., and Hetzenecker, M. (2016). Advancements for snowmelt monitoring by means of Sentinel-1 SAR. Remote Sensing, 8(4), 348. https://doi.org/10.3390/rs8040348.


