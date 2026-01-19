---
layout: default
title: Installation
nav_order: 2
---

# Installation
{: .no_toc }

Complete guide to installing and setting up openamundsen_da.
{: .fs-6 .fw-300 }

<details markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **Memory**: 16 GB RAM minimum (32 GB recommended for large ensembles)
- **Storage**: 50 GB free space minimum (depends on domain size and ensemble size)
- **CPU**: Multi-core processor (parallelization scales with core count)

### Software Dependencies

1. **Docker**
   - Windows/macOS: [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Linux: Docker Engine + Docker Compose

2. **Git** (for cloning the repository)

3. **Python 3.10+** (if running without Docker)
   - Required packages listed in `pyproject.toml`
   - GDAL and PROJ must be installed (prefer Conda)

---

## Docker Installation (Recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/franzwagner-uibk/openamundsen_da.git
cd openamundsen_da
```

### 2. Build the Docker Image

```bash
docker build -t oa-da .
```

This creates a containerized environment with all dependencies pre-installed.

### 3. Configure Environment Variables

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Edit `.env` with your local paths:

```bash
# .env file
REPO=/path/to/openamundsen_da        # Repository root on your machine
PROJ=/path/to/your/project           # Your project data directory
# Optional (Linux/macOS): map container user to host user to avoid root-owned files
# UID=1000
# GID=1000
CPUS=8                               # Number of CPUs for Docker
MEMORY=16G                           # Memory limit for Docker
MAX_WORKERS=7                        # Default max parallel workers
# OA_BASE_SEED=1337                  # Optional global base seed for DA stages
```

{: .note }
> The `.env` file is machine-specific and should not be committed to Git. It's already in `.gitignore`.
> Compose pins OMP/BLAS threads to 1 inside the container; you don't need an `OMP_THREADS` entry.

### 4. Verify Installation

Test the installation:

```bash
docker compose run --rm oa python --version
docker compose run --rm oa python -c "import openamundsen_da; print('Success!')"
```

---

## Native Installation (Without Docker)

{: .warning }
> Native installation requires manual dependency management. Docker installation is recommended for most users.

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev python3-dev build-essential
```

**macOS (with Homebrew):**
```bash
brew install gdal proj
```

**Windows:**
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- GDAL/PROJ will be installed via Conda in the next step

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate openamundsen_da
```

### 3. Install the Package

```bash
pip install -e .
```

### 4. Verify Installation

```bash
python -c "import openamundsen_da; print('Success!')"
oa-da-season --help
```

---

## Setting Up a Project

### 1. Copy Project Template

The repository includes a project template with the required directory structure:

```bash
cp -r templates/project /path/to/your/project
```

### 2. Project Directory Structure

Your project should have this structure:

```
project/
├── env/
│   └── roi.gpkg              # Single-feature ROI polygon (required)
├── grids/
│   └── lc_<domain>_<resolution>.asc  # Land-cover classes for DA masking
├── meteo/
│   ├── stations.csv          # Station metadata
│   ├── station_001.csv       # Meteorological forcing data
│   └── ...
├── obs/
│   └── season_YYYY-YYYY/
│       ├── scf_summary.csv   # SCF observations
│       └── ...
├── propagation/
│   └── season_YYYY-YYYY/     # Created by the framework
└── project.yml               # Main configuration (required)
```

### 3. Configure project.yml

Edit `project/project.yml` to configure your experiment:

```yaml
domain: "your_domain"
resolution: 100            # spatial resolution (m)
timestep: "3H"             # temporal resolution
crs: "epsg:25832"          # CRS of the input grids
timezone: 1                # UTC offset in hours

data_assimilation:
  prior_forcing:
    ensemble_size: 20       # number of ensemble members
    random_seed: 42
    sigma_t: 0.5            # temperature perturbation stddev (deg C)
    mu_p: 0.0               # log-space mean for precip factor
    sigma_p: 0.5            # log-space stddev for precip factor

  h_of_x:
    method: depth_threshold # or "logistic"
    variable: hs            # or "swe"
    params:
      h0: 0.01
      k: 80

  resampling:
    algorithm: systematic
    ess_threshold_ratio: 0.5

  rejuvenation:
    sigma_t: 0.2
    sigma_p: 0.2

  landcover_mask:
    # Classes: 1 rock, 2 ice, 3 water, 4 grassland, 5 shrubland, 6 farmland,
    # 7 transitional, 8 deciduous 30-60, 9 deciduous 60-100, 10 mixed,
    # 11 coniferous 30-60, 12 coniferous 60-100, 13 built-up.
    enabled: true
    classes_to_exclude: [2, 8, 9, 10, 11, 12, 13]
```

See the [Configuration Guide]({{ site.baseurl }}{% link guides/configuration.md %}) for all options.

---

## Next Steps

- [Project Structure]({{ site.baseurl }}{% link project-structure.md %}) - Understand the directory layout
- [Workflow Overview]({{ site.baseurl }}{% link workflow.md %}) - Learn the DA workflow
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments.md %}) - Start your first experiment

---

## Troubleshooting

### Docker Issues

**Problem**: "Cannot connect to Docker daemon"
```bash
# Start Docker Desktop (Windows/macOS)
# Or start Docker service (Linux)
sudo systemctl start docker
```

**Problem**: "Permission denied" on Linux
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### GDAL Issues

**Problem**: "GDAL not found" or import errors

```bash
# Check GDAL installation
gdalinfo --version

# Set environment variables (Linux/macOS)
export GDAL_DATA=$(gdal-config --datadir)
export PROJ_LIB=/path/to/proj/share/proj

# Or add to project.yml
environment:
  GDAL_DATA: /usr/share/gdal
  PROJ_LIB: /usr/share/proj
```

See [Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) for more issues and solutions.
