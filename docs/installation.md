---
layout: default
title: Installation
nav_order: 2
---

# Installation
{: .no_toc }

Complete guide to installing and setting up openAMUNDSEN-DA.
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

---

## Quickstart (no clone): Rofental example via Docker

Everything needed ships inside the image; the command copies the bundled example to your host, then runs the season.

1. Pull the image:
   ```bash
   docker pull ghcr.io/franzwagner-uibk/openamundsen_da:latest
   ```

2. Run and persist outputs:
   ```bash
   mkdir -p oa_run
   docker run --rm -v "$(pwd)/oa_run:/data" \
     ghcr.io/franzwagner-uibk/openamundsen_da:latest \
     bash -lc "cp -a /workspace/examples/rofental /data/rofental && \
               python -m openamundsen_da.pipeline.season \
                 --project-dir /data/rofental \
                 --season-dir /data/rofental/propagation/season_2022_2023 \
                 --max-workers 4 \
                 --log-level INFO"
   ```

What this does:
- `cp -a ...` copies the bundled Rofental project (configs + sample data) from the image to your mounted host path `/data/rofental`.
- The pipeline runs the 2022/2023 season with SCF/WETSNOW assimilation events from `season.yml`.
- Outputs are written under `oa_run/rofental/propagation/season_2022_2023` on your host.

3. **Python 3.10+** (if running without Docker)
   - Required packages listed in `pyproject.toml`
   - GDAL and PROJ must be installed (prefer Conda)

---

## Developer install (clone + compose)

Use this when you want to modify the code or run Compose with mounted source.

1. Clone the repo:
   ```bash
   git clone https://github.com/franzwagner-uibk/openamundsen_da.git
   cd openamundsen_da
   ```

2. (Optional) Build a local image instead of pulling:
   ```bash
   docker build -t ghcr.io/franzwagner-uibk/openamundsen_da:local .
   ```

3. Compose with defaults (no `.env` needed): `REPO` defaults to `.` and `PROJ` to `./examples/rofental`. Override inline if needed:
   ```bash
   REPO=/path/to/repo PROJ=/path/to/project \
   docker compose run --rm oa python -c "import openamundsen_da; print('Success!')"
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
