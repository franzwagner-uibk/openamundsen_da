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
- **CPU**: Multi-core processor (parallelization scales with core count)

### Software Dependencies

1. **Docker**
   - Windows/macOS: [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Linux: Docker Engine + Docker Compose

2. **Git** (for cloning the repository)

---

## Quickstart (no clone): Rofental example via Docker

Everything needed ships inside the image; the commands below copy the bundled Rofental example (with a ready-made season skeleton at `examples/rofental/propagation/season_2022_2023`) to your host, then run the season. You can still generate a new skeleton yourself with `python -m openamundsen_da.pipeline.season_skeleton ...` if you want to start from scratch, but that isn’t required for the quickstart.

1. Pull the image (optional—`docker run` will pull if missing):

   ```bash
   docker pull ghcr.io/franzwagner-uibk/openamundsen_da
   ```

2. Prepare a host folder for outputs:
   ```bash
   mkdir -p openamundsen-da
   ```

3. Run and persist outputs:
   ```bash
   docker run --rm -v "$(pwd)/openamundsen-da:/data" \
     ghcr.io/franzwagner-uibk/openamundsen_da \
     bash -lc "cp -a /workspace/examples/rofental /data/rofental && \
               python -m openamundsen_da.pipeline.season \
                 --project-dir /data/rofental \
                 --season-dir /data/rofental/propagation/season_2022_2023 \
                 --max-workers 8 \
                 --perf-monitor \
                 --overwrite \
                 --log-level INFO"
   ```

What this does:

- Copies the bundled Rofental project (configs, sample data, season skeleton under `propagation/season_2022_2023`) to your host at `/data/rofental`.
- Runs the full 2022/2023 season (propagation + SCF/WETSNOW assimilation) using that skeleton; `--overwrite` clears any previous run outputs in the target directories.
- Outputs and plots land under `openamundsen-da/rofental/propagation/season_2022_2023` on your host. Logs stream to the terminal.

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
resolution: 100 # spatial resolution (m)
timestep: "3H" # temporal resolution
crs: "epsg:25832" # CRS of the input grids
timezone: 1 # UTC offset in hours

data_assimilation:
  prior_forcing:
    ensemble_size: 20 # number of ensemble members
    random_seed: 42
    sigma_t: 0.5 # temperature perturbation stddev (deg C)
    mu_p: 0.0 # log-space mean for precip factor
    sigma_p: 0.5 # log-space stddev for precip factor

  h_of_x:
    method: depth_threshold # or "logistic"
    variable: hs # or "swe"
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
