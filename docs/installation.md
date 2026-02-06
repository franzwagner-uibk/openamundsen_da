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
   - **Windows**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop); enable WSL2 backend. Reboot if prompted.
   - **macOS**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop).
   - **Linux**: Install Docker Engine (and Compose, usually bundled). Quick steps (Ubuntu/Debian):
     ```bash
     sudo apt-get update
     sudo apt-get install -y docker.io docker-compose-plugin
     sudo usermod -aG docker $USER   # re-login to drop sudo
     sudo systemctl enable --now docker
     docker run hello-world          # verify
     ```

2. **Git** (optional, only for developer install)

---

## Quickstart: Rofental example via Docker

Everything needed ships inside the image; the commands below copy the bundled Rofental example to your host, generate the season skeleton, and then run the season.

1. **Pull and alias the image (once):**

   This downloads the container and assigns a short local name so the later commands are easier to read.

   ```bash
   docker pull ghcr.io/franzwagner-uibk/openamundsen_da
   docker tag ghcr.io/franzwagner-uibk/openamundsen_da openamundsen_da
   ```

2. **Prepare a host folder and set variables:**

   This creates your local workspace and defines reusable paths for the image, project, and season.

   ```bash
   mkdir -p openamundsen-da && cd openamundsen-da
   IMAGE=openamundsen_da
   PROJECT=/data/rofental
   SEASON=/data/rofental/propagation/season_2022_2023
   SCF_SUM=/data/rofental/obs/season_2022_2023/scf_summary.csv
   WET_SUM=/data/rofental/obs/season_2022_2023/wet_snow_summary.csv
   ```

3. **Copy the bundled Rofental example to your host:**

   This puts a ready-to-run project template into your local workspace.

   ```bash
   docker run --rm -v "$(pwd):/data" \
     "$IMAGE" \
     cp -a /workspace/examples/rofental /data/rofental
   ```

4. **Generate the season skeleton (step\_\* directories/YAMLs from `season.yml`):**

   This creates the step structure for the season based on the assimilation dates.

   ```bash
   docker run --rm -v "$(pwd):/data" \
     "$IMAGE" \
     python -m openamundsen_da.pipeline.season_skeleton \
       --project-dir "$PROJECT" \
       --season-dir "$SEASON"
   ```

5. **Distribute observation summaries to steps (SCF + wet snow):**

   This creates per-step `obs_*.csv` files from the season summaries using the assimilation dates in `season.yml`.

   ```bash
   docker run --rm -v "$(pwd):/data" \
     "$IMAGE" \
     oa-da-scf \
       --season-dir "$SEASON" \
       --summary-csv "$SCF_SUM" \
       --overwrite

   docker run --rm -v "$(pwd):/data" \
     "$IMAGE" \
     oa-da-wetsnow-season \
       --season-dir "$SEASON" \
       --summary-csv "$WET_SUM" \
       --overwrite
   ```

6. **Run the season (propagation + SCF/WETSNOW assimilation):**

   This runs the full 2022/2023 season and writes outputs to your local `openamundsen-da` folder.

   ```bash
   docker run --rm -v "$(pwd):/data" \
     "$IMAGE" \
     python -m openamundsen_da.pipeline.season \
       --project-dir "$PROJECT" \
       --season-dir "$SEASON" \
       --max-workers 8 \
       --perf-monitor \
       --overwrite
   ```

Files to inspect after a run (all under `openamundsen-da/rofental/propagation/season_2022_2023`):

- Logs: container stdout; if present, `season.log`.
- Perf monitor: `plots/perf/season_perf.png` and `plots/perf/season_perf_metrics.csv`.
- Assimilation diagnostics: `plots/assim/` (weights/ESS), `assim/weights.csv`.
- Results: `plots/results/` (ensemble envelopes), `ensembles/posterior/` per step.

What the bundled Rofental example contains (copied to `/data/rofental`):

- `project.yml` - DA config (products, landcover mask, resampling, rejuvenation, etc.).
- `propagation/season_2022_2023/season.yml` - season dates + assimilation events (used by skeleton + season).
- `meteo/` - sample station metadata/forcing CSVs.
- `obs/` - summarized observation CSVs for SCF/WETSNOW.
- `grids/` - land-cover grid for masking.
- `env/roi.gpkg` - ROI polygon used for clipping/masking.

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

Your project should have this structure (mirrors the Rofental example):

```
project/
|-- project.yml                        # Main configuration (required)
|-- obs_selection.config.yml           # Optional: observation selection presets
|-- env/
|   `-- roi.gpkg                       # Single-feature ROI polygon (required)
|-- grids/
|   |-- lc_<domain>_<resolution>.asc   # Land-cover classes for DA masking
|   |-- dem_<domain>_<resolution>.asc  # DEM (if used)
|   |-- svf_<domain>_<resolution>.asc  # Sky-view factor (if used)
|   `-- srf_<domain>_<resolution>.asc  # Slope/relief (if used)
|-- meteo/
|   |-- stations.csv                   # Station metadata
|   |-- station_001.csv                # Meteorological forcing data
|   `-- ...
|-- obs/
|   |-- stations/                      # Station metadata exports (optional)
|   |-- snowcover/                     # FSC inputs (Rofental naming)
|   |-- wetsnow/                       # Wet-snow SAR inputs (Rofental naming)
|   |-- summaries/                     # Precomputed summaries (Rofental)
|   `-- season_YYYY-YYYY/
|       |-- scf_summary.csv            # SCF observations (season summary)
|       |-- wet_snow_summary.csv       # Wet-snow observations (season summary)
|       `-- ...
|-- propagation/
|   `-- season_YYYY-YYYY/
|       `-- season.yml                 # Season definition used by season_skeleton
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
