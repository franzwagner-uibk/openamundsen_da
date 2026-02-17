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

Everything needed ships inside the image.

1. **Pull and alias the image (once):**

   This downloads the container and assigns a short local name.

   ```bash
   docker pull ghcr.io/franzwagner-uibk/openamundsen_da
   docker tag ghcr.io/franzwagner-uibk/openamundsen_da openamundsen_da
   ```

2. **Prepare a local folder:**

   This creates your local workspace.

   ```bash
   mkdir -p openamundsen-da
   cd openamundsen-da
   ```

3. **Set reusable variables:**

   This defines paths used in the next commands.

   ```bash
   IMAGE=openamundsen_da
   SETUP=/data/rofental
   PROJECT=/data/rofental/projects/project_2022_2023
   SCF_SUM=/data/rofental/obs/project_2022_2023/scf_summary.csv
   WET_SUM=/data/rofental/obs/project_2022_2023/wet_snow_summary.csv
   ```

4. **Copy the bundled Rofental example to your host:**

   Copy a ready-to-run project template from the local image into your workspace.

   ```bash
   docker run --rm -v "$(pwd):/data" \
     "$IMAGE" \
     cp -a /workspace/examples/rofental /data/rofental
   ```

5. **Generate the project skeleton (step\_\* directories/YAMLs from `project.yml`):**

   Create the step structure for the project based on the assimilation dates.

   ```bash
   docker run --rm -v "$(pwd):/data" \
     "$IMAGE" \
     python -m openamundsen_da.pipeline.project_skeleton \
       --project-dir "$PROJECT" \
       --setup-dir "$SETUP"
   ```

6. **Distribute observation summaries to steps (SCF + wet snow):**

   Create per-step `obs_*.csv` files from the project summaries using the assimilation dates in `project.yml`.

   ```bash
   docker run --rm -v "$(pwd):/data" \
     "$IMAGE" \
     oa-da-scf \
       --project-dir "$PROJECT" \
       --summary-csv "$SCF_SUM" \
       --overwrite

   docker run --rm -v "$(pwd):/data" \
     "$IMAGE" \
     oa-da-wetsnow-project \
       --project-dir "$PROJECT" \
       --summary-csv "$WET_SUM" \
       --overwrite
   ```

7. **Run the project (propagation + SCF/WETSNOW assimilation):**

   Run the full 2022/2023 project and write outputs to your local `openamundsen-da` folder. Set the number of cores to match your computer.

   ```bash
   docker run --rm -v "$(pwd):/data" \
     "$IMAGE" \
     python -m openamundsen_da.pipeline.project \
       --project-dir "$PROJECT" \
       --setup-dir "$SETUP" \
       --max-workers 8 \
       --monitor-perf \
       --overwrite
   ```

**Files to inspect**

after and during a run in `openamundsen-da/rofental/projects/project_2022_2023`:

- Logs: `project_2022_2023.log`.
- Perf monitor: `plots/perf/project_perf.png`
- Results: `plots/results/` (ensemble envelopes), `ensembles/posterior/` per step.
- Assimilation diagnostics: `plots/assim/` (weights/ESS), `assim/weights.csv`.

**The Rofental example bundle**

contains data covering the projects 2019-2023 (copied to `/data/rofental`):

- `rofental.yml` - openAMUNDSEN/global setup configuration
- `projects/project_2022_2023/project_2022_2023.yml` - project dates + DA keys + assimilation events
- `meteo/` - station metadata and forcing CSVs.
- `obs/` - summarized observation CSVs for SCF/WETSNOW, wet snow maps and snow cover maps, station observations (snow depth)
- `grids/` - dem, landcover, srf, svf grids (spatial resolution 100-500m)
- `env/roi.gpkg` - ROI polygon used for clipping/masking.

{: .highlight }

> ### First possible experiments with the Rofental example
>
> - modify `resolution` in `rofental/rofental.yml` (100/250/500 m).  
>   Test scale effects and runtime/memory tradeoffs. See [Configuration Reference (basic config)](https://openamundsen-da.pages.dev/guides/configuration#basic-configuration) and [Project Structure](https://openamundsen-da.pages.dev/project-structure#project-data-structure).
> - modify `ensemble_size` in `rofental/projects/project_2022_2023/project_2022_2023.yml` (e.g. 10/20/50).  
>   Test uncertainty representation versus compute cost. See [Configuration Reference (prior forcing)](https://openamundsen-da.pages.dev/guides/configuration#prior-forcing-configuration) and [Workflow (prior ensemble)](https://openamundsen-da.pages.dev/workflow#prior-ensemble-generation).
> - modify `sigma_p` and `sigma_t` in `rofental/projects/project_2022_2023/project_2022_2023.yml` (e.g. 0.2-2).  
>   Tune forcing perturbation strength. See [Configuration Reference (perturbation details)](https://openamundsen-da.pages.dev/guides/configuration#perturbation-details) and [Workflow](https://openamundsen-da.pages.dev/workflow#meteorological-forcing-perturbation).
> - modify `resampling.ess_threshold_ratio` in `rofental/projects/project_2022_2023/project_2022_2023.yml` (e.g. 0.2-0.8).  
>   Tune ESS-triggered resampling frequency (lower = less frequent, higher = more frequent). See [Configuration Reference (resampling + ESS)](https://openamundsen-da.pages.dev/guides/configuration#resampling-parameters) and [Workflow (ESS/resampling)](https://openamundsen-da.pages.dev/workflow#effective-sample-size-ess).
> - modify assimilation dates and variables in `rofental/projects/project_2022_2023/project_2022_2023.yml`.  
>   Define a new propagation sequence. Pick only dates that exist in `rofental/obs/project_2022_2023/scf_summary.csv` and `rofental/obs/project_2022_2023/wet_snow_summary.csv`, then rerun project preprocessing (`project_skeleton`, `oa-da-scf`, `oa-da-wetsnow-project`) before running `oa-da-project`. See [Observation Processing (quality control)](https://openamundsen-da.pages.dev/guides/observations#quality-control), [Running Experiments (build setup skeleton)](https://openamundsen-da.pages.dev/guides/experiments#5-build-setup-skeleton), and [CLI Reference (`oa-da-scf`)](https://openamundsen-da.pages.dev/guides/cli#oa-da-scf).

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

## Next Steps

- [Running Experiments]({{ '/guides/experiments/' | relative_url }}) - Set up your own project and run custom setups
- [Project Structure]({{ site.baseurl }}{% link project-structure.md %}) - Understand the directory layout
- [Workflow Overview]({{ site.baseurl }}{% link workflow.md %}) - Learn the DA workflow

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

# Or add to setup YAML
environment:
  GDAL_DATA: /usr/share/gdal
  PROJ_LIB: /usr/share/proj
```

See [Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) for more issues and solutions.




