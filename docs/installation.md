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

## Tutorial first

The full Rofental walkthrough has been moved to the Tutorial section:

- [Tutorial overview]({{ '/tutorial/' | relative_url }})
- [4. Framework]({{ '/tutorial/framework/' | relative_url }})
- [5. Pre-processing]({{ '/tutorial/pre-processing/' | relative_url }})
- [6. Running the project]({{ '/tutorial/running-the-project/' | relative_url }})

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
sudo systemctl enable --now docker
# Then refresh your login session:
# - log out and back in, or
# - reboot, or
# - run: newgrp docker
```

Until your session is refreshed, running Docker commands with `sudo` can be necessary.

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



