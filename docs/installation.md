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

## How to Use First

The full Rofental walkthrough lives in the How to Use section:

- [How to Use overview]({{ '/tutorial/' | relative_url }})
- [1. Overview]({{ '/tutorial/openamundsen-da/' | relative_url }})
- [3. Example Data: Rofental]({{ '/tutorial/example-data-rofental/' | relative_url }})
- [4. Preprocessing]({{ '/tutorial/pre-processing/' | relative_url }})
- [5. Running the Model]({{ '/tutorial/running-the-project/' | relative_url }})

## Developer install (clone + compose)

Use this when you want to modify the code or run Compose with mounted source.

1. Clone the repo:

   ```bash
   git clone https://github.com/franzwagner-uibk/openamundsen_da.git
   cd openamundsen_da
   ```

2. (Optional) Build the distribution and a local image instead of pulling:

   ```bash
   python -m pip install build twine
   bash scripts/ci/build_distribution.sh
   docker build -t openamundsen-da:local .
   ```

3. Add the source-development Compose overlay. `PROJ` defaults to
   `./examples/rofental`; override it inline when needed:

   ```bash
   IMAGE=openamundsen-da:local PROJ=/path/to/project \
   docker compose -f compose.yml -f compose.dev.yml run --rm oa \
     python -c "import openamundsen_da; print(openamundsen_da.__version__)"
   ```

Release-mode `compose.yml` mounts the setup and cache only. It executes the
non-editable wheel installed in the image; it does not shadow that wheel with a
source checkout.

---

## Next Steps

- [Running Experiments]({{ '/guides/experiments/' | relative_url }}) - Set up your own project and run custom setups
- [Project Structure]({{ site.baseurl }}{% link project-structure.md %}) - Understand the directory layout
- [Workflow Overview]({{ site.baseurl }}{% link workflow.md %}) - Learn the data assimilation workflow

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

**Problem**: Tutorial or example files copied from a container are owned by `root`

This usually happens on Linux when a bind-mounted copy command was run with `sudo docker ...`.
The container process writes into the mounted host directory as `root`, so later edits to
`rofental.yml`, project YAMLs, or generated files may ask for `sudo`.

Preferred fix:

```bash
sudo usermod -aG docker $USER
# Then log out/in, reboot, or run: newgrp docker
```

If the files are already root-owned, repair ownership on the host:

```bash
cd /path/to/tutorial-workdir
sudo chown -R "$USER":"$USER" rofental
chmod -R u+rwX rofental
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
