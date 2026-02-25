---
layout: default
title: 2. Dependencies
parent: Tutorial
nav_order: 2
permalink: /tutorial/dependencies/
---

# 2. Dependencies

## Docker

Before running openAMUNDSEN-DA, make sure [Docker](https://docs.docker.com)  is available and working.

This tutorial uses a **Docker image** and a **single interactive container session**:

- **Image** = the packaged runtime (Python, openAMUNDSEN-DA, dependencies)
- **Container** = one running instance of that image
- **Bind mount** = your local tutorial folder is mounted into the container as `/data`
- **Working directory** = we start the shell in `/data`, so tutorial commands can be copied directly

## Step 1. Install and Verify Docker

{: .step }
> Install Docker on your host, verify the runtime, and pull the tutorial image used in this tutorial.

System requirements:

- Operating system: Windows 10/11, macOS, or Linux
- Memory: 16 GB RAM minimum (32 GB recommended for larger ensembles)
- CPU: Multi-core processor (parallel runs scale with available cores)

{: .references }
> If you are new to Docker, these official docs pages are the most relevant for this tutorial:
>
> - [Docker overview](https://docs.docker.com/get-started/docker-overview/)
> - [Get Docker (installation)](https://docs.docker.com/get-docker/)
> - [Bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)
> - [`docker run` reference](https://docs.docker.com/reference/cli/docker/container/run/)
> - [Docker Desktop + WSL 2 (Windows)](https://docs.docker.com/desktop/features/wsl/)

### Docker Installation

Install Docker on the **host machine** (choose exactly one path):

1. **Windows (Docker Desktop + WSL2)**:
   install [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/), enable the WSL2 backend, and enable WSL integration for your distro.
2. **macOS (Docker Desktop)**:
   install [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/).
3. **Linux (native host)**:
   install [Docker Engine](https://docs.docker.com/engine/install/) and the [Compose plugin](https://docs.docker.com/compose/install/linux/) by following the official Docker documentation.

{: .note }
> On Linux, if `docker` requires `sudo` after installation, follow the Docker post-install steps in the official Docker documentation.

{: .warning }
> Do **not** install Docker Engine inside WSL if you are using Docker Desktop on Windows.
> Docker Desktop provides Docker to WSL through the WSL integration.

## Continuing Later (Restart the Tutorial Container Shell)

If you continue the tutorial on another day (for example after shutting down your computer),
your previous tutorial container will no longer be running. That is expected.

Important:

- the tutorial uses `--rm`, so the container itself is temporary,
- your files are **not** lost because they live in your local tutorial workspace (bind-mounted as `/data`).

If Docker is already installed and the image is already pulled, you can restart the
tutorial shell directly with the same command from Step 2 (use the same host path you used before):

```bash
docker run --rm -it \
  -v "/absolute/path/to/tutorial-workdir:/data" \
  -w /data \
  --cpus 8 \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 \
  ghcr.io/franzwagner-uibk/openamundsen_da:latest \
  bash --noprofile --norc
```


Verify Docker:

**🟢 Run this command:**

```bash
docker run hello-world
```

Pull the openAMUNDSEN-DA image:

**🟢 Run this command:**

```bash
docker pull ghcr.io/franzwagner-uibk/openamundsen_da:latest
```

<details markdown="block">
  <summary>Windows / PowerShell note (same commands)</summary>

Use the same Docker commands on Windows. Recommended: run them in **WSL** after Docker Desktop + WSL integration is enabled. In **PowerShell**, adjust only host path syntax (for example `C:/...:/data`) and line continuation (PowerShell uses `` ` `` instead of `\`).

</details>

{: .checks }
> Confirm Docker is working and the tutorial image is available locally.
>
> - `docker run hello-world` prints a success message from Docker.
> - `docker pull ...openamundsen_da:latest` completes without errors.
> - Optional: `docker image ls | grep openamundsen_da` shows the image locally.

## Step 2. Start the Tutorial Container Shell

{: .step }
> Start one interactive tutorial container and run the next tutorial commands **inside that shell**.
>
> This avoids repeating the host path mount on every command and keeps the later chapters copy-paste friendly.

openAMUNDSEN-DA is designed for multi-core processing. In the container start command
below, set `--cpus` to the number of CPU cores/threads you want to make available to
Docker/openAMUNDSEN-DA on your machine (for example `--cpus 8` is only an example).
Also replace `"/absolute/path/to/tutorial-workdir"` with your own local tutorial folder
path.

Keep `--max-workers` in later project commands consistent with the CPU capacity you
assign here (do not set it much higher than `--cpus`).

### Commands

Recommended tutorial shell start (run once):

**🟢 Run this command:**

```bash
docker run --rm -it \
  -v "/absolute/path/to/tutorial-workdir:/data" \
  -w /data \
  --cpus 8 \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 \
  ghcr.io/franzwagner-uibk/openamundsen_da:latest \
  bash --noprofile --norc
```

<details markdown="block">
  <summary>Windows PowerShell variant (same container start command)</summary>

**🟢 Run this command:**

```powershell
docker run --rm -it `
  -v "C:/absolute/path/to/tutorial-workdir:/data" `
  -w /data `
  --cpus 8 `
  -e OMP_NUM_THREADS=1 `
  -e OPENBLAS_NUM_THREADS=1 `
  -e MKL_NUM_THREADS=1 `
  -e NUMEXPR_NUM_THREADS=1 `
  ghcr.io/franzwagner-uibk/openamundsen_da:latest `
  bash --noprofile --norc
```

</details>

Leave the tutorial container when you are done:

```bash
exit
```

Why these options matter for reproducible and stable tutorial runs:

- `-v "...:/data"` mounts your local tutorial folder into the container.
- `-w /data` starts the shell in the mounted folder.
- `--cpus` limits CPU capacity available to Docker/openAMUNDSEN-DA.
- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` disable nested BLAS/OpenMP threading (important for stable parallel runs).
- `bash` starts an interactive shell inside the image.

Files written under `/data/...` are stored in your local tutorial folder (host
machine). Files written elsewhere in the container are typically ephemeral and
disappear when the container exits.

Use the command as shown (`bash --noprofile --norc`). Shell startup files can override the
activated environment in this image (for example by activating a `base` conda env) and make `python` / `oa-da-*` commands unavailable.

{: .checks }
> Verify that you are inside the container and working in `/data`.
>
> - Container shell opens and your prompt changes to the container environment.
> - Running `pwd` inside the container returns `/data`.
> - Later tutorial commands should be run inside this shell (not on the host).

```bash
pwd
```

Example of a later tutorial command (do not run yet):

```bash
oa-da-snowcover --input-dir /data/rofental/obs/snowcover --project-label project_2022_2023 --setup-dir /data/rofental --overwrite
```

{: .references }
> Useful Docker references for the command above:
>
> - [`docker run` reference](https://docs.docker.com/reference/cli/docker/container/run/) (syntax and options)
> - [Bind mounts](https://docs.docker.com/engine/storage/bind-mounts/) (host folder -> container path)

## Step 3. Initialize the Tutorial Workspace

{: .step }
> Copy the bundled Rofental example from the image into your mounted local tutorial workspace (run once, inside the container shell).

### Command

**🟢 Run this command:**

```bash
cp -a /workspace/examples/rofental /data/rofental
```

<details markdown="block">
  <summary>Windows / PowerShell users (recommended approach)</summary>

Start the tutorial container from **WSL** if possible. If you start it from **PowerShell** (using the variant above), the commands inside the container shell are still identical.

</details>

This command copies the bundled example from the image into your mounted local tutorial
workspace:

- `/workspace/examples/rofental` is the example bundled in the Docker image.
- `/data/rofental` is your local tutorial workdir (via bind mount).
- `cp -a` copies the example recursively and preserves file metadata.

Result: the bundled example is copied into your local tutorial workdir (`/data`), where
you can inspect and rerun everything.

Bash is shown as the primary command syntax in the tutorial (works directly on
Linux/macOS and well in WSL/Git Bash on Windows).

{: .checks }
> Confirm the example was copied to your mounted workspace.
>
> - `ls /data/rofental` shows the copied example folder contents inside the container.
> - The same folder appears on the host in your local tutorial workdir.

```bash
ls /data/rofental
```

### Quick peek into the bundled setup data (optional, recommended)

At this point, the example is now in your local workspace and you can inspect the shared
setup inputs before any preprocessing or DA project execution.

```bash
ls /data/rofental/meteo
```

Rofental station file (`/data/rofental/meteo/stations.csv`):

| id | name | x | y | alt |
| --- | --- | --- | --- | --- |
| bellavista | Bella Vista | 636823 | 5182569 | 2805 |
| proviantdepot | Proviantdepot | 639377 | 5187724 | 2659 |
| latschbloder | Latschbloder | 637854 | 5184641 | 2919 |

This station table provides the metadata (station ID, coordinates, elevation) for the
meteorological forcing station CSV files in `meteo/`.

Rofental bellavista forcing file snippet (`/data/rofental/meteo/bellavista.csv`):

| date | temp | precip | sw_in | rel_hum | wind_speed | wind_dir |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023-05-24 17:00:00 | 273.78 | 1.20 | 127.00 | 99.60 | 1.08 | 192.34 |
| 2023-05-24 18:00:00 | 273.25 | 1.00 | 74.50 | 99.60 | 2.20 | 191.87 |
| 2023-05-24 19:00:00 | 272.47 | 1.20 | 31.83 | 99.60 | 1.68 | 194.09 |
| 2023-05-24 20:00:00 | 272.35 | 1.70 | 8.00 | 99.60 | 1.40 | 192.78 |
| 2023-05-24 21:00:00 | 272.30 | 1.20 | 0.17 | 99.60 | 0.80 | 203.38 |

This gives a concrete example of the forcing time-series format used by the setup before
the tutorial moves on to preprocessing and DA-specific inputs.

Quick look at the observation folders bundled with the setup:

```bash
ls /data/rofental/obs
```

Example raw DA observation raster files:

- FSC / snow cover raster: `/data/rofental/obs/snowcover/s2_fsc_snowflake_rofental_2023_01_01.tif`
- Wet-snow raster: `/data/rofental/obs/wetsnow/WSM_S1A_SAR_track117_2023_05_11_17_07_26.tif`

Rofental Proviantdepot snow observation file snippet (`/data/rofental/obs/stations/proviantdepot.csv`):

| time | snow_depth | swe |
| --- | ---: | ---: |
| 2022-10-01 00:00:00 | 0.10 | 15.12 |
| 2022-10-01 01:00:00 | 0.10 | 14.78 |
| 2022-10-01 02:00:00 | 0.10 | 14.78 |
| 2022-10-01 03:00:00 | 0.10 | 14.80 |
| 2022-10-01 04:00:00 | 0.10 | 15.20 |

This station observation file is used later for evaluation/diagnostics (not as the raster
DA input itself), which helps distinguish the different observation roles in the tutorial.

{: .references }
> Continue after the workspace copy succeeds:
>
> - [3. Workflow](/tutorial/workflow/) (conceptual execution order before running commands)
