---
layout: default
title: 2. Dependencies
parent: Tutorial
nav_order: 2
permalink: /tutorial/dependencies/
---

# 2. Dependencies

## Docker

Before running openAMUNDSEN-DA, make sure Docker is available and working.

This tutorial uses a **Docker image** and a **single interactive container session**:

- **Image** = the packaged runtime (Python, openAMUNDSEN-DA, dependencies)
- **Container** = one running instance of that image
- **Bind mount** = your local tutorial folder is mounted into the container as `/data`
- **Working directory** = we start the shell in `/data`, so tutorial commands can be copied directly

## Step 1. Install and Verify Docker

{: .step }
> Goal: install Docker on your host, verify the runtime, and pull the tutorial image.

### Description

Install Docker on your system, confirm it works, and pull the tutorial image.

System requirements:

- Operating system: Windows 10/11, macOS, or Linux
- Memory: 16 GB RAM minimum (32 GB recommended for larger ensembles)
- CPU: Multi-core processor (parallel runs scale with available cores)

{: .note }

> Docker background (recommended if you are new to Docker):
>
> - [Docker overview](https://docs.docker.com/get-started/docker-overview/)
> - [Get Docker (installation)](https://docs.docker.com/get-docker/)
> - [Bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)
> - [`docker run` reference](https://docs.docker.com/reference/cli/docker/container/run/)
> - [Docker Desktop + WSL 2 (Windows)](https://docs.docker.com/desktop/features/wsl/)

### Commands

{: .commands }
> Run these commands on your **host machine** (not inside a container).

Install Docker (host machine):

- Windows: install [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) and use the WSL2 backend
- macOS: install [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/)
- Linux: install [Docker Engine](https://docs.docker.com/engine/install/) + [Compose plugin](https://docs.docker.com/compose/install/linux/)

Optional Linux post-install (host machine):

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker <your-user>
sudo systemctl enable --now docker
```

Verify Docker:

```bash
docker run hello-world
```

Pull the openAMUNDSEN-DA image:

```bash
docker pull ghcr.io/franzwagner-uibk/openamundsen_da:latest
```

<details markdown="block">
  <summary>Windows / PowerShell note (same commands)</summary>

Use the same Docker commands on Windows. Recommended: run them in **WSL**. In **PowerShell**, adjust only host path syntax (for example `C:/...:/data`) and line continuation (PowerShell uses `` ` `` instead of `\`).

</details>

### Checks

{: .checks }
> Confirm Docker is working and the tutorial image is available locally.

- `docker run hello-world` prints a success message from Docker.
- `docker pull ...openamundsen_da:latest` completes without errors.
- Optional: `docker image ls | grep openamundsen_da` shows the image locally.

### References

{: .references }
> Background reading for the Docker concepts used in this tutorial.

- External: Docker docs links listed in the note above.

## Step 2. Start the Tutorial Container Shell

{: .step }
> Goal: start one interactive tutorial container and run later commands inside it.

### Description

For this tutorial, you start **one interactive container** and run the tutorial commands
**inside that container shell**. This avoids repeating the host path mount on every command
and makes later chapters copy-paste friendly across operating systems.

{: .note }

> openAMUNDSEN-DA is designed for multi-core processing. In the container start command below, set `--cpus` to the number of CPU cores/threads you want to make available to Docker/openAMUNDSEN-DA on your machine (for example `--cpus 8` is only an example). Also replace `"/absolute/path/to/tutorial-workdir"` with your own local tutorial folder path.

{: .tip }
> Keep `--max-workers` in later project commands consistent with the CPU capacity you assign here (do not set it much higher than `--cpus`).

### Commands

{: .commands }
> Start the container once, then run the next tutorial chapters inside this shell.

Recommended tutorial shell start (run once):

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
  bash
```

<details markdown="block">
  <summary>Windows PowerShell variant (same container start command)</summary>

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
  bash
```

</details>

Leave the tutorial container when you are done:

```bash
exit
```

### Command Explanation

{: .commands }
> Why these options matter for reproducible and stable tutorial runs.

- `-v "...:/data"` mounts your local tutorial folder into the container.
- `-w /data` starts the shell in the mounted folder.
- `--cpus` limits CPU capacity available to Docker/openAMUNDSEN-DA.
- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` disable nested BLAS/OpenMP threading (important for stable parallel runs).
- `bash` starts an interactive shell inside the image.

{: .note }

> Files written under `/data/...` are stored in your local tutorial folder (host machine). Files written elsewhere in the container are typically ephemeral and disappear when the container exits.

{: .note }

> Use the command as shown (`bash`, not `bash -l`). A login shell can override the activated environment in this image and make `python` / `oa-da-*` commands unavailable.

### Checks

{: .checks }
> Verify that you are inside the container and working in `/data`.

- Container shell opens and your prompt changes to the container environment.
- Running `pwd` inside the container returns `/data`.
- Later tutorial commands should be run inside this shell (not on the host).

```bash
pwd
```

Example of a later tutorial command (do not run yet unless you already prepared inputs):

```bash
oa-da-snowcover --input-dir /data/rofental/obs/snowcover --project-label project_2022_2023 --setup-dir /data/rofental --overwrite
```

### References

{: .references }
> External references for the command options used above.

- External: [`docker run` reference](https://docs.docker.com/reference/cli/docker/container/run/)
- External: [Bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)

## Step 3. Initialize the Tutorial Workspace

{: .step }
> Goal: copy the bundled Rofental example from the image into your local tutorial workdir.

### Description

Inside the tutorial container shell, copy the bundled Rofental example to your mounted
workspace (run once).

### Commands

{: .commands }
> Run this command **inside the tutorial container shell**.

```bash
cp -a /workspace/examples/rofental /data/rofental
```

<details markdown="block">
  <summary>Windows / PowerShell users (recommended approach)</summary>

Start the tutorial container from **WSL** if possible. If you start it from **PowerShell** (using the variant above), the commands inside the container shell are still identical.

</details>

### Command Explanation

{: .commands }
> Source path is inside the image; target path is your mounted local workspace.

- `/workspace/examples/rofental` is the example bundled in the Docker image.
- `/data/rofental` is your local tutorial workdir (via bind mount).
- `cp -a` copies the example recursively and preserves file metadata.

{: .note }

> This copies the example from the image (`/workspace/examples/rofental`) into your local tutorial workdir (mounted as `/data`), where you can inspect and rerun everything.

{: .highlight }

> Tutorial command style: Bash is shown as the primary command syntax (works directly on Linux/macOS and well in WSL/Git Bash on Windows).

### Checks

{: .checks }
> Confirm the example was copied to your mounted workspace.

- `ls /data/rofental` shows the copied example folder contents inside the container.
- On the host, the same folder appears in your local tutorial workdir.

```bash
ls /data/rofental
```

### References

{: .references }
> Continue with the next chapter after the workspace copy succeeded.

- Internal: continue with the next chapter, [`3. Workflow`](/tutorial/workflow/).
