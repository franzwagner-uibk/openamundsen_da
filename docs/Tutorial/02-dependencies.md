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

{: .note }
> Docker background (recommended if you are new to Docker):
> - [Docker overview](https://docs.docker.com/get-started/docker-overview/)
> - [Install Docker](https://docs.docker.com/get-docker/)
> - [Bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)

System requirements:

- Operating system: Windows 10/11, macOS, or Linux
- Memory: 16 GB RAM minimum (32 GB recommended for larger ensembles)
- CPU: Multi-core processor (parallel runs scale with available cores)

Install Docker:

- Windows: install Docker Desktop and use the WSL2 backend
- macOS: install Docker Desktop
- Linux: install Docker Engine + Compose plugin

Verify Docker:

```bash
docker run hello-world
```
<details markdown="block">
  <summary>Windows / PowerShell note (same command)</summary>

Use the same command on Windows in one of these ways:

- **Recommended:** run the Bash command in **WSL** (works as shown).
- **PowerShell:** keep the same Docker/image/container paths, adjust only:
  - host mount path syntax (e.g. `C:/...:/data`)
  - line continuation (PowerShell uses the backtick `` ` `` instead of `\`)

For Bash-specific host-shell constructs (for example heredocs), prefer WSL/Git Bash or use a PowerShell here-string equivalent.
</details>


Pull the openAMUNDSEN-DA image:

```bash
docker pull ghcr.io/franzwagner-uibk/openamundsen_da:latest
```
<details markdown="block">
  <summary>Windows / PowerShell note (same command)</summary>

Use the same command on Windows in one of these ways:

- **Recommended:** run the Bash command in **WSL** (works as shown).
- **PowerShell:** keep the same Docker/image/container paths, adjust only:
  - host mount path syntax (e.g. `C:/...:/data`)
  - line continuation (PowerShell uses the backtick `` ` `` instead of `\`)

For Bash-specific host-shell constructs (for example heredocs), prefer WSL/Git Bash or use a PowerShell here-string equivalent.
</details>

### Tutorial command model (important)

For this tutorial, you start **one interactive container** and run the tutorial commands
**inside that container shell**. This avoids repeating the host path mount on every command
and makes later chapters copy-paste friendly across operating systems.

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
  <summary>Windows PowerShell variant (start the tutorial container shell)</summary>

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

What this command does:

- mounts your local tutorial folder into the container as `/data`
- starts a Bash shell in `/data` (`-w /data`)
- limits CPU usage (`--cpus 8`, example value)
- disables nested BLAS/OpenMP threading (important for stable parallel runs)
- keeps tutorial results persistent on your machine because `/data` is a bind mount

{: .note }
> Files written under `/data/...` are stored in your local tutorial folder (host machine). Files written elsewhere in the container are typically ephemeral and disappear when the container exits.

{: .note }
> Use the command as shown (`bash`, not `bash -l`). A login shell can override the activated environment in this image and make `python` / `oa-da-*` commands unavailable.

After the shell starts, later tutorial commands look like:

```bash
oa-da-snowcover --input-dir /data/rofental/obs/snowcover --project-label project_2022_2023 --setup-dir /data/rofental --overwrite
```

Leave the tutorial container when you are done:

```bash
exit
```

{: .highlight }
> All command blocks in the next tutorial chapters are shown as commands **inside this running container shell**.

### Initialize the tutorial workspace (copy the bundled example)

Inside the tutorial container shell, copy the bundled Rofental example to your mounted
workspace (run once):

```bash
cp -a /workspace/examples/rofental /data/rofental
```

{: .note }
> This copies the example from the image (`/workspace/examples/rofental`) into your local tutorial workdir (mounted as `/data`), where you can inspect and rerun everything.

{: .highlight }
> Tutorial command style: Bash is shown as the primary command syntax (works directly on Linux/macOS and well in WSL/Git Bash on Windows).

<details markdown="block">
  <summary>Windows / PowerShell users (recommended approach)</summary>

You can still follow the same tutorial:

- easiest path: start the tutorial container from **WSL** and run the chapter commands as shown
- alternatively: start the container from **PowerShell** (using the variant above); once inside the container shell, the tutorial commands are identical

The Docker image, container paths (`/data/...`), and framework commands stay the same.
</details>

### Operating system notes

Linux:

- Ensure your user can access Docker (`docker` group or `sudo`)
- Typical install:
  ```bash
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-plugin
  sudo usermod -aG docker <your-user>
  sudo systemctl enable --now docker
  ```
<details markdown="block">
  <summary>Windows / PowerShell note (same command)</summary>

Use the same command on Windows in one of these ways:

- **Recommended:** run the Bash command in **WSL** (works as shown).
- **PowerShell:** keep the same Docker/image/container paths, adjust only:
  - host mount path syntax (e.g. `C:/...:/data`)
  - line continuation (PowerShell uses the backtick `` ` `` instead of `\`)

For Bash-specific host-shell constructs (for example heredocs), prefer WSL/Git Bash or use a PowerShell here-string equivalent.
</details>


Mac:

- Docker Desktop is the recommended runtime

Windows (WSL):

- Docker Desktop + WSL2 backend is required
- Run the tutorial commands in PowerShell (or WSL shell)

{: .note }
> If you use PowerShell instead of WSL, prefer forward slashes in Docker mount paths where possible (for example `C:/path/to/workdir:/data`).

### Runtime and resources

- Set `--cpus` (in the tutorial shell start command above) to the number of cores available in Docker on your machine.
- Use `--max-workers` in project/subdomain runs to control parallelism.
- For stable CPU usage in numerical libraries, use these environment variables in the container start command: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` (already included in the tutorial command above).

{: .note }
> In this tutorial, these variables are already set in the recommended container startup command above.

