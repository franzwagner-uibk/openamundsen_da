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


### Commands

Use one-off container runs (host folder mounted to `/data`):

```bash
docker run --rm -v "<host_path>:/data" ghcr.io/franzwagner-uibk/openamundsen_da:latest <command>
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


For this tutorial, commands are shown with explicit image names and paths (instead of shell
variables) to keep the workflow easier to follow.

{: .highlight }
> Tutorial command style: Bash is shown as the primary command syntax (works directly on Linux/macOS and well in WSL/Git Bash on Windows).

<details markdown="block">
  <summary>Windows / PowerShell users (recommended approach)</summary>

You can still follow the same tutorial:

- easiest path: run the Bash commands in **WSL**
- alternatively: use **PowerShell** and translate only the shell syntax (line continuation / quoting)

The Docker image, container paths (`/data/...`), and framework commands stay the same.
</details>

### OS Dependency

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

- Set `--cpus` to the number of cores available in Docker on your machine.
- Use `--max-workers` in project/subdomain runs to control parallelism.
- For stable CPU usage in numerical libraries, set:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
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

