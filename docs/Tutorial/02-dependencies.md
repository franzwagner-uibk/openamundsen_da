---
layout: default
title: 2. Dependencies
parent: Tutorial
nav_order: 2
permalink: /tutorial/dependencies/
---

# 2. Dependencies

## Docker

### Container and Images

### Commands

### OS Dependency

### Runtime and resources

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

Pull the openAMUNDSEN-DA image:

```bash
docker pull ghcr.io/franzwagner-uibk/openamundsen_da:latest
```

### Commands

Use one-off container runs (host folder mounted to `/data`):

```bash
docker run --rm -v "<host_path>:/data" ghcr.io/franzwagner-uibk/openamundsen_da:latest <command>
```

You can also define reusable shell variables (`$img`, `$setup`, `$project`) and reuse them in all tutorial steps.

### OS Dependency

Linux:

- Ensure your user can access Docker (`docker` group or `sudo`)
- Typical install:
  ```bash
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-plugin
  sudo usermod -aG docker $USER
  sudo systemctl enable --now docker
  ```

Mac:

- Docker Desktop is the recommended runtime

Windows (WSL):

- Docker Desktop + WSL2 backend is required
- Run the tutorial commands in PowerShell (or WSL shell)

### Runtime and resources

- Set `--cpus` to the number of cores available in Docker on your machine.
- Use `--max-workers` in project/subdomain runs to control parallelism.
- For stable CPU usage in numerical libraries, set:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```
