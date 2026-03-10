---
layout: default
title: 2. Installation
parent: How to Use
nav_order: 2
permalink: /tutorial/dependencies/
---

# 2. Installation

This tutorial uses Docker so that Python, GDAL/PROJ, openAMUNDSEN, and openAMUNDSEN-DA
are already bundled in one runtime environment.

In Docker terms:

- the **image** is the packaged runtime
- the **container** is one running instance of that image
- a **bind mount** makes a local host folder visible inside the container

In this tutorial, your local tutorial folder is mounted as `/data` inside the container.
That means you can edit files locally with your normal editor while running commands in
the container shell.

## 1. Install Docker And Verify It

Install Docker on the host machine:

- Windows/macOS: Docker Desktop
- Linux: Docker Engine + Compose plugin

Verify that Docker works:

**🟢 Run this command:**

```bash
docker run hello-world
```

Pull the tutorial image:

**🟢 Run this command:**

```bash
docker pull ghcr.io/franzwagner-uibk/openamundsen_da:latest
```

## 2. Copy The Bundled Example To Your Local Tutorial Folder

The tutorial assumes you first copy the bundled Rofental example out of the image and
only then start an interactive container that mounts that local copy.

Choose a local host folder that will hold the tutorial files. The command below copies the
bundled example from the image into that folder.

**🟢 Run this command:**

```bash
docker run --rm \
  -v "/absolute/path/to/tutorial-workdir:/data" \
  ghcr.io/franzwagner-uibk/openamundsen_da:latest \
  bash -lc 'cp -a /workspace/examples/rofental /data/rofental'
```

After this command, your local host folder contains a full editable copy of the example at
`rofental/`.

## 3. Start The Tutorial Container Shell

Now start one interactive container and mount the same local tutorial folder as `/data`.
The tutorial commands in later chapters are executed inside this shell.

**🟢 Run this command:**

```bash
docker run --rm -it \
  -v "/absolute/path/to/tutorial-workdir:/data" \
  -w /data/rofental \
  --cpus 8 \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 \
  ghcr.io/franzwagner-uibk/openamundsen_da:latest \
  bash --noprofile --norc
```

Why these options are used:

- `-v ...:/data` mounts your local tutorial folder into the container
- `-w /data/rofental` starts you inside the copied setup
- `--cpus 8` is an example CPU limit for the container
- the BLAS/OpenMP variables prevent nested threading and unstable oversubscription

## 4. Editing Files During The Tutorial

You do not need a text editor inside the container. Edit files in your local tutorial
folder on the host machine. Because that folder is mounted into the container as `/data`,
changes made locally are immediately visible inside the running container shell.

The two files you will inspect most often are:

- `/data/rofental/rofental.yml`
- `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`

Continue with [3. Example Data: Rofental]({{ site.baseurl }}{% link Tutorial/03-example-data-rofental.md %}).
