---
layout: default
title: Releases and Distribution
parent: Reference
nav_order: 5
---

# Releases and Distribution

openAMUNDSEN-DA publishes a tested Python distribution and a tested
multi-architecture container. The coupled openAMUNDSEN model remains a separate
upstream project; see its [technical documentation](https://doc.openamundsen.org/).

## Available now

- The stable Python package is available from PyPI; see the
  [installation instructions]({{ site.baseurl }}{% link installation.md %}).
- [GitHub Container Registry](https://github.com/openamundsen/openamundsen-da/pkgs/container/openamundsen-da)
  provides the tested `0.9.4` multi-architecture container.
- [GitHub Releases](https://github.com/openamundsen/openamundsen-da/releases)
  provides release archives, checksums and supporting release evidence.
- [conda-forge](https://anaconda.org/conda-forge/openamundsen-da) provides the
  conda package through its independently maintained feedstock.

The documentation uses the stable `0.9.4` image in its commands. Conda-forge
updates can follow PyPI releases after a short feedstock delay.
