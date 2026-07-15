---
layout: default
title: Releases and Distribution
nav_order: 9
---

# Releases and Distribution

openAMUNDSEN-DA releases one tested Python distribution and one tested
multi-architecture container. The v0.9 release does not include conda-forge.

The coupled openAMUNDSEN model remains a separate upstream project: use its
[technical documentation](https://doc.openamundsen.org/),
[GitHub repository](https://github.com/openamundsen/openamundsen) or
[PyPI package](https://pypi.org/project/openamundsen/) directly. The
openAMUNDSEN-DA container already includes the compatible model dependency.

## Where releases are published

| Artifact | Location | Stable selector |
| --- | --- | --- |
| Python package | [PyPI](https://pypi.org/project/openamundsen-da/) | `pip install openamundsen-da` |
| Container | [GitHub Container Registry](https://github.com/franzwagner-uibk/openamundsen_da/pkgs/container/openamundsen_da) | `ghcr.io/franzwagner-uibk/openamundsen_da:latest` |
| Archives and evidence | [GitHub Releases](https://github.com/franzwagner-uibk/openamundsen_da/releases) | exact release tag |

Use an exact version or digest for reproducible work. `edge` follows the latest
green commit on `main`; it is not a release. Main also publishes an immutable
`sha-<full-commit>` tag and never updates `latest`.

## Release gates

The tag is the sole version source. `v0.9.0rc1` produces `0.9.0rc1` and
`v0.9.0` produces `0.9.0`. The release workflow rejects all other tag forms or
a wheel/sdist version mismatch.

Before publication, the exact release artifacts pass:

1. strict wheel and source-archive content checks;
2. installation and CLI checks on Linux, macOS and Windows with Python 3.11,
   3.12, 3.13 and 3.14;
3. a critical-vulnerability scan of the staged multiarch image built from the
   tested wheel, never from an editable source checkout; and
4. unit and three integration suites against that exact staged image digest on
   the trusted Lenovo P8 runner.

After the gates pass, the workflow promotes the tested digest to the public
version tags without rebuilding it.

Release candidates publish to TestPyPI, an exact prerelease GHCR tag and a
GitHub prerelease. Stable tags wait for approval in the `pypi` GitHub
environment, then publish to PyPI, the exact GHCR version, `latest` and a
GitHub Release. Release assets include SHA-256 checksums and an SPDX JSON SBOM;
the archives and container digest receive GitHub artifact attestations.

## Repository setup

Create two GitHub environments:

- `testpypi` for release-candidate publication; and
- `pypi` with at least one required reviewer for stable publication.

Register `.github/workflows/release.yml` as a Trusted Publisher for the
`openamundsen-da` project on both TestPyPI and PyPI. Set the matching GitHub
environment name in each publisher. No PyPI token or GHCR personal access
token belongs in repository secrets; the workflows use OIDC and the scoped
repository `GITHUB_TOKEN`.

## Release procedure

1. Merge the release candidate only after the normal CI workflow is green.
2. Create and push an exact release-candidate tag such as `v0.9.0rc1`.
3. Confirm the TestPyPI install, GHCR prerelease image, GitHub prerelease,
   checksums, SBOM and attestations.
4. Fix issues on a new commit and issue a new RC tag; never reuse a tag or
   overwrite an artifact.
5. After the RC rehearsal is accepted, create and push the stable tag
   `v0.9.0`.
6. Review the stable workflow evidence and approve the `pypi` environment.

The release workflow verifies that the tagged commit belongs to `main`. It
publishes only after all portable and trusted integration gates are green.
