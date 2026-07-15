#!/usr/bin/env python3
"""Install one wheel and exercise its public interface outside the source tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--expected-version")
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Only exercise dependency-free import, metadata and parser paths",
    )
    return parser.parse_args()


def _run(
    arguments: list[str],
    *,
    cwd: Path,
    check: bool = True,
    pythonpath: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(arguments)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def main() -> int:
    args = _arguments()
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        raise FileNotFoundError(f"Wheel not found: {wheel}")

    with tempfile.TemporaryDirectory(prefix="openamundsen-da-wheel-") as raw_tmp:
        tmp = Path(raw_tmp)
        install_root = tmp / "install"
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--ignore-installed",
                "--prefix",
                str(install_root),
                str(wheel),
            ],
            cwd=tmp,
        )
        site_candidates = tuple(
            path
            for path in install_root.rglob("*")
            if path.is_dir() and path.name in {"site-packages", "dist-packages"}
        )
        if len(site_candidates) != 1:
            raise RuntimeError(f"Expected one installed package directory, found {site_candidates}")
        site_packages = site_candidates[0]
        cli_names = {"openamundsen-da", "openamundsen-da.exe"}
        cli_candidates = tuple(
            path for path in install_root.rglob("*") if path.is_file() and path.name in cli_names
        )
        if len(cli_candidates) != 1:
            raise RuntimeError(f"Expected one installed CLI executable, found {cli_candidates}")
        cli = cli_candidates[0]
        python = Path(sys.executable)
        if not site_packages.is_dir() or not cli.is_file():
            raise RuntimeError(f"Incomplete prefix install: site_packages={site_packages}, cli={cli}")

        inspection = _run(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata as m, json, openamundsen_da as p; "
                    "d=m.distribution('openamundsen-da'); "
                    "print(json.dumps({'metadata_version': d.version, 'runtime_version': p.__version__, "
                    "'origin': p.__file__, 'scripts': sorted(e.name for e in d.entry_points "
                    "if e.group == 'console_scripts')}))"
                ),
            ],
            cwd=tmp,
            pythonpath=site_packages,
        )
        details = json.loads(inspection.stdout)
        if details["scripts"] != ["openamundsen-da"]:
            raise RuntimeError(f"Unexpected installed console scripts: {details['scripts']}")
        if details["runtime_version"] != details["metadata_version"]:
            raise RuntimeError(f"Runtime/metadata version mismatch: {details}")
        if args.expected_version and details["metadata_version"] != args.expected_version:
            raise RuntimeError(
                f"Installed version {details['metadata_version']!r} does not match expected {args.expected_version!r}"
            )
        if not Path(details["origin"]).resolve().is_relative_to(install_root.resolve()):
            raise RuntimeError(f"Package imported outside the isolated environment: {details['origin']}")

        version_output = _run([str(cli), "--version"], cwd=tmp, pythonpath=site_packages).stdout.strip()
        if details["metadata_version"] not in version_output:
            raise RuntimeError(f"CLI version output does not contain installed version: {version_output!r}")

        help_outputs: dict[tuple[str, ...], str] = {}
        for arguments in (
            ["--help"],
            ["observations", "--help"],
            ["subdomains", "--help"],
            ["subdomains", "model", "--help"],
        ):
            completed = _run([str(cli), *arguments], cwd=tmp, pythonpath=site_packages)
            help_outputs[tuple(arguments)] = completed.stdout

        subdomain_help = help_outputs[("subdomains", "--help")]
        for command in ("prepare", "run", "merge", "render", "model"):
            if command not in subdomain_help:
                raise RuntimeError(f"Missing subdomain command in installed help: {command}")
        for removed in ("pipeline", "model-pipeline", "model-prepare", "model-run", "model-merge", "plot"):
            if removed in subdomain_help:
                raise RuntimeError(f"Removed subdomain alias remains installed: {removed}")

        model_help = help_outputs[("subdomains", "model", "--help")]
        for command in ("prepare", "run", "merge"):
            if command not in model_help:
                raise RuntimeError(f"Missing plain-model subdomain command in installed help: {command}")

        if not args.portable:
            failed = _run(
                [str(cli), "clean", str(tmp / "missing-project"), "--json"],
                cwd=tmp,
                check=False,
                pythonpath=site_packages,
            )
            if failed.returncode != 1:
                raise RuntimeError(f"Expected JSON failure exit code 1, got {failed.returncode}: {failed.stderr}")
            payload = json.loads(failed.stdout)
            if payload.get("ok") is not False or payload.get("command") != "clean":
                raise RuntimeError(f"Unexpected JSON failure envelope: {payload}")
            if payload.get("error", {}).get("type") != "ProjectValidationError":
                raise RuntimeError(f"Unexpected JSON failure type: {payload}")

    print(
        "Installed-wheel smoke test passed: "
        f"version={details['metadata_version']}, portable={args.portable}, console_scripts=openamundsen-da"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
