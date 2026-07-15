#!/usr/bin/env python3
"""Build and exercise the installed command from outside the source checkout."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


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
        command = " ".join(arguments)
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {command}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace").resolve()
    with tempfile.TemporaryDirectory(prefix="openamundsen-da-wheel-") as raw_tmp:
        tmp = Path(raw_tmp)
        wheel_dir = tmp / "wheel"
        wheel_dir.mkdir()
        _run(
            [sys.executable, "-m", "pip", "wheel", str(source), "--no-deps", "--wheel-dir", str(wheel_dir)],
            cwd=tmp,
        )
        wheels = tuple(wheel_dir.glob("openamundsen_da-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one built wheel, found: {wheels}")
        with zipfile.ZipFile(wheels[0]) as archive:
            wheel_members = tuple(archive.namelist())
        forbidden_roots = ("build/", "tests/", "scripts/", "docs/", "examples/", "context/")
        forbidden = [
            member
            for member in wheel_members
            if member.startswith(forbidden_roots) or "/build/" in member or "/tests/" in member
        ]
        if forbidden:
            raise RuntimeError(f"Wheel contains non-package workspace content: {forbidden[:10]}")

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
                str(wheels[0]),
            ],
            cwd=tmp,
        )
        site_packages = tuple(install_root.glob("lib/python*/site-packages"))
        if len(site_packages) != 1:
            raise RuntimeError(f"Expected one installed site-packages directory, found: {site_packages}")
        installed_site = site_packages[0]
        python = Path(sys.executable)
        cli = install_root / "bin" / "openamundsen-da"

        entry_points = _run(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata as m, json; "
                    "d=m.distribution('openamundsen-da'); "
                    "print(json.dumps(sorted(e.name for e in d.entry_points "
                    "if e.group == 'console_scripts')))"
                ),
            ],
            cwd=tmp,
            pythonpath=installed_site,
        )
        scripts = json.loads(entry_points.stdout)
        if scripts != ["openamundsen-da"]:
            raise RuntimeError(f"Unexpected installed console scripts: {scripts}")

        for arguments in (
            ["--help"],
            ["observations", "--help"],
            ["subdomains", "--help"],
            ["subdomains", "model", "--help"],
        ):
            _run([str(cli), *arguments], cwd=tmp, pythonpath=installed_site)

        failed = _run(
            [str(cli), "clean", str(tmp / "missing-project"), "--json"],
            cwd=tmp,
            check=False,
            pythonpath=installed_site,
        )
        if failed.returncode != 1:
            raise RuntimeError(f"Expected JSON failure exit code 1, got {failed.returncode}: {failed.stderr}")
        payload = json.loads(failed.stdout)
        if payload.get("ok") is not False or payload.get("command") != "clean":
            raise RuntimeError(f"Unexpected JSON failure envelope: {payload}")
        if payload.get("error", {}).get("type") != "ProjectValidationError":
            raise RuntimeError(f"Unexpected JSON failure type: {payload}")

    print("Installed-wheel smoke test passed; console scripts: openamundsen-da")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
