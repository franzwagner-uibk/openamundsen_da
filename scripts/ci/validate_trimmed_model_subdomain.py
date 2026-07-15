from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ERROR_PATTERNS = [
    re.compile(r"\bERROR\b"),
    re.compile(r"\bCRITICAL\b"),
    re.compile(r"Traceback"),
    re.compile(r"\bException\b"),
]

SEVERE_WARNING_PATTERNS = [
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"\babort", re.IGNORECASE),
    re.compile(r"\bmissing\b", re.IGNORECASE),
    re.compile(r"\bnot found\b", re.IGNORECASE),
]

BENIGN_WARNING_PATTERNS = [
    re.compile(r"Coverage check: .* within tolerance", re.IGNORECASE),
    re.compile(
        r"Skipping analysis benchmark for .* on \d{4}-\d{2}-\d{2}: missing observation row",
        re.IGNORECASE,
    ),
    re.compile(
        r"Skipping wet_snow_line benchmark case at \d{4}-\d{2}-\d{2}: missing model values",
        re.IGNORECASE,
    ),
]


def _assert_non_empty(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing expected file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Expected non-empty file but got empty file: {path}")


def _check_logs(log_file: Path) -> None:
    _assert_non_empty(log_file)
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()

    fatal_lines: list[str] = []
    severe_warning_lines: list[str] = []
    for line in lines:
        if any(p.search(line) for p in ERROR_PATTERNS):
            fatal_lines.append(line)
            continue
        if "WARNING" in line.upper():
            if any(p.search(line) for p in SEVERE_WARNING_PATTERNS) and not any(
                p.search(line) for p in BENIGN_WARNING_PATTERNS
            ):
                severe_warning_lines.append(line)

    if fatal_lines:
        sample = "\n".join(fatal_lines[:20])
        raise ValueError(f"Model sub-domain integration log contains fatal lines:\n{sample}")
    if severe_warning_lines:
        sample = "\n".join(severe_warning_lines[:20])
        raise ValueError(f"Model sub-domain integration log contains severe warning lines:\n{sample}")


def _check_manifest(subdomain_root: Path) -> dict:
    manifest_path = subdomain_root / "subdomain_manifest.json"
    _assert_non_empty(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(data.get("run_mode", "")).lower() != "model":
        raise ValueError(f"Manifest run_mode is not 'model': {data.get('run_mode')!r}")

    stages = data.get("stages") or {}
    for stage in ("prepare", "run", "merge"):
        actual = str((stages.get(stage) or {}).get("status", "missing"))
        if actual != "completed":
            raise ValueError(f"Model sub-domain stage {stage!r} is {actual!r}, expected 'completed'")

    subdomains = data.get("subdomains") or {}
    if len(subdomains) < 8:
        raise ValueError(f"Expected at least 8 sub-domains in manifest, got {len(subdomains)}")

    for sid, meta in subdomains.items():
        status = str(meta.get("status", ""))
        if status.lower() != "success":
            raise ValueError(f"Sub-domain {sid} did not finish successfully (status={status!r})")
        setup_dir = Path(meta["setup_dir"])
        setup_yaml = Path(meta["setup_yaml"])
        _assert_non_empty(setup_yaml)
        if (setup_dir / "projects").exists():
            raise FileExistsError(f"Model sub-domain {sid} unexpectedly contains a projects/ directory")

        run_manifest = meta.get("run_manifest")
        if not run_manifest:
            raise ValueError(f"Sub-domain {sid} missing run_manifest path in manifest")
        run_manifest_path = Path(run_manifest)
        _assert_non_empty(run_manifest_path)
        run_data = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if str(run_data.get("status", "")).lower() != "success":
            raise ValueError(f"Sub-domain {sid} run manifest status is not success: {run_data.get('status')!r}")

        grids = setup_dir / "results" / "grids"
        if not grids.is_dir():
            raise FileNotFoundError(f"Missing model grid output directory for {sid}: {grids}")
        if not (list(grids.glob("*.nc")) or list(grids.glob("*.tif")) or list(grids.glob("*.tiff"))):
            raise FileNotFoundError(f"No model grid outputs found for {sid}: {grids}")

    return data


def _check_merged_results(subdomain_root: Path) -> None:
    grids = subdomain_root / "results" / "grids"
    if not grids.is_dir():
        raise FileNotFoundError(f"Missing merged model grids directory: {grids}")
    outputs = sorted([*grids.glob("*.nc"), *grids.glob("*.tif"), *grids.glob("*.tiff")])
    if not outputs:
        raise FileNotFoundError(f"No merged model grid outputs found under {grids}")
    for path in outputs:
        _assert_non_empty(path)


def validate_model_subdomain_root(subdomain_root: Path, log_file: Path) -> None:
    _check_logs(log_file)
    _check_manifest(subdomain_root)
    _check_merged_results(subdomain_root)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate trimmed model sub-domain integration outputs and logs.")
    p.add_argument("--subdomain-root", type=Path, required=True)
    p.add_argument("--log-file", type=Path, required=True)
    args = p.parse_args()

    validate_model_subdomain_root(args.subdomain_root, args.log_file)
    print(f"Model sub-domain integration output validation passed: {args.subdomain_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
