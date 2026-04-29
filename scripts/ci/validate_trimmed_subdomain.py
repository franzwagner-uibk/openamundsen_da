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
        raise ValueError(f"Sub-domain integration log contains fatal lines:\n{sample}")
    if severe_warning_lines:
        sample = "\n".join(severe_warning_lines[:20])
        raise ValueError(f"Sub-domain integration log contains severe warning lines:\n{sample}")


def _check_manifest(subdomain_root: Path) -> dict:
    manifest_path = subdomain_root / "subdomain_manifest.json"
    _assert_non_empty(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(data.get("run_mode", "")).lower() != "subdomain":
        raise ValueError(f"Manifest run_mode is not 'subdomain': {data.get('run_mode')!r}")

    subdomains = data.get("subdomains") or {}
    if len(subdomains) < 3:
        raise ValueError(f"Expected at least 3 sub-domains in manifest, got {len(subdomains)}")

    for sid, meta in subdomains.items():
        status = str(meta.get("status", ""))
        if status.lower() != "success":
            raise ValueError(f"Sub-domain {sid} did not finish successfully (status={status!r})")
        run_manifest = meta.get("run_manifest")
        if not run_manifest:
            raise ValueError(f"Sub-domain {sid} missing run_manifest path in manifest")
        run_manifest_path = Path(run_manifest)
        _assert_non_empty(run_manifest_path)
        run_data = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if str(run_data.get("status", "")).lower() != "success":
            raise ValueError(f"Sub-domain {sid} run manifest status is not success: {run_data.get('status')!r}")

        project_dir = Path(meta["project_dir"])
        steps_dir = project_dir / "steps"
        if not steps_dir.is_dir():
            raise FileNotFoundError(f"Missing steps directory for {sid}: {steps_dir}")
        steps = sorted(p for p in steps_dir.glob("step_*") if p.is_dir())
        if not steps:
            raise FileNotFoundError(f"No steps found for {sid}: {steps_dir}")
        latest_step = steps[-1]

        prior_root = latest_step / "ensembles" / "prior"
        if not prior_root.is_dir():
            raise FileNotFoundError(f"Missing prior ensemble root for {sid}: {prior_root}")

        open_loop_results = prior_root / "open_loop" / "results"
        if not open_loop_results.is_dir():
            raise FileNotFoundError(f"Missing open_loop results for {sid}: {open_loop_results}")
        if not list(open_loop_results.glob("point_*.csv")):
            raise FileNotFoundError(f"No point outputs in {open_loop_results}")

        member_dirs = sorted(p for p in prior_root.glob("member_*") if p.is_dir())
        if len(member_dirs) < 3:
            raise FileNotFoundError(f"Expected at least 3 prior members for {sid}, got {len(member_dirs)}")
        for member_dir in member_dirs:
            member_results = member_dir / "results"
            if not member_results.is_dir():
                raise FileNotFoundError(f"Missing member results for {sid}: {member_results}")
            if not list(member_results.glob("point_*.csv")):
                raise FileNotFoundError(f"No point outputs in {member_results}")

    return data


def _check_project_results(subdomain_root: Path) -> None:
    project_dir = subdomain_root.parent
    results = project_dir / "results"
    grids = results / "grids"
    if not grids.is_dir():
        raise FileNotFoundError(f"Missing project results grids directory: {grids}")

    da_output = grids / "da_output_grids.nc"
    _assert_non_empty(da_output)

    _assert_non_empty(results / "subdomain_overview.csv")
    _assert_non_empty(results / "subdomain_assimilation_stats.csv")
    _assert_non_empty(results / "subdomain_assimilation_aggregate.csv")
    if (project_dir / "maps.yml").is_file():
        map_outputs = sorted((results / "maps").glob("*.png"))
        if not map_outputs:
            raise FileNotFoundError(f"Expected merged project maps under {results / 'maps'}")
        for path in map_outputs:
            _assert_non_empty(path)


def validate_subdomain_root(subdomain_root: Path, log_file: Path) -> None:
    _check_logs(log_file)
    _check_manifest(subdomain_root)
    _check_project_results(subdomain_root)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate trimmed sub-domain integration outputs and logs.")
    p.add_argument("--subdomain-root", type=Path, required=True)
    p.add_argument("--log-file", type=Path, required=True)
    args = p.parse_args()

    validate_subdomain_root(args.subdomain_root, args.log_file)
    print(f"Sub-domain integration output validation passed: {args.subdomain_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
