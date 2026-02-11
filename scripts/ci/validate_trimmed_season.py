from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


ERROR_PATTERNS = [
    re.compile(r"\bERROR\b"),
    re.compile(r"\bCRITICAL\b"),
    re.compile(r"Traceback"),
    re.compile(r"\bException\b"),
]

# Warning lines are not automatically fatal.
# Only warnings matching these phrases are treated as severe.
SEVERE_WARNING_PATTERNS = [
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"\babort", re.IGNORECASE),
    re.compile(r"\bmissing\b", re.IGNORECASE),
    re.compile(r"\bnot found\b", re.IGNORECASE),
    re.compile(r"excludes entire ROI", re.IGNORECASE),
]


def _read_weights(path: Path) -> list[float]:
    weights: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "weight" not in (reader.fieldnames or []):
            raise ValueError(f"{path} missing required 'weight' column")
        for row in reader:
            try:
                w = float(row["weight"])
            except Exception as exc:
                raise ValueError(f"{path} has non-numeric weight value: {row.get('weight')}") from exc
            if w < 0.0 or w > 1.0:
                raise ValueError(f"{path} has out-of-range weight {w} (expected 0..1)")
            weights.append(w)
    return weights


def _assert_non_empty(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing expected file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Expected non-empty file but got empty file: {path}")


@dataclass(frozen=True)
class CheckSpec:
    label: str
    patterns: tuple[str, ...]
    min_count: int = 1


def _collect_non_empty(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for p in root.glob(pattern):
            if p.is_file() and p not in seen:
                _assert_non_empty(p)
                files.append(p)
                seen.add(p)
    return sorted(files)


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
            if any(p.search(line) for p in SEVERE_WARNING_PATTERNS):
                severe_warning_lines.append(line)

    if fatal_lines:
        sample = "\n".join(fatal_lines[:20])
        raise ValueError(f"Integration log contains fatal error lines:\n{sample}")
    if severe_warning_lines:
        sample = "\n".join(severe_warning_lines[:20])
        raise ValueError(f"Integration log contains severe warning lines:\n{sample}")


def _check_plot_outputs(season_dir: Path) -> None:
    plot_specs = [
        CheckSpec(
            label="season forcing plots",
            patterns=("plots/forcing/**/*.png", "plots/forcing/**/*.svg"),
        ),
        CheckSpec(
            label="season results plots",
            patterns=("plots/results/**/*.png", "plots/results/**/*.svg"),
        ),
        CheckSpec(
            label="assimilation weights plots",
            patterns=("plots/assim/weights/**/*.png", "plots/assim/weights/**/*.svg"),
        ),
        CheckSpec(
            label="assimilation ESS plots",
            patterns=("plots/assim/ess/**/*.png", "plots/assim/ess/**/*.svg"),
        ),
    ]

    missing: list[str] = []
    for spec in plot_specs:
        found = _collect_non_empty(season_dir, spec.patterns)
        if len(found) < spec.min_count:
            missing.append(f"{spec.label}: expected >= {spec.min_count}, found {len(found)}")

    if missing:
        raise FileNotFoundError("Plot output checks failed:\n- " + "\n- ".join(missing))


def _check_openamundsen_outputs(steps_dir: Path) -> None:
    member_result_dirs = sorted(steps_dir.glob("step_*/ensembles/prior/member_*/results"))
    open_loop_result_dirs = sorted(steps_dir.glob("step_*/ensembles/prior/open_loop/results"))
    if not member_result_dirs:
        raise FileNotFoundError("No member results directories found under steps/*/ensembles/prior/member_*/results")
    if not open_loop_result_dirs:
        raise FileNotFoundError("No open_loop results directories found under steps/*/ensembles/prior/open_loop/results")

    # openAMUNDSEN outputs that remain after cleanup:
    # - station point outputs (CSV)
    # - gridded outputs (netCDF)
    for results_dir in member_result_dirs + open_loop_result_dirs:
        point_csvs = _collect_non_empty(results_dir, ("point_*.csv",))
        grid_ncs = _collect_non_empty(results_dir, ("*.nc",))
        if not point_csvs:
            raise FileNotFoundError(f"{results_dir}: no point_*.csv outputs found")
        if not grid_ncs:
            raise FileNotFoundError(f"{results_dir}: no netCDF grid outputs (*.nc) found")


def _check_required_outputs(steps_dir: Path) -> None:
    step_dirs = sorted(steps_dir.glob("step_*"))
    if not step_dirs:
        raise FileNotFoundError(f"No step directories found under {steps_dir}")

    obs_files = sorted(steps_dir.glob("step_*/obs/obs_scf_*.csv"))
    if not obs_files:
        raise FileNotFoundError("No per-step SCF obs files found (obs_scf_*.csv)")
    for p in obs_files:
        _assert_non_empty(p)

    weights_files = sorted(steps_dir.glob("step_*/assim/weights_scf_*.csv"))
    if not weights_files:
        raise FileNotFoundError("No SCF weights files found (weights_scf_*.csv)")
    for p in weights_files:
        _assert_non_empty(p)

    point_scf_files = sorted(steps_dir.glob("step_*/ensembles/prior/member_*/results/point_scf_roi.csv"))
    if not point_scf_files:
        raise FileNotFoundError("No model SCF time series found (point_scf_roi.csv)")
    for p in point_scf_files:
        _assert_non_empty(p)


def _check_minimal_weight_sanity(steps_dir: Path) -> None:
    weights_files = sorted(steps_dir.glob("step_*/assim/weights_scf_*.csv"))
    for wf in weights_files:
        weights = _read_weights(wf)
        if not weights:
            raise ValueError(f"{wf} has no rows")
        s = sum(weights)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"{wf} weights do not sum to 1.0 (sum={s})")


def validate_season(season_dir: Path, log_file: Path) -> None:
    steps_dir = season_dir / "steps"
    if not steps_dir.is_dir():
        raise FileNotFoundError(f"Missing steps directory: {steps_dir}")
    _check_logs(log_file)
    _check_required_outputs(steps_dir)
    _check_plot_outputs(season_dir)
    _check_openamundsen_outputs(steps_dir)
    _check_minimal_weight_sanity(steps_dir)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate trimmed season integration outputs and logs.")
    p.add_argument("--season-dir", type=Path, required=True)
    p.add_argument("--log-file", type=Path, required=True)
    args = p.parse_args()

    validate_season(args.season_dir, args.log_file)
    print(f"Integration output validation passed: {args.season_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
