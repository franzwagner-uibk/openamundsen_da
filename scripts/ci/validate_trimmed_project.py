from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
import yaml

from openamundsen_da.util.storage_policy import DA_SUMMARY_NC_FILL_VALUE, METEO_CSV_DECIMALS
from openamundsen_da.manifests import file_inventory, inventory_digest, load_manifest
from openamundsen_da.util.da_observables import (
    station_diagnostics_glob_pattern,
    weights_glob_pattern,
)
from openamundsen_da.util.station_da import station_observation_csvs


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

# Known benign warnings for the example-project CI setup.
# These indicate optional observation overlays are absent, not a failed run.
BENIGN_WARNING_PATTERNS = [
    re.compile(r"SCF obs not found .* plotting without obs points", re.IGNORECASE),
    re.compile(r"Wet-snow obs not found .* plotting without obs points", re.IGNORECASE),
    re.compile(r"No member series found for point_wet_snow_roi\.csv", re.IGNORECASE),
    re.compile(r"No data for station point_scf_roi\.csv across setup; skipping\.", re.IGNORECASE),
    re.compile(r"Missing liquid water grids for .*", re.IGNORECASE),
    re.compile(
        r"Skipping analysis benchmark for .* on \d{4}-\d{2}-\d{2}: missing observation row",
        re.IGNORECASE,
    ),
    re.compile(
        r"Skipping wet_snow_line benchmark case at \d{4}-\d{2}-\d{2}: missing model values",
        re.IGNORECASE,
    ),
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
            if any(p.search(line) for p in SEVERE_WARNING_PATTERNS) and not any(
                p.search(line) for p in BENIGN_WARNING_PATTERNS
            ):
                severe_warning_lines.append(line)

    if fatal_lines:
        sample = "\n".join(fatal_lines[:20])
        raise ValueError(f"Integration log contains fatal error lines:\n{sample}")
    if severe_warning_lines:
        sample = "\n".join(severe_warning_lines[:20])
        raise ValueError(f"Integration log contains severe warning lines:\n{sample}")


def _check_plot_outputs(project_dir: Path) -> None:
    plot_specs = [
        CheckSpec(
            label="step forcing plots",
            patterns=("steps/step_*/plots/forcing/**/*.png", "steps/step_*/plots/forcing/**/*.svg"),
        ),
        CheckSpec(
            label="project overview plots",
            patterns=("results/plots/results/**/*.png", "results/plots/results/**/*.svg"),
        ),
        CheckSpec(
            label="project point plots",
            patterns=("results/plots/points/**/*.png", "results/plots/points/**/*.svg"),
        ),
        CheckSpec(
            label="assimilation weights plots",
            patterns=(
                "results/plots/assim/weights/**/*.png",
                "results/plots/assim/weights/**/*.svg",
                "steps/step_*/assim/weights_*.png",
                "steps/step_*/assim/weights_*.svg",
            ),
        ),
        CheckSpec(
            label="assimilation ESS plots",
            patterns=("results/plots/assim/ess/**/*.png", "results/plots/assim/ess/**/*.svg"),
        ),
        CheckSpec(
            label="benchmark plots",
            patterns=("results/plots/assim/scores/performance_scores.png", "results/plots/assim/scores/performance_scores.svg"),
            min_count=1,
        ),
    ]
    if (project_dir / "maps.yml").is_file():
        plot_specs.append(
            CheckSpec(
                label="project maps",
                patterns=("results/maps/*.png", "results/maps/*.svg"),
                min_count=1,
            )
        )

    missing: list[str] = []
    for spec in plot_specs:
        found = _collect_non_empty(project_dir, spec.patterns)
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

    # openAMUNDSEN outputs that remain after compact retention:
    # - station point outputs (CSV)
    for results_dir in member_result_dirs + open_loop_result_dirs:
        point_csvs = _collect_non_empty(results_dir, ("point_*.csv",))
        if not point_csvs:
            raise FileNotFoundError(f"{results_dir}: no point_*.csv outputs found")


def _check_da_output_grid(project_dir: Path) -> None:
    da_output = project_dir / "results" / "grids" / "da_output_grids.nc"
    _assert_non_empty(da_output)
    import numpy as np
    import xarray as xr

    expected_scales = {
        "snowdepth_daily": np.float32(0.001),
        "swe_daily": np.float32(1.0),
        "liquid_water_content": np.float32(1.0),
    }
    with xr.open_dataset(da_output, decode_cf=False) as raw:
        grid_vars = [name for name, da in raw.data_vars.items() if "y" in da.dims and "x" in da.dims]
        if not grid_vars:
            raise ValueError(f"{da_output} contains no grid payload variables")
        for name in grid_vars:
            var = raw[name]
            matched_scale = None
            for suffix, scale in expected_scales.items():
                if name == suffix or name.endswith(f"_{suffix}"):
                    matched_scale = scale
                    break
            if matched_scale is None:
                if var.dtype == np.dtype("float64"):
                    raise ValueError(f"{da_output}:{name} is float64; expected compact storage")
                continue
            if var.dtype != np.dtype("int16"):
                raise ValueError(f"{da_output}:{name} stored as {var.dtype}; expected int16")
            if var.attrs.get("_FillValue") != DA_SUMMARY_NC_FILL_VALUE:
                raise ValueError(f"{da_output}:{name} has unexpected _FillValue {var.attrs.get('_FillValue')!r}")
            scale_factor = var.attrs.get("scale_factor")
            if scale_factor is None or abs(float(scale_factor) - float(matched_scale)) > 1e-12:
                raise ValueError(f"{da_output}:{name} has unexpected scale_factor {var.attrs.get('scale_factor')!r}")
            if var.encoding.get("zlib") is not True or var.encoding.get("shuffle") is not True:
                raise ValueError(f"{da_output}:{name} is missing zlib/shuffle compression")


def _check_member_grid_storage(steps_dir: Path) -> None:
    import numpy as np
    import xarray as xr

    member_grids = sorted(steps_dir.glob("step_*/ensembles/prior/member_*/results/grids/output_grids.nc"))
    if not member_grids:
        member_grids = sorted(steps_dir.glob("step_*/ensembles/prior/member_*/results/output_grids.nc"))
    if not member_grids:
        raise FileNotFoundError("No member output_grids.nc found for storage validation")

    with xr.open_dataset(member_grids[0], decode_cf=False) as raw:
        grid_vars = [name for name, da in raw.data_vars.items() if "y" in da.dims and "x" in da.dims]
        if not grid_vars:
            raise ValueError(f"{member_grids[0]} contains no grid payload variables")
        for name in grid_vars:
            dtype = raw[name].dtype
            if dtype != np.dtype("float32"):
                raise ValueError(f"{member_grids[0]}:{name} stored as {dtype}; expected float32")


def _check_meteo_csv_precision(steps_dir: Path) -> None:
    meteo_files = sorted(p for p in steps_dir.glob("step_*/ensembles/prior/member_*/meteo/*.csv") if p.name != "stations.csv")
    if not meteo_files:
        raise FileNotFoundError("No generated member meteo CSV files found for precision validation")
    rows = _read_csv_rows(meteo_files[0])
    if not rows:
        raise ValueError(f"Generated meteo CSV has no rows: {meteo_files[0]}")
    row = rows[0]
    for column, decimals in METEO_CSV_DECIMALS.items():
        value = row.get(column)
        if value in (None, ""):
            continue
        pattern = r"^-?\d+$" if decimals == 0 else rf"^-?\d+\.\d{{{decimals}}}$"
        if re.match(pattern, value) is None:
            raise ValueError(
                f"{meteo_files[0]}:{column} value {value!r} does not match expected {decimals}-decimal storage precision"
            )


def _check_wet_snow_mask_storage(steps_dir: Path) -> None:
    import rasterio

    masks = sorted(steps_dir.glob("step_*/ensembles/prior/member_*/results/**/wet_snow_mask_*.tif"))
    if not masks:
        return
    with rasterio.open(masks[0]) as src:
        if src.dtypes[0] != "uint8":
            raise ValueError(f"{masks[0]} stored as {src.dtypes[0]}; expected uint8")
        if src.nodata != 255:
            raise ValueError(f"{masks[0]} has nodata={src.nodata}; expected 255")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    _assert_non_empty(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _find_project_yaml(project_dir: Path) -> Path:
    direct = project_dir / f"{project_dir.name}.yml"
    if direct.is_file():
        return direct
    matches = sorted(project_dir.glob("*.yml"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing project YAML under {project_dir}")


def _assimilation_event_counts(project_dir: Path) -> dict[str, int]:
    project_yaml = _find_project_yaml(project_dir)
    with project_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    da_cfg = cfg.get("data_assimilation") or {}
    events = da_cfg.get("assimilation_events") or []
    counts: dict[str, int] = {}
    for event in events:
        variable = str((event or {}).get("variable", "")).strip().lower()
        if not variable:
            continue
        counts[variable] = counts.get(variable, 0) + 1
    return counts


def _require_files(root: Path, *, label: str, patterns: tuple[str, ...], min_count: int) -> None:
    found = _collect_non_empty(root, patterns)
    if len(found) < min_count:
        raise FileNotFoundError(f"{label}: expected >= {min_count}, found {len(found)}")


def _check_station_obs_inputs(project_dir: Path) -> None:
    stations_dir = project_dir.parent.parent / "obs" / "stations"
    if not stations_dir.is_dir():
        raise FileNotFoundError(f"Missing station obs directory: {stations_dir}")
    station_csvs = station_observation_csvs(stations_dir)
    if not station_csvs:
        raise FileNotFoundError(f"No station observation CSVs found under {stations_dir}")
    for path in station_csvs:
        _assert_non_empty(path)


def _check_required_outputs(project_dir: Path, steps_dir: Path) -> None:
    step_dirs = sorted(steps_dir.glob("step_*"))
    if not step_dirs:
        raise FileNotFoundError(f"No step directories found under {steps_dir}")

    event_counts = _assimilation_event_counts(project_dir)
    if not event_counts:
        raise ValueError("Project YAML defines no assimilation_events for example-project CI validation")

    if event_counts.get("scf", 0) > 0:
        _require_files(
            steps_dir,
            label="SCF obs files (obs_scf_*.csv)",
            patterns=("step_*/obs/obs_scf_*.csv",),
            min_count=event_counts["scf"],
        )
        _require_files(
            steps_dir,
            label="SCF weights files (weights_scf_*.csv)",
            patterns=(f"step_*/assim/{weights_glob_pattern('scf')}",),
            min_count=event_counts["scf"],
        )
        _require_files(
            steps_dir,
            label="SCF model time series (point_scf_roi.csv)",
            patterns=("step_*/ensembles/prior/member_*/results/point_scf_roi.csv",),
            min_count=1,
        )

    if event_counts.get("wet_snow", 0) > 0:
        _require_files(
            steps_dir,
            label="Wet-snow obs files (obs_wet_snow_*.csv)",
            patterns=("step_*/obs/obs_wet_snow_*.csv",),
            min_count=event_counts["wet_snow"],
        )
        _require_files(
            steps_dir,
            label="Wet-snow weights files (weights_wet_snow_*.csv)",
            patterns=(f"step_*/assim/{weights_glob_pattern('wet_snow')}",),
            min_count=event_counts["wet_snow"],
        )
        _require_files(
            steps_dir,
            label="Wet-snow model time series (point_wet_snow_roi.csv)",
            patterns=("step_*/ensembles/prior/member_*/results/point_wet_snow_roi.csv",),
            min_count=1,
        )

    if event_counts.get("station_hs", 0) > 0:
        _check_station_obs_inputs(project_dir)
        _require_files(
            steps_dir,
            label="Station HS weights files (weights_station_hs_*.csv)",
            patterns=(f"step_*/assim/{weights_glob_pattern('station_hs')}",),
            min_count=event_counts["station_hs"],
        )
        _require_files(
            steps_dir,
            label="Station HS diagnostics (station_diagnostics_station_hs_*.csv)",
            patterns=(f"step_*/assim/{station_diagnostics_glob_pattern('station_hs')}",),
            min_count=event_counts["station_hs"],
        )

    if event_counts.get("station_swe", 0) > 0:
        _check_station_obs_inputs(project_dir)
        _require_files(
            steps_dir,
            label="Station SWE weights files (weights_station_swe_*.csv)",
            patterns=(f"step_*/assim/{weights_glob_pattern('station_swe')}",),
            min_count=event_counts["station_swe"],
        )
        _require_files(
            steps_dir,
            label="Station SWE diagnostics (station_diagnostics_station_swe_*.csv)",
            patterns=(f"step_*/assim/{station_diagnostics_glob_pattern('station_swe')}",),
            min_count=event_counts["station_swe"],
        )

    _require_files(
        steps_dir,
        label="ROI SWE model time series (point_swe_roi.csv)",
        patterns=("step_*/ensembles/prior/member_*/results/point_swe_roi.csv",),
        min_count=1,
    )
    _require_files(
        steps_dir,
        label="ROI snow-depth model time series (point_snow_depth_roi.csv)",
        patterns=("step_*/ensembles/prior/member_*/results/point_snow_depth_roi.csv",),
        min_count=1,
    )


def _check_minimal_weight_sanity(steps_dir: Path) -> None:
    weights_files = sorted(steps_dir.glob("step_*/assim/weights_*_*.csv"))
    for wf in weights_files:
        weights = _read_weights(wf)
        if not weights:
            raise ValueError(f"{wf} has no rows")
        s = sum(weights)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"{wf} weights do not sum to 1.0 (sum={s})")


def _check_benchmark_outputs(project_dir: Path) -> None:
    results_dir = project_dir / "results" / "benchmark"
    manifest_path = results_dir / "manifest.json"
    summary_path = results_dir / "summary.md"
    continuous_path = results_dir / "cases" / "continuous_case_scores.csv"
    analysis_path = results_dir / "cases" / "analysis_case_scores.csv"
    event_path = results_dir / "scores" / "event_scores.csv"
    project_path = results_dir / "scores" / "project_scores.csv"
    reliability_path = results_dir / "scores" / "project_reliability.csv"
    project_summary_path = results_dir / "tables" / "project_summary.csv"
    update_summary_path = results_dir / "tables" / "update_summary.csv"
    stale_md_paths = (
        results_dir / "tables" / "project_summary.md",
        results_dir / "tables" / "update_summary.md",
        results_dir / "tables" / "project_summary_wide.md",
        results_dir / "tables" / "event_summary_wide.md",
        results_dir / "tables" / "reliability_summary_wide.md",
        results_dir / "tables" / "improvement_summary.md",
    )
    stale_csv_paths = (
        results_dir / "tables" / "project_summary_wide.csv",
        results_dir / "tables" / "event_summary_wide.csv",
        results_dir / "tables" / "reliability_summary_wide.csv",
        results_dir / "tables" / "improvement_summary.csv",
    )
    stale_plot_dirs = (
        results_dir / "plots",
        project_dir / "plots" / "benchmark" / "core",
        project_dir / "plots" / "benchmark" / "extended",
        project_dir / "plots" / "benchmark",
    )

    for path in (
        manifest_path,
        summary_path,
        continuous_path,
        analysis_path,
        event_path,
        project_path,
        reliability_path,
        project_summary_path,
        update_summary_path,
    ):
        _assert_non_empty(path)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if int(manifest.get("case_rows", 0)) <= 0:
        raise ValueError(f"Benchmark manifest reports no case rows: {manifest_path}")
    if int(manifest.get("project_rows", 0)) <= 0:
        raise ValueError(f"Benchmark manifest reports no project rows: {manifest_path}")
    headline_plot_outputs = [
        value
        for value in manifest.get("outputs", {}).values()
        if str(value).replace("\\", "/").endswith("/results/plots/assim/scores/performance_scores.png")
    ]
    if len(headline_plot_outputs) != 1:
        raise ValueError(f"Benchmark manifest is missing the shipped performance_scores plot: {manifest_path}")

    continuous_rows = _read_csv_rows(continuous_path)
    analysis_rows = _read_csv_rows(analysis_path)
    project_rows = _read_csv_rows(project_path)
    update_rows = _read_csv_rows(update_summary_path)

    if not any(row.get("score_set") == "continuous" for row in continuous_rows):
        raise ValueError(f"Benchmark continuous case table contains no continuous rows: {continuous_path}")
    if not any(row.get("score_set") == "analysis" for row in analysis_rows):
        raise ValueError(f"Benchmark analysis case table contains no analysis rows: {analysis_path}")
    if not project_rows:
        raise ValueError(f"Benchmark project scores CSV has no rows: {project_path}")
    if not any(
        row.get("variable") == "station_swe" and row.get("stream") == "semi_independent"
        for row in project_rows
    ):
        raise ValueError(
            "Benchmark project scores are missing the shipped semi_independent station_swe benchmark view"
        )
    if update_rows and "stream" not in update_rows[0]:
        raise ValueError(f"Benchmark update summary is missing the required stream column: {update_summary_path}")
    for path in stale_csv_paths:
        if path.exists():
            raise ValueError(f"Stale benchmark CSV should not be written anymore: {path}")
    for path in stale_md_paths:
        if path.exists():
            raise ValueError(f"Stale benchmark markdown table should not be written anymore: {path}")
    for path in stale_plot_dirs:
        if path.exists():
            raise ValueError(f"Stale benchmark plot directory should not be written anymore: {path}")
    if (project_dir / "plots").exists():
        raise ValueError(f"Legacy project-level plots root should not be written anymore: {project_dir / 'plots'}")


def _check_run_manifest_and_cleanup(project_dir: Path) -> None:
    manifest_path = project_dir / "results" / "run_manifest.json"
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    if manifest.get("status") != "success":
        raise ValueError(f"Run manifest is not successful: {manifest_path}")
    if manifest.get("stages") != {"execution": "success", "render": "success", "cleanup": "success"}:
        raise ValueError(f"Run manifest stages are incomplete: {manifest.get('stages')}")

    cleanup = manifest.get("cleanup") or {}
    deleted_paths = cleanup.get("deleted_paths") or []
    if int(cleanup.get("deleted_count", 0)) != len(deleted_paths) or not deleted_paths:
        raise ValueError(f"Run manifest does not record applied restart cleanup: {manifest_path}")
    if int(cleanup.get("freed_bytes", 0)) <= 0:
        raise ValueError(f"Run manifest cleanup did not record freed bytes: {manifest_path}")
    for relative in deleted_paths:
        if (project_dir / relative).exists():
            raise ValueError(f"Cleaned restart artifact still exists: {project_dir / relative}")

    remaining_states = sorted((project_dir / "steps").glob("step_*/ensembles/*/*/results/model_state.pickle.gz"))
    remaining_pointers = sorted((project_dir / "steps").glob("step_*/ensembles/*/*/state_pointer.json"))
    if remaining_states or remaining_pointers:
        raise ValueError(
            "Successful run retained package-owned restart artifacts: "
            f"states={len(remaining_states)} pointers={len(remaining_pointers)}"
        )

    outputs = manifest.get("outputs") or []
    output_paths = [project_dir.parent.parent / entry["path"] for entry in outputs]
    current = file_inventory(root=project_dir.parent.parent, files=output_paths)
    if inventory_digest(current) != manifest.get("output_digest"):
        raise ValueError(f"Run output inventory does not match its manifest: {manifest_path}")
    for relative in manifest.get("performance_outputs") or []:
        _assert_non_empty(project_dir / relative)


def validate_project(project_dir: Path, log_file: Path) -> None:
    steps_dir = project_dir / "steps"
    if not steps_dir.is_dir():
        raise FileNotFoundError(f"Missing steps directory: {steps_dir}")
    _check_logs(log_file)
    _check_required_outputs(project_dir, steps_dir)
    _check_plot_outputs(project_dir)
    _check_openamundsen_outputs(steps_dir)
    _check_da_output_grid(project_dir)
    _check_member_grid_storage(steps_dir)
    _check_meteo_csv_precision(steps_dir)
    _check_wet_snow_mask_storage(steps_dir)
    _check_benchmark_outputs(project_dir)
    _check_minimal_weight_sanity(steps_dir)
    _check_run_manifest_and_cleanup(project_dir)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate example-project integration outputs and logs.")
    p.add_argument("--project-dir", type=Path, required=True)
    p.add_argument("--log-file", type=Path, required=True)
    args = p.parse_args()

    validate_project(args.project_dir, args.log_file)
    print(f"Integration output validation passed: {args.project_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
