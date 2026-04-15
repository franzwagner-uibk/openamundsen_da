from __future__ import annotations

"""
openamundsen_da.io.paths

Purpose
- Centralize filesystem helpers for discovering configs, ensembles, and rasters.
-
Key Behaviors
- Locate project/setup/step YAMLs, member meteo/results layouts, and daily rasters.
- Normalize user paths relative to the project root.
- Provide ensemble helpers (open_loop/member naming) used across core/methods modules.

Inputs/Outputs
- Functions accept `Path`-like args and return `Path` objects or structured metadata.

Assumptions
- Repository layout matches the openAMUNDSEN convention (setup/step/ensembles/member_*).
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Union

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import (
    ENSEMBLE_PRIOR,
    MEMBER_PREFIX,
    OPEN_LOOP,
    VAR_HS,
    VAR_SWE,
)

# ---- Raster/NetCDF discovery helpers ---------------------------------------

_VAR_TO_NC_NAME = {
    VAR_HS: "snowdepth_daily",
    VAR_SWE: "swe_daily",
}


@dataclass(frozen=True)
class GridSlice:
    """Descriptor for a daily grid slice in GeoTIFF or NetCDF form."""

    kind: Literal["geotiff", "netcdf"]
    path: Path
    variable: str
    date: datetime
    band: int = 1
    nc_var: str | None = None

# ---- YAML discovery helpers -------------------------------------------------

def _discover_named_yaml(
    root_dir: Path,
    *,
    preferred_name: str,
    fallback_name: str,
    allow_single_candidate: bool = False,
) -> Path:
    """
    Discover a YAML file in `root_dir` with deterministic precedence.

    Priority:
    1) `<preferred_name>.yml`
    2) `<fallback_name>.yml`
    3) exactly one `*.yml` in directory (optional)

    Always logs which file was selected.
    """
    preferred = root_dir / f"{preferred_name}.yml"
    if preferred.is_file():
        logger.debug("Resolved YAML: {}", preferred)
        return preferred

    fallback = root_dir / f"{fallback_name}.yml"
    if fallback.is_file():
        logger.debug("Resolved YAML: {}", fallback)
        return fallback

    if allow_single_candidate:
        candidates = sorted(root_dir.glob("*.yml"))
        if len(candidates) == 1:
            logger.debug("Resolved YAML: {}", candidates[0])
            return candidates[0]

    raise FileNotFoundError(f"Missing YAML in {root_dir}: expected {preferred.name} or {fallback.name}")


def find_setup_yaml(setup_dir: str | Path) -> Path:
    """
    Return setup-level YAML from a setup root directory.

    Setup YAML naming is strict-by-convention:
    - preferred: `<setup_dir.name>.yml`
    - template fallback: `setup.yml`
    """
    setup_dir = Path(setup_dir)
    if not (setup_dir / "projects").is_dir():
        raise FileNotFoundError(f"Not a setup root (missing projects/): {setup_dir}")
    return _discover_named_yaml(
        setup_dir,
        preferred_name=setup_dir.name,
        fallback_name="setup",
        allow_single_candidate=True,
    )


def find_project_yaml(project_dir: str | Path) -> Path:
    """
    Return project-level YAML from a project directory.

    Project YAML naming is strict-by-convention:
    - preferred: `<project_dir.name>.yml`
    - fallback: `project.yml`
    """
    project_dir = Path(project_dir)
    if project_dir.parent.name != "projects":
        raise FileNotFoundError(f"Not a project directory under projects/: {project_dir}")
    return _discover_named_yaml(project_dir, preferred_name=project_dir.name, fallback_name="project")


def infer_project_dir(path: str | Path) -> Path:
    """Infer a project directory by walking upward until a project YAML is found."""
    p = Path(path).resolve()
    for base in (p, *p.parents):
        try:
            _ = find_project_yaml(base)
            return base
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Could not infer project directory from {path}")


def infer_setup_dir(path: str | Path) -> Path:
    """Infer a setup directory by walking upward until a setup YAML is found."""
    p = Path(path).resolve()
    for base in (p, *p.parents):
        try:
            _ = find_setup_yaml(base)
            return base
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Could not infer setup directory from {path}")


def infer_setup_dir_from_project(project_dir: str | Path) -> Path:
    """Return the setup directory that owns a project directory."""
    return infer_setup_dir(Path(project_dir).resolve())


def projects_root(setup_dir: str | Path) -> Path:
    """Return `<setup_dir>/projects`."""
    root = Path(setup_dir) / "projects"
    if not root.is_dir():
        raise FileNotFoundError(f"Projects directory not found: {root}")
    return root


def list_project_dirs(setup_dir: str | Path) -> list[Path]:
    """List project directories under `<setup_dir>/projects`."""
    root = projects_root(setup_dir)
    return [p for p in sorted(root.iterdir()) if p.is_dir()]

def find_step_yaml(step_dir: str | Path) -> Path:
    step_dir = Path(step_dir)
    # allow flexible step file name (e.g. step_00.yml)
    ymls = sorted(step_dir.glob("*.yml")) + sorted(step_dir.glob("*.yaml"))
    if not ymls:
        raise FileNotFoundError(f"No step YAML found in {step_dir}")
    return ymls[0]


def read_step_config(step_dir: str | Path) -> dict[str, Any]:
    """Read and return the step YAML as a dict.

    Uses ruamel.yaml safe loader and raises on invalid/missing YAML.
    """
    import ruamel.yaml as _yaml

    yml = find_step_yaml(step_dir)
    y = _yaml.YAML(typ="safe")
    with Path(yml).open("r", encoding="utf-8") as f:
        loaded = y.load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Step YAML root must be a mapping: {yml}")
    return loaded

# ---- Project step discovery helpers -----------------------------------------

def steps_root(project_dir: str | Path) -> Path:
    """Return the directory that contains step_* folders for a project.

    New layout uses <project_dir>/steps and does not support top-level step_*.
    """
    project_dir = Path(project_dir)
    candidate = project_dir / "steps"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Steps directory not found: {candidate}")
    return candidate


def list_step_dirs(project_dir: str | Path) -> list[Path]:
    """List step_* directories under a project, unsorted."""
    root = steps_root(project_dir)
    return [p for p in sorted(root.glob("step_*")) if p.is_dir()]


def list_steps_sorted(project_dir: str | Path) -> list[Path]:
    """List step_* directories sorted by start_date then name."""
    items: list[tuple[datetime, Path]] = []
    for p in list_step_dirs(project_dir):
        cfg = read_step_config(p)
        sd = cfg.get("start_date")
        if sd is None or str(sd).strip() == "":
            raise ValueError(f"Missing required key 'start_date' in step YAML under {p}")
        try:
            start = datetime.fromisoformat(str(sd))
        except Exception as exc:
            raise ValueError(f"Invalid start_date in step YAML under {p}: {sd!r}") from exc
        items.append((start, p))
    items.sort(key=lambda t: (t[0], t[1].name))
    return [p for _, p in items]

# ---- Ensemble layout helpers -----------------------------------------------

def meteo_dir_for_member(member_dir: str | Path) -> Path:
    """Member meteo directory: <member_dir>/meteo"""
    return Path(member_dir) / "meteo"

def default_results_dir(member_dir: str | Path) -> Path:
    """Default outputs under <member_dir>/results"""
    return Path(member_dir) / "results"

def list_member_dirs(base_dir: str | Path, ensemble: str) -> list[Path]:
    base_dir = Path(base_dir)
    if ensemble not in {"prior", "posterior"}:
        raise ValueError("ensemble must be 'prior' or 'posterior'")

    roots = [base_dir / ensemble, base_dir / "ensembles" / ensemble]
    for root in roots:
        if root.is_dir():
            return [p for p in sorted(root.glob("member_*")) if p.is_dir()]
    return []


# ---- Per-step file discovery helpers ---------------------------------------

def list_station_files_forcing(step_dir: str | Path, ensemble: str = "prior") -> tuple[Path | None, list[str]]:
    """Return (open_loop_meteo_dir_if_any, station_filenames) for forcing.

    Prefers `<step>/ensembles/<ensemble>/open_loop/meteo/*.csv` (excluding
    stations.csv). Falls back to the first member's meteo directory.
    """
    step_dir = Path(step_dir)
    base = step_dir / "ensembles" / ensemble
    ol_meteo = base / "open_loop" / "meteo"
    if ol_meteo.is_dir():
        files = [f.name for f in sorted(ol_meteo.glob("*.csv")) if f.name.lower() != "stations.csv"]
        return ol_meteo, files
    members = list_member_dirs(step_dir / "ensembles", ensemble)
    if not members:
        return None, []
    first_meteo = members[0] / "meteo"
    files = [f.name for f in sorted(first_meteo.glob("*.csv")) if f.name.lower() != "stations.csv"]
    return None, files


def list_point_files_results(step_dir: str | Path, ensemble: str = "prior") -> tuple[Path | None, list[str]]:
    """Return (open_loop_results_dir_if_any, sorted point_*.csv files) for results."""
    step_dir = Path(step_dir)
    base = step_dir / "ensembles" / ensemble
    ol_results = base / "open_loop" / "results"
    files: list[str] = []
    if ol_results.is_dir():
        files = [f.name for f in sorted(ol_results.glob("point_*.csv"))]
    if not files:
        members = list_member_dirs(step_dir / "ensembles", ensemble)
        for member in members:
            res_dir = member / "results"
            if not res_dir.is_dir():
                continue
            files = [f.name for f in sorted(res_dir.glob("point_*.csv"))]
            if files:
                break
    return (ol_results if ol_results.is_dir() else None), files


# ---- Generic path helpers ---------------------------------------------------

def abspath_relative_to(base: str | Path, p: str | Path) -> str:
    """Return absolute path string, resolving `p` against `base` if relative."""
    base = Path(base)
    pp = Path(p)
    return str(pp if pp.is_absolute() else (base / pp))


def project_results_root(project_dir: str | Path) -> Path:
    """Return the canonical project-level results root."""
    return Path(project_dir) / "results"


def project_plots_root(project_dir: str | Path) -> Path:
    """Return the canonical project-level plots root."""
    return project_results_root(project_dir) / "plots"


def project_plot_results_dir(project_dir: str | Path) -> Path:
    """Return the canonical project-level result-plots directory."""
    return project_plots_root(project_dir) / "results"


def project_plot_assim_dir(project_dir: str | Path) -> Path:
    """Return the canonical project-level assimilation-plots directory."""
    return project_plots_root(project_dir) / "assim"


def project_plot_assim_weights_dir(project_dir: str | Path) -> Path:
    """Return the canonical project-level weights-plots directory."""
    return project_plot_assim_dir(project_dir) / "weights"


def project_plot_assim_ess_dir(project_dir: str | Path) -> Path:
    """Return the canonical project-level ESS-plots directory."""
    return project_plot_assim_dir(project_dir) / "ess"


def project_plot_assim_scores_dir(project_dir: str | Path) -> Path:
    """Return the canonical project-level benchmark score-plots directory."""
    return project_plot_assim_dir(project_dir) / "scores"


def project_plot_perf_dir(project_dir: str | Path) -> Path:
    """Return the canonical project-level performance-plots directory."""
    return project_plots_root(project_dir) / "perf"


def project_plot_points_dir(project_dir: str | Path) -> Path:
    """Return the canonical project-level points-plots directory."""
    return project_plots_root(project_dir) / "points"


def project_result_overview_output_path(project_dir: str | Path) -> Path:
    """Return the canonical default result-overview plot path."""
    return project_plot_results_dir(project_dir) / "result_overview.png"


def project_result_overview_custom_output_path(project_dir: str | Path) -> Path:
    """Return the canonical custom result-overview plot path."""
    return project_plot_results_dir(project_dir) / "result_overview_custom.png"


def project_obs_selection_plot_path(project_dir: str | Path) -> Path:
    """Return the canonical observation-selection plot path."""
    return project_plot_results_dir(project_dir) / "obs_selection.png"


def project_misc_root(project_dir: str | Path) -> Path:
    """Return the canonical project-level misc-artifacts directory."""
    return project_results_root(project_dir) / "misc"


def project_fraction_envelope_path(project_dir: str | Path, observable: str) -> Path:
    """Return the canonical fraction-envelope CSV path for one observable."""
    token = str(observable).strip().lower()
    if token == "scf":
        name = "point_scf_roi_envelope.csv"
    elif token in {"wet_snow", "wet_snow_fraction"}:
        name = "point_wet_snow_roi_envelope.csv"
    else:
        raise ValueError(f"Unsupported fraction envelope observable: {observable}")
    return project_misc_root(project_dir) / name


def project_landcover_mask_report_path(project_dir: str | Path) -> Path:
    """Return the canonical land-cover mask report path."""
    return project_misc_root(project_dir) / "lc_mask_report.csv"


def project_grids_root(project_dir: str | Path) -> Path:
    """Return the canonical project-level grids directory."""
    return project_results_root(project_dir) / "grids"


def project_da_output_grids_path(project_dir: str | Path) -> Path:
    """Return the canonical compact DA summary NetCDF path."""
    return project_grids_root(project_dir) / "da_output_grids.nc"


def project_maps_root(project_dir: str | Path) -> Path:
    """Return the canonical project-level maps directory."""
    return project_results_root(project_dir) / "maps"


def project_map_family_dir(project_dir: str | Path, family: str) -> Path:
    """Return the canonical output directory for one project-map family."""
    token = str(family).strip().lower()
    valid = {"overview", "comparison", "observation_context"}
    if token not in valid:
        raise ValueError(f"Unsupported project map family '{family}', expected one of {sorted(valid)}")
    return project_maps_root(project_dir) / token


def project_benchmark_root(project_dir: str | Path) -> Path:
    """Return the canonical project-level benchmark directory."""
    return project_results_root(project_dir) / "benchmark"


def project_benchmark_plots_dir(project_dir: str | Path) -> Path:
    """Legacy alias for the canonical project-level benchmark score-plots directory."""
    return project_plot_assim_scores_dir(project_dir)


# ---- Prior ensemble layout helpers -----------------------------------------

PathLike = Union[str, Path]

def prior_root(step_dir: PathLike) -> Path:
    """<step_dir>/ensembles/prior root directory."""
    return Path(step_dir) / "ensembles" / ENSEMBLE_PRIOR

def open_loop_dir(step_dir: PathLike) -> Path:
    """<step_dir>/ensembles/prior/open_loop directory."""
    return prior_root(step_dir) / OPEN_LOOP

def member_dir_for_index(step_dir: PathLike, index: int, width: int = 3) -> Path:
    """Member directory path using zero-padded index: member_XXX."""
    name = f"{MEMBER_PREFIX}{index:0{width}d}"
    return prior_root(step_dir) / name


# ---- Member results helpers -------------------------------------------------

def member_id_from_results_dir(results_dir: str | Path) -> str:
    """Return member ID (e.g., 'member_001') given a member results dir."""
    return Path(results_dir).parent.name


def find_member_daily_raster(results_dir: str | Path, variable: str, date_str: str) -> Path:
    """Find a daily raster for a given variable and date in a member results dir.

    Parameters
    ----------
    results_dir : Path-like
        Path to the member results directory (contains daily GeoTIFFs).
    variable : str
        One of VAR_HS ('hs') or VAR_SWE ('swe').
    date_str : str
        Date string in 'YYYY-MM-DD' format.

    Returns
    -------
    Path
        Path to the first matching raster.
    """
    results_dir = Path(results_dir)
    if variable == VAR_HS:
        prefix = "snowdepth_daily_"
    elif variable == VAR_SWE:
        prefix = "swe_daily_"
    else:
        raise ValueError(f"Unknown variable '{variable}', expected '{VAR_HS}' or '{VAR_SWE}'")
    patt = f"{prefix}{date_str}T*.tif"
    matches = sorted(results_dir.glob(patt))
    if not matches:
        raise FileNotFoundError(f"No raster found matching {patt} in {results_dir}")
    return matches[0]


def find_member_daily_grid_slice(
    results_dir: str | Path,
    variable: str,
    date_str: str,
    preferred_format: str | None = None,
) -> GridSlice:
    """
    Locate a daily gridded output for the given variable/date.

    The search order honors `preferred_format` when provided. Otherwise it
    tries GeoTIFF first for backward compatibility, then NetCDF.
    """
    results_dir = Path(results_dir)
    date = datetime.fromisoformat(date_str)

    fmt = (preferred_format or "").lower().strip() or None
    if fmt not in {None, "geotiff", "netcdf"}:
        fmt = None  # fall back to autodetect

    def _try_geotiff() -> GridSlice | None:
        try:
            tif = find_member_daily_raster(results_dir, variable, date_str)
            return GridSlice(kind="geotiff", path=tif, variable=variable, date=date, band=1)
        except FileNotFoundError:
            return None

    nc_var = _VAR_TO_NC_NAME.get(variable)
    if nc_var is None:
        raise FileNotFoundError(f"No NetCDF mapping for variable '{variable}'")

    def _try_netcdf() -> GridSlice | None:
        candidates = sorted(results_dir.glob("*.nc"))
        for nc_path in candidates:
            try:
                import xarray as xr  # lazy import
            except Exception as exc:  # pragma: no cover - defensive
                raise FileNotFoundError("xarray is required to read NetCDF outputs.") from exc

            try:
                with xr.open_dataset(nc_path) as ds:
                    if nc_var not in ds:
                        continue
                    da = ds[nc_var]
                    time_dims = [d for d in da.dims if d.startswith("time")]
                    if not time_dims:
                        continue
                    time_dim = time_dims[0]
                    times = pd.to_datetime(ds[time_dim].values)
                    # Prefer exact datetime match; otherwise fall back to calendar date match.
                    matches_dt = [i for i, t in enumerate(times) if pd.to_datetime(t) == date]
                    idx = None
                    if matches_dt:
                        idx = matches_dt[0]
                    else:
                        matches_date = [i for i, t in enumerate(times) if pd.to_datetime(t).date() == date.date()]
                        if matches_date:
                            idx = matches_date[0]
                    if idx is None:
                        continue
                    band = idx + 1  # rasterio flattens time to band (1-based)
                    return GridSlice(
                        kind="netcdf",
                        path=nc_path,
                        variable=variable,
                        date=date,
                        band=band,
                        nc_var=nc_var,
                    )
            except Exception:
                continue
        return None

    # Resolve according to preferred_format
    if fmt == "geotiff":
        tif_slice = _try_geotiff()
        if tif_slice:
            return tif_slice
        raise FileNotFoundError(f"No GeoTIFF found for variable '{variable}' and date {date_str} in {results_dir}")
    if fmt == "netcdf":
        nc_slice = _try_netcdf()
        if nc_slice:
            return nc_slice
        raise FileNotFoundError(f"No NetCDF grid found for variable '{variable}' and date {date_str} in {results_dir}")

    # fmt None -> try GeoTIFF then NetCDF
    tif_slice = _try_geotiff()
    if tif_slice:
        return tif_slice
    nc_slice = _try_netcdf()
    if nc_slice:
        return nc_slice

    raise FileNotFoundError(
        f"No GeoTIFF or NetCDF daily grid found for variable '{variable}' and date {date_str} in {results_dir}"
    )
