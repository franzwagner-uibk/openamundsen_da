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

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Union

from openamundsen_da.core.constants import (
    ENSEMBLE_PRIOR,
    MEMBER_SOURCE_POINTER,
    MEMBER_PREFIX,
    OPEN_LOOP,
)

# ---- Raster/NetCDF discovery helpers ---------------------------------------

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

def _canonical_yaml(root_dir: Path, *, kind: str) -> Path:
    """Return the one canonical YAML path for a setup or project directory."""
    root_dir = Path(root_dir)
    canonical = root_dir / f"{root_dir.name}.yml"
    legacy_name = "setup.yml" if kind == "setup" else "project.yml"
    if canonical.name == legacy_name:
        raise FileNotFoundError(
            f"Directory name {root_dir.name!r} would require removed legacy alias {legacy_name}; "
            f"use a descriptive {kind} directory name"
        )
    if canonical.is_file():
        return canonical

    legacy = root_dir / legacy_name
    hint = f"; rename legacy alias {legacy.name} to {canonical.name}" if legacy.is_file() else ""
    raise FileNotFoundError(f"Missing canonical {kind} YAML {canonical}{hint}")


def find_setup_yaml(setup_dir: str | Path) -> Path:
    """Return the one unambiguous canonical setup YAML from a setup root.

    The directory name may change at a container mount boundary (for example,
    ``rofental`` mounted as ``/data``), so identity is established by requiring
    exactly one non-legacy root ``.yml`` file rather than comparing basenames.
    """
    setup_dir = Path(setup_dir)
    if not (setup_dir / "projects").is_dir():
        raise FileNotFoundError(f"Not a setup root (missing projects/): {setup_dir}")
    return find_plain_setup_yaml(setup_dir)


def find_plain_setup_yaml(setup_dir: str | Path) -> Path:
    """Return one canonical setup YAML without requiring a DA projects tree."""
    setup_dir = Path(setup_dir)
    legacy = setup_dir / "setup.yml"
    candidates = sorted(
        path
        for path in setup_dir.glob("*.yml")
        if path.is_file() and path.name != legacy.name
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(f"Ambiguous setup YAMLs in {setup_dir}: {names}")
    if legacy.is_file():
        raise FileNotFoundError(
            f"Missing canonical setup YAML in {setup_dir}; rename legacy alias setup.yml to <setup-name>.yml"
        )
    raise FileNotFoundError(f"Missing canonical setup YAML in {setup_dir}: expected one <setup-name>.yml")


def find_project_yaml(project_dir: str | Path) -> Path:
    """Return the required ``<project-name>.yml`` project configuration."""
    project_dir = Path(project_dir)
    if project_dir.parent.name != "projects":
        raise FileNotFoundError(f"Not a project directory under projects/: {project_dir}")
    return _canonical_yaml(project_dir, kind="project")


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


def resolve_member_source_dir(member_dir: str | Path) -> Path:
    """Resolve a PF posterior member pointer back to its source member directory."""
    member_dir = Path(member_dir)
    ptr = member_dir / MEMBER_SOURCE_POINTER
    if not ptr.is_file():
        return member_dir

    try:
        data = json.loads(ptr.read_text(encoding="utf-8")) or {}
    except Exception:
        return member_dir

    raw_target = data.get("member_dir")
    if not raw_target:
        return member_dir

    target = Path(str(raw_target))
    if not target.is_absolute():
        target = (member_dir / target).resolve()
    if target.is_dir():
        return target

    if not target.is_absolute():
        return member_dir

    project_dir = None
    for parent in member_dir.parents:
        try:
            _ = find_project_yaml(parent)
            project_dir = parent
            break
        except Exception:
            continue
    if project_dir is None:
        return member_dir

    try:
        parts = list(target.parts)
        if project_dir.name not in parts:
            return member_dir
        idx = parts.index(project_dir.name)
        remapped = project_dir.joinpath(*parts[idx + 1 :])
    except Exception:
        return member_dir
    return remapped if remapped.is_dir() else member_dir


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


def project_paper_root(project_dir: str | Path) -> Path:
    """Return the canonical project-level paper-output root."""
    return project_results_root(project_dir) / "paper"


def project_poster_root(project_dir: str | Path) -> Path:
    """Return the canonical project-level poster-output root."""
    return project_results_root(project_dir) / "poster"


def project_paper_output_path(project_dir: str | Path, output_path: str | Path) -> Path:
    """Mirror a project results output under ``results/paper``.

    Examples
    --------
    ``results/maps/da_events/da_6.png`` becomes
    ``results/paper/maps/da_events/da_6.png``.
    """
    project_dir = Path(project_dir)
    output_path = Path(output_path)
    results_root = project_results_root(project_dir)
    try:
        relative = output_path.relative_to(results_root)
    except ValueError:
        relative = output_path.name
    return project_paper_root(project_dir) / relative


def project_poster_output_path(project_dir: str | Path, output_path: str | Path) -> Path:
    """Mirror a project results output under ``results/poster``.

    If ``output_path`` already points inside ``results/poster``, it is returned
    unchanged so poster renderers can be safely composed without nesting.
    """
    project_dir = Path(project_dir)
    output_path = Path(output_path)
    poster_root = project_poster_root(project_dir)
    try:
        output_path.relative_to(poster_root)
    except ValueError:
        pass
    else:
        return output_path

    results_root = project_results_root(project_dir)
    try:
        relative = output_path.relative_to(results_root)
    except ValueError:
        relative = output_path.name
    return poster_root / relative


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
    elif token == "wet_snow_line":
        name = "point_wet_snow_line_roi_envelope.csv"
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


def project_maps_output_dir(project_dir: str | Path) -> Path:
    """Return the canonical flat output directory for project maps."""
    return project_maps_root(project_dir)


def project_reports_root(project_dir: str | Path) -> Path:
    """Return the canonical project-level reports directory."""
    return project_results_root(project_dir) / "reports"


def project_plots_maps_collection_pdf_path(project_dir: str | Path) -> Path:
    """Return the canonical project plots/maps collection PDF path."""
    return project_reports_root(project_dir) / "project_report.pdf"


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


def find_member_daily_grid_slice(
    results_dir: str | Path,
    variable: str,
    date_str: str,
    preferred_format: str | None = None,
) -> GridSlice:
    """Resolve a daily grid through an explicitly selected format adapter."""
    if preferred_format is None:
        raise ValueError("preferred_format is required; cross-format model-grid discovery is not supported")
    from openamundsen_da.io.model_grids import resolve_model_grid_slice

    return resolve_model_grid_slice(
        results_dir=results_dir,
        variable=variable,
        date=datetime.fromisoformat(date_str),
        grid_format=preferred_format,
    )
