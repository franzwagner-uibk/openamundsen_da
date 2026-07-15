"""Public observation-preprocessing operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from openamundsen_da.configuration import ProjectConfiguration, load_project_configuration
from openamundsen_da.exceptions import ObservationPreprocessingError
from openamundsen_da.manifests import (
    file_inventory,
    inventory_digest,
    load_manifest,
    workflow_manifest_path,
    write_manifest_atomic,
)
from openamundsen_da.methods.wet_snow.area import summarize_s1_directory
from openamundsen_da.observer.class_config import load_wetsnow_classes
from openamundsen_da.observer.snowcover import summarize_snowcover_directory
from openamundsen_da.results import (
    ObservationPreprocessingResult,
    ObservationProduct,
    WorkflowStatus,
)
from openamundsen_da.util.landcover_mask import resolve_landcover_mask
from openamundsen_da.util.roi_grid import ensure_setup_roi_vector
from openamundsen_da.util.ts import parse_datetime_opt


def _product_config(config: ProjectConfiguration, key: str) -> dict[str, Any]:
    obs = config.project["obs"]
    section = obs[key]
    if not isinstance(section, dict):  # defensive after strict validation
        raise ObservationPreprocessingError(f"project.obs.{key} must be a mapping")
    return section


def _setup_path(config: ProjectConfiguration, raw: object) -> Path:
    return (config.setup_dir / str(raw)).resolve()


def _observation_inputs(config: ProjectConfiguration, key: str) -> tuple[Path, list[Path]]:
    section = _product_config(config, key)
    input_dir = _setup_path(config, section["dir"])
    if not input_dir.is_dir():
        raise ObservationPreprocessingError(f"Configured observation directory not found: {input_dir}")
    configured = str(section["format"]).lower()
    geotiffs = sorted([*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")])
    netcdfs = sorted(input_dir.glob("*.nc"))
    if geotiffs and netcdfs:
        raise ObservationPreprocessingError(f"Mixed GeoTIFF and NetCDF observation artifacts in {input_dir}")
    selected = geotiffs if configured == "geotiff" else netcdfs
    if not selected:
        raise ObservationPreprocessingError(
            f"No {configured} observation files found in configured directory {input_dir}"
        )
    return input_dir, selected


def _manifest_inputs(config: ProjectConfiguration, selected: list[Path]) -> tuple[list[dict[str, Any]], str]:
    roi = ensure_setup_roi_vector(config.setup_dir)
    files = [config.setup_yaml, config.project_yaml, roi, *selected]
    inventory = file_inventory(root=config.setup_dir, files=files)
    return inventory, inventory_digest(inventory)


def _reuse_or_start(
    *,
    manifest_path: Path,
    summary_path: Path,
    digest: str,
    overwrite: bool,
) -> WorkflowStatus | None:
    existing = load_manifest(manifest_path)
    if existing is not None and existing.get("status") == "success":
        if existing.get("input_digest") == digest and summary_path.is_file() and not overwrite:
            return WorkflowStatus.REUSED
        if not overwrite:
            raise ObservationPreprocessingError(
                f"Observation inputs differ from completed manifest {manifest_path}; rerun with overwrite=True"
            )
    elif summary_path.is_file() and not overwrite:
        raise ObservationPreprocessingError(
            f"Observation summary exists without a matching completed manifest: {summary_path}; "
            "rerun with overwrite=True"
        )
    return None


def _source_count(summary_path: Path) -> int:
    try:
        frame = pd.read_csv(summary_path, usecols=["source"])
    except (OSError, ValueError):
        return 0
    sources: set[str] = set()
    for raw in frame["source"].dropna().astype(str):
        sources.update(item for item in raw.split(";") if item)
    return len(sources)


def _diagnostics(summary_path: Path, product: ObservationProduct) -> tuple[Path, ...]:
    token = "scf" if product is ObservationProduct.SNOW_COVER else "wet_snow"
    return tuple(
        path.resolve()
        for path in sorted(summary_path.parent.rglob("*"))
        if path.is_file()
        and path.resolve() != summary_path.resolve()
        and token in path.name.lower()
    )


def _result(
    *,
    config: ProjectConfiguration,
    product: ObservationProduct,
    status: WorkflowStatus,
    summary_path: Path,
    selected_count: int,
    manifest_path: Path,
) -> ObservationPreprocessingResult:
    processed = _source_count(summary_path)
    return ObservationPreprocessingResult(
        project_dir=config.project_dir,
        product=product,
        status=status,
        summary_path=summary_path.resolve(),
        diagnostic_paths=_diagnostics(summary_path, product),
        processed_count=processed,
        rejected_count=max(0, selected_count - processed),
        manifest_path=manifest_path.resolve(),
    )


def preprocess_snow_cover(
    project_dir: str | Path,
    *,
    overwrite: bool = False,
) -> ObservationPreprocessingResult:
    """Summarize configured snow-cover inputs without rewriting YAML."""
    config = load_project_configuration(project_dir)
    if str(config.project["run_mode"]).lower() != "single":
        raise ObservationPreprocessingError(
            "preprocess_snow_cover supports run_mode: single; subdomain observations are prepared by "
            "the subdomains command tree"
        )
    section = _product_config(config, "snowcover")
    input_dir, selected = _observation_inputs(config, "snowcover")
    summary_path = _setup_path(config, section["summary_csv"])
    manifest_path = workflow_manifest_path(config.project_dir, "observations-snow-cover")
    inventory, digest = _manifest_inputs(config, selected)
    reused = _reuse_or_start(
        manifest_path=manifest_path,
        summary_path=summary_path,
        digest=digest,
        overwrite=overwrite,
    )
    if reused is not None:
        return _result(
            config=config,
            product=ObservationProduct.SNOW_COVER,
            status=reused,
            summary_path=summary_path,
            selected_count=len(selected),
            manifest_path=manifest_path,
        )
    manifest = {
        "operation": "preprocess-snow-cover",
        "status": "running",
        "input_digest": digest,
        "inputs": inventory,
    }
    write_manifest_atomic(manifest_path, manifest)
    try:
        written = summarize_snowcover_directory(
            setup_dir=config.setup_dir,
            input_dir=input_dir,
            aoi=ensure_setup_roi_vector(config.setup_dir),
            project_label=config.project_dir.name,
            output_root=summary_path.parent.parent,
            recursive=False,
            start=parse_datetime_opt(str(config.project["start_date"])),
            end=parse_datetime_opt(str(config.project["end_date"])),
        )
        expected = summary_path.resolve()
        generated = summary_path.parent.parent / config.project_dir.name / "scf_summary.csv"
        if generated.resolve() != expected or not expected.is_file():
            raise ObservationPreprocessingError(f"Snow-cover preprocessing did not create {expected}")
        manifest.update(
            {
                "status": "success",
                "outputs": [expected.relative_to(config.setup_dir).as_posix()],
                "processed_count": len(written),
            }
        )
        write_manifest_atomic(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error": str(exc),
            }
        )
        write_manifest_atomic(manifest_path, manifest)
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, ObservationPreprocessingError):
            raise
        raise ObservationPreprocessingError(f"Snow-cover preprocessing failed: {exc}") from exc
    return _result(
        config=config,
        product=ObservationProduct.SNOW_COVER,
        status=WorkflowStatus.COMPLETED,
        summary_path=summary_path,
        selected_count=len(selected),
        manifest_path=manifest_path,
    )


def preprocess_wet_snow(
    project_dir: str | Path,
    *,
    overwrite: bool = False,
) -> ObservationPreprocessingResult:
    """Summarize configured wet-snow inputs without rewriting YAML."""
    config = load_project_configuration(project_dir)
    if str(config.project["run_mode"]).lower() != "single":
        raise ObservationPreprocessingError(
            "preprocess_wet_snow supports run_mode: single; subdomain observations are prepared by "
            "the subdomains command tree"
        )
    section = _product_config(config, "wetsnow")
    input_dir, selected = _observation_inputs(config, "wetsnow")
    summary_path = _setup_path(config, section["summary_csv"])
    diagnostics_raw = section.get("wet_snow_line_diagnostics_csv")
    diagnostics_path = (
        _setup_path(config, diagnostics_raw)
        if diagnostics_raw is not None
        else summary_path.parent / "wet_snow_line_diagnostics.csv"
    )
    manifest_path = workflow_manifest_path(config.project_dir, "observations-wet-snow")
    inventory, digest = _manifest_inputs(config, selected)
    reused = _reuse_or_start(
        manifest_path=manifest_path,
        summary_path=summary_path,
        digest=digest,
        overwrite=overwrite,
    )
    if reused is not None:
        return _result(
            config=config,
            product=ObservationProduct.WET_SNOW,
            status=reused,
            summary_path=summary_path,
            selected_count=len(selected),
            manifest_path=manifest_path,
        )
    manifest = {
        "operation": "preprocess-wet-snow",
        "status": "running",
        "input_digest": digest,
        "inputs": inventory,
    }
    write_manifest_atomic(manifest_path, manifest)
    try:
        wet, valid, exclude = load_wetsnow_classes(config.project_dir)
        summarize_s1_directory(
            setup_dir=config.setup_dir,
            project_dir=config.project_dir,
            raster_dir=input_dir,
            aoi_path=ensure_setup_roi_vector(config.setup_dir),
            output_csv=summary_path,
            landcover_cfg=resolve_landcover_mask(config.setup_dir, config.project_dir),
            overwrite=True,
            start=parse_datetime_opt(str(config.project["start_date"])),
            end=parse_datetime_opt(str(config.project["end_date"])),
            wet_values=wet,
            valid_values=valid,
            exclude_values=exclude,
            recursive=False,
            wsl_diagnostics_csv=diagnostics_path,
            wsl_profile_dir=summary_path.parent / "wet_snow_line_profiles",
        )
        if not summary_path.is_file():
            raise ObservationPreprocessingError(f"Wet-snow preprocessing did not create {summary_path}")
        outputs = [summary_path, *_diagnostics(summary_path, ObservationProduct.WET_SNOW)]
        manifest.update(
            {
                "status": "success",
                "outputs": [path.relative_to(config.setup_dir).as_posix() for path in outputs],
                "processed_count": _source_count(summary_path),
            }
        )
        write_manifest_atomic(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error": str(exc),
            }
        )
        write_manifest_atomic(manifest_path, manifest)
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, ObservationPreprocessingError):
            raise
        raise ObservationPreprocessingError(f"Wet-snow preprocessing failed: {exc}") from exc
    return _result(
        config=config,
        product=ObservationProduct.WET_SNOW,
        status=WorkflowStatus.COMPLETED,
        summary_path=summary_path,
        selected_count=len(selected),
        manifest_path=manifest_path,
    )


__all__ = ["preprocess_snow_cover", "preprocess_wet_snow"]
