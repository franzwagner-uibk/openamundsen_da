"""Durable compact-leaf finalization for subdomain execution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from openamundsen_da.io.paths import (
    project_ensemble_forcing_path,
    project_ensemble_points_path,
    project_map_support_path,
    project_plots_maps_collection_pdf_path,
)
from openamundsen_da.manifests import (
    file_inventory,
    inventory_digest,
    load_manifest,
    recursive_files,
    write_manifest_atomic,
)
from openamundsen_da.methods.viz.maps.panel_renderers import (
    project_da_map_support_fields,
)
from openamundsen_da.pipeline.cleanup import clean_project_artifacts
from openamundsen_da.pipeline.rendering import validate_render_completion
from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.da_observables import weights_csv_name
from openamundsen_da.util.da_output import (
    output_retention_mode,
    validate_compact_output_file,
)
from openamundsen_da.util.map_support import validate_map_support
from openamundsen_da.util.retention import validate_retained_consumers


LEAF_FINALIZATION_MANIFEST = "leaf_finalization_manifest.json"
LEAF_FINALIZATION_CONTRACT = "compact-leaf-v1"


def leaf_finalization_manifest_path(setup_dir: Path) -> Path:
    """Return the durable compact-leaf acceptance manifest path."""
    return Path(setup_dir) / LEAF_FINALIZATION_MANIFEST


def _required_map_support_fields(events) -> set[str]:
    variables = {event.variable for event in events}
    fields: set[str] = set()
    if "scf" in variables:
        fields.update(
            {
                "scf_open_loop_binary",
                "scf_prior_probability",
                "scf_posterior_probability",
            }
        )
    if variables & {"wet_snow", "wet_snow_line"}:
        fields.update(
            {
                "wet_snow_open_loop",
                "wet_snow_prior_probability",
                "wet_snow_posterior_probability",
            }
        )
    return fields


def _leaf_parent_support_files(subdomain: SubdomainMeta) -> tuple[Path, ...]:
    """Return the retained files required for parent merge/render and analysis."""
    project_dir = Path(subdomain.project_dir).resolve()
    events = load_assimilation_events(project_dir)
    required = [
        project_dir / "results" / "grids" / "da_output_grids.nc",
        project_plots_maps_collection_pdf_path(project_dir),
        validate_render_completion(project_dir),
    ]
    if output_retention_mode(project_dir) == "compact":
        required.extend(
            [
                project_ensemble_points_path(project_dir),
                project_ensemble_forcing_path(project_dir),
            ]
        )
    map_fields = _required_map_support_fields(events)
    if map_fields:
        if output_retention_mode(project_dir) == "compact":
            support = project_map_support_path(project_dir)
            validate_map_support(
                project_dir,
                dates=[event.date for event in events],
                fields=map_fields,
                input_nc=support,
            )
            required.append(support)
        else:
            rebuilt = project_da_map_support_fields(project_dir)
            if rebuilt is None:
                raise RuntimeError(
                    "Full-retention leaf cannot rebuild configured DA-event map support"
                )
            dates, fields, _roi_mask = rebuilt
            missing_fields = sorted(map_fields - set(fields))
            event_dates = {event.date for event in events}
            rebuilt_dates = {date.date() for date in dates}
            missing_dates = sorted(event_dates - rebuilt_dates)
            if missing_fields or missing_dates:
                raise RuntimeError(
                    "Full-retention leaf raw DA-event map support is incomplete: "
                    f"missing fields={missing_fields}, missing dates={missing_dates}"
                )
    for event in events:
        name = weights_csv_name(
            event.variable,
            datetime.combine(event.date, datetime.min.time()),
        )
        matches = sorted((project_dir / "steps").glob(f"step_*/assim/{name}"))
        if len(matches) != 1:
            raise RuntimeError(
                "Leaf parent support requires exactly one weights artifact for "
                f"{event.date.isoformat()} {event.variable}; found {len(matches)}"
            )
        required.append(matches[0])
    missing = [path for path in required if not path.is_file() or path.is_symlink()]
    if missing:
        raise FileNotFoundError(f"Required retained leaf support is missing: {missing[0]}")
    validate_compact_output_file(
        project_dir=project_dir,
        output_nc=project_dir / "results" / "grids" / "da_output_grids.nc",
    )
    return tuple(dict.fromkeys(Path(path).resolve() for path in required))


def _validate_leaf_inventory(setup_dir: Path, inventory: list[dict]) -> None:
    paths = [Path(setup_dir) / str(row.get("path", "")) for row in inventory]
    actual = file_inventory(root=setup_dir, files=paths)
    if actual != inventory or inventory_digest(actual) != inventory_digest(inventory):
        raise RuntimeError(f"Retained leaf support changed after finalization: {setup_dir}")


def measure_leaf_bytes(setup_dir: Path) -> int:
    """Measure retained regular files without following shared-data symlinks."""
    return sum(path.stat().st_size for path in recursive_files(Path(setup_dir)))


def finalize_leaf(subdomain: SubdomainMeta, *, resume: bool = False) -> dict:
    """Validate, compact-clean and durably accept one successful leaf."""
    setup_dir = Path(subdomain.setup_dir).resolve()
    project_dir = Path(subdomain.project_dir).resolve()
    retention = output_retention_mode(project_dir)
    if retention != "compact":
        _leaf_parent_support_files(subdomain)
        return {
            "status": "success",
            "retention": retention,
            "cleanup_deleted_files": 0,
            "cleanup_freed_bytes": 0,
            "retained_leaf_bytes": measure_leaf_bytes(setup_dir),
        }
    manifest_path = leaf_finalization_manifest_path(setup_dir)
    existing = load_manifest(manifest_path) if manifest_path.is_file() else None
    if resume and existing is not None:
        if existing.get("contract") != LEAF_FINALIZATION_CONTRACT:
            raise RuntimeError(f"Unsupported leaf finalization contract: {manifest_path}")
        if Path(str(existing.get("project_dir", ""))).resolve() != project_dir:
            raise RuntimeError(f"Leaf finalization project identity changed: {manifest_path}")
        inventory = list(existing.get("retained_support") or [])
        _validate_leaf_inventory(setup_dir, inventory)
        if existing.get("status") == "success":
            validate_retained_consumers(project_dir, require_complete=True)
            _leaf_parent_support_files(subdomain)
            return existing

    support = _leaf_parent_support_files(subdomain)
    retained_inventory = file_inventory(root=setup_dir, files=support)
    planned = {
        "contract": LEAF_FINALIZATION_CONTRACT,
        "status": "planned",
        "project_dir": str(project_dir),
        "retention": retention,
        "retained_support": retained_inventory,
        "retained_support_sha256": inventory_digest(retained_inventory),
    }
    write_manifest_atomic(manifest_path, planned)

    cleanup = clean_project_artifacts(project_dir, apply=True)
    if cleanup.failures:
        raise RuntimeError(
            f"Compact leaf cleanup failed for {subdomain.id}: "
            f"{len(cleanup.failures)} artifact(s)"
        )
    _validate_leaf_inventory(setup_dir, retained_inventory)
    validate_retained_consumers(project_dir, require_complete=True)
    _leaf_parent_support_files(subdomain)
    completed = {
        **planned,
        "status": "success",
        "cleanup_deleted_files": len(cleanup.deleted_paths),
        "cleanup_freed_bytes": int(cleanup.freed_bytes),
        "retained_leaf_bytes": measure_leaf_bytes(setup_dir),
    }
    write_manifest_atomic(manifest_path, completed)
    return load_manifest(manifest_path) or completed


def measured_retained_leaf_bytes(manifest: SubdomainManifest) -> int:
    """Return completed leaf bytes already represented in filesystem usage."""
    total = 0
    for subdomain in manifest.subdomains.values():
        finalization = leaf_finalization_manifest_path(subdomain.setup_dir)
        if not finalization.is_file():
            continue
        try:
            data = load_manifest(finalization) or {}
        except ValueError:
            continue
        if data.get("status") != "success":
            continue
        total += measure_leaf_bytes(subdomain.setup_dir)
    return total


__all__ = [
    "finalize_leaf",
    "leaf_finalization_manifest_path",
    "measure_leaf_bytes",
    "measured_retained_leaf_bytes",
]
