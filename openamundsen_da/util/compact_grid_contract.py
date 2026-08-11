"""Shared compact DA grid configuration contract."""

from __future__ import annotations

from collections.abc import Mapping


DEFAULT_SUMMARY_METRICS = (
    "open_loop",
    "ens_mean",
    "ens_std",
    "ens_min",
    "ens_max",
    "increment",
)
ANALYSIS_SUMMARY_METRICS = ("analysis_mean", "analysis_increment")
ALL_SUMMARY_METRICS = frozenset((*DEFAULT_SUMMARY_METRICS, *ANALYSIS_SUMMARY_METRICS))


def configured_compact_grid_metrics(project_cfg: Mapping[str, object]) -> dict[str, set[str]] | None:
    """Return explicitly configured compact source variables and metrics."""

    da_cfg = project_cfg.get("data_assimilation")
    da_cfg = da_cfg if isinstance(da_cfg, Mapping) else {}
    out_cfg = da_cfg.get("output")
    out_cfg = out_cfg if isinstance(out_cfg, Mapping) else {}
    grids_cfg = out_cfg.get("grids")
    grids_cfg = grids_cfg if isinstance(grids_cfg, Mapping) else {}
    variables = grids_cfg.get("variables")
    if not isinstance(variables, list) or not variables:
        return None

    configured: dict[str, set[str]] = {}
    for item in variables:
        if not isinstance(item, Mapping):
            continue
        source_name = str(item.get("var") or item.get("name") or "").strip()
        if not source_name:
            continue
        raw_metrics = item.get("metrics")
        if raw_metrics is None:
            metrics = set(ALL_SUMMARY_METRICS)
        elif isinstance(raw_metrics, (list, tuple, set)):
            metrics = {str(metric).strip() for metric in raw_metrics if str(metric).strip()}
        else:
            metric = str(raw_metrics).strip()
            metrics = {metric} if metric else set()
        configured[source_name] = metrics or set(ALL_SUMMARY_METRICS)
    return configured or None


def configured_model_grid_output_names(setup_cfg: Mapping[str, object]) -> set[str]:
    """Return openAMUNDSEN grid output names declared by the setup."""

    output_data = setup_cfg.get("output_data")
    output_data = output_data if isinstance(output_data, Mapping) else {}
    grids_cfg = output_data.get("grids")
    grids_cfg = grids_cfg if isinstance(grids_cfg, Mapping) else {}
    variables = grids_cfg.get("variables")
    if not isinstance(variables, list):
        return set()
    return {
        name
        for item in variables
        if isinstance(item, Mapping)
        if (name := str(item.get("name") or "").strip())
    }


def expected_compact_data_vars(grid_metrics: Mapping[str, set[str]]) -> set[str]:
    """Return metric-prefixed NetCDF variable names required by a contract."""

    return {
        f"{metric}_{source_name}"
        for source_name, metrics in grid_metrics.items()
        for metric in metrics
    }


def compact_grid_configuration_errors(
    *,
    setup_cfg: Mapping[str, object],
    project_cfg: Mapping[str, object],
) -> list[str]:
    """Return cross-configuration errors for explicit compact output requests."""

    grid_metrics = configured_compact_grid_metrics(project_cfg)
    if grid_metrics is None:
        return []

    errors: list[str] = []
    model_names = configured_model_grid_output_names(setup_cfg)
    missing_sources = sorted(set(grid_metrics) - model_names)
    if missing_sources:
        errors.append(
            "project.data_assimilation.output.grids.variables requests model grid "
            "source name(s) not produced by setup.output_data.grids.variables: "
            + ", ".join(missing_sources)
        )

    da_cfg = project_cfg.get("data_assimilation")
    da_cfg = da_cfg if isinstance(da_cfg, Mapping) else {}
    out_cfg = da_cfg.get("output")
    out_cfg = out_cfg if isinstance(out_cfg, Mapping) else {}
    grids_cfg = out_cfg.get("grids")
    grids_cfg = grids_cfg if isinstance(grids_cfg, Mapping) else {}
    variables = grids_cfg.get("variables")
    if isinstance(variables, list):
        for index, item in enumerate(variables):
            if not isinstance(item, Mapping):
                continue
            source_name = str(item.get("var") or item.get("name") or "").strip()
            if not source_name:
                continue
            metrics = grid_metrics.get(source_name, set())
            unknown = sorted(metrics - ALL_SUMMARY_METRICS)
            if unknown:
                errors.append(
                    "Unknown compact grid metric(s) at "
                    f"project.data_assimilation.output.grids.variables[{index}].metrics: "
                    + ", ".join(unknown)
                )
    return errors


__all__ = [
    "ALL_SUMMARY_METRICS",
    "ANALYSIS_SUMMARY_METRICS",
    "DEFAULT_SUMMARY_METRICS",
    "compact_grid_configuration_errors",
    "configured_compact_grid_metrics",
    "configured_model_grid_output_names",
    "expected_compact_data_vars",
]
