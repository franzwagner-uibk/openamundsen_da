from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from loguru import logger

from openamundsen_da.io.paths import project_map_family_dir
from openamundsen_da.methods.viz.project_maps.config import (
    ComparisonMapRecipe,
    ObservationContextMapRecipe,
    ProjectMapsConfig,
    default_project_maps_config_path,
    load_project_maps_config,
)
from openamundsen_da.methods.viz.project_maps.data import (
    load_model_fields,
    load_observation_scene,
    load_static_context,
    resolve_comparison_dates,
    resolve_observation_context_dates,
)
from openamundsen_da.methods.viz.project_maps.render import (
    render_comparison_map,
    render_observation_context_map,
    render_overview_map,
)
from openamundsen_da.methods.viz.project_maps.styles import require_variable_preset
from openamundsen_da.util.loguru_utils import configure_cli_logger


FAMILY_CHOICES = ("overview", "comparison", "observation_context")


def project_maps_enabled(project_dir: Path, config_path: Path | None = None) -> bool:
    target = Path(config_path) if config_path is not None else default_project_maps_config_path(Path(project_dir))
    return target.is_file()


def _filtered_names(config: ProjectMapsConfig, *, families: set[str] | None, names: set[str] | None) -> ProjectMapsConfig:
    allowed_families = set(FAMILY_CHOICES) if not families else set(families)
    allowed_names = names or set(config.all_names())
    return ProjectMapsConfig(
        path=config.path,
        overview_maps=tuple(
            item for item in config.overview_maps if "overview" in allowed_families and item.name in allowed_names
        ),
        comparison_maps=tuple(
            item for item in config.comparison_maps if "comparison" in allowed_families and item.name in allowed_names
        ),
        observation_context_maps=tuple(
            item
            for item in config.observation_context_maps
            if "observation_context" in allowed_families and item.name in allowed_names
        ),
    )


def _family_output_path(project_dir: Path, family: str, recipe_name: str, date_token: str | None = None) -> Path:
    stem = recipe_name if date_token is None else f"{recipe_name}_{date_token}"
    return project_map_family_dir(project_dir, family) / f"{stem}.png"


def _clean_legacy_project_map_outputs(project_dir: Path) -> None:
    legacy_maps_root = project_dir / "plots" / "maps"
    if legacy_maps_root.is_dir():
        logger.info("Removing legacy project map outputs under {}", legacy_maps_root)
        shutil.rmtree(legacy_maps_root)
    legacy_plots_root = project_dir / "plots"
    if legacy_plots_root.is_dir():
        try:
            legacy_plots_root.rmdir()
        except OSError:
            pass


def _resolve_title(recipe_title: str | None, *, default: str, date_token: str | None = None) -> str:
    if recipe_title:
        return recipe_title if date_token is None else f"{recipe_title} ({date_token})"
    return default if date_token is None else f"{default} ({date_token})"


def _render_overview_family(project_dir: Path, config: ProjectMapsConfig) -> list[Path]:
    if not config.overview_maps:
        return []
    context = load_static_context(project_dir)
    outputs: list[Path] = []
    for recipe in config.overview_maps:
        title = _resolve_title(recipe.title, default="Project overview")
        output_path = _family_output_path(project_dir, "overview", recipe.name)
        outputs.append(render_overview_map(context=context, title=title, output_path=output_path))
    return outputs


def _render_comparison_recipe(project_dir: Path, recipe: ComparisonMapRecipe, *, context) -> list[Path]:
    preset = require_variable_preset(recipe.variable)
    dates = resolve_comparison_dates(project_dir, recipe.variable, recipe.dates)
    fields = load_model_fields(project_dir, recipe.variable, dates)
    outputs: list[Path] = []
    for field in fields:
        date_token = field.date.strftime("%Y-%m-%d") if len(fields) > 1 else None
        output_path = _family_output_path(project_dir, "comparison", recipe.name, date_token)
        title = _resolve_title(recipe.title, default=preset.title, date_token=date_token)
        outputs.append(
            render_comparison_map(
                context=context,
                fields=field,
                all_fields=fields,
                preset=preset,
                title=title,
                output_path=output_path,
            )
        )
    return outputs


def _render_observation_context_recipe(
    project_dir: Path,
    recipe: ObservationContextMapRecipe,
    *,
    context,
) -> list[Path]:
    preset = require_variable_preset(recipe.model_variable)
    dates = resolve_observation_context_dates(
        project_dir,
        model_variable=recipe.model_variable,
        observation=recipe.observation,
        selector=recipe.dates,
    )
    fields = load_model_fields(project_dir, recipe.model_variable, dates)
    outputs: list[Path] = []
    by_date = {field.date.normalize(): field for field in fields}
    for date in dates:
        observation_scene = load_observation_scene(
            project_dir,
            context,
            observation=recipe.observation,
            date=date,
        )
        date_token = date.strftime("%Y-%m-%d") if len(dates) > 1 else None
        output_path = _family_output_path(project_dir, "observation_context", recipe.name, date_token)
        title = _resolve_title(
            recipe.title,
            default=f"{preset.title} and {recipe.observation}",
            date_token=date_token,
        )
        outputs.append(
            render_observation_context_map(
                context=context,
                fields=by_date[date.normalize()],
                all_fields=fields,
                observation_scene=observation_scene,
                preset=preset,
                title=title,
                output_path=output_path,
            )
        )
    return outputs


def render_project_maps(
    *,
    project_dir: Path,
    config_path: Path | None = None,
    families: set[str] | None = None,
    names: set[str] | None = None,
) -> list[Path]:
    project_dir = Path(project_dir).resolve()
    _clean_legacy_project_map_outputs(project_dir)
    target_config = Path(config_path) if config_path is not None else default_project_maps_config_path(project_dir)
    config = load_project_maps_config(target_config)
    filtered = _filtered_names(config, families=families, names=names)
    if not (filtered.overview_maps or filtered.comparison_maps or filtered.observation_context_maps):
        raise ValueError("Project maps selection resolved to no recipes")

    outputs: list[Path] = []
    context = None
    if filtered.overview_maps or filtered.comparison_maps or filtered.observation_context_maps:
        context = load_static_context(project_dir)

    if filtered.overview_maps:
        for recipe in filtered.overview_maps:
            title = _resolve_title(recipe.title, default="Project overview")
            output_path = _family_output_path(project_dir, "overview", recipe.name)
            outputs.append(render_overview_map(context=context, title=title, output_path=output_path))

    if filtered.comparison_maps:
        for recipe in filtered.comparison_maps:
            outputs.extend(_render_comparison_recipe(project_dir, recipe, context=context))

    if filtered.observation_context_maps:
        for recipe in filtered.observation_context_maps:
            outputs.extend(_render_observation_context_recipe(project_dir, recipe, context=context))

    return outputs


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oa-da-plot-project-maps",
        description="Render publication-style project maps from a completed openAMUNDSEN-DA project.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory")
    parser.add_argument("--config", type=Path, help="Override project maps config path (default: <project>/project_maps.yml)")
    parser.add_argument("--family", action="append", choices=FAMILY_CHOICES, help="Restrict rendering to one or more map families")
    parser.add_argument("--name", action="append", help="Restrict rendering to one or more recipe names")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = parser.parse_args(argv)

    configure_cli_logger(args.log_level)
    try:
        outputs = render_project_maps(
            project_dir=Path(args.project_dir),
            config_path=Path(args.config) if args.config else None,
            families=set(args.family or ()),
            names=set(args.name or ()),
        )
    except Exception as exc:
        logger.error("Project maps rendering failed: {}", exc)
        return 1

    for output in outputs:
        logger.info("Wrote {}", output)
    return 0
