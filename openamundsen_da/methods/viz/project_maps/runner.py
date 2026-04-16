from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from loguru import logger

from openamundsen_da.io.paths import project_maps_output_dir, project_maps_root
from openamundsen_da.methods.viz.project_maps.config import (
    ProjectMapsConfig,
    default_project_maps_config_path,
    load_project_maps_config,
)
from openamundsen_da.methods.viz.project_maps.data import load_static_context
from openamundsen_da.methods.viz.project_maps.render import render_map_recipe
from openamundsen_da.util.loguru_utils import configure_cli_logger


def project_maps_enabled(project_dir: Path, config_path: Path | None = None) -> bool:
    target = Path(config_path) if config_path is not None else default_project_maps_config_path(Path(project_dir))
    return target.is_file()


def _filtered_names(config: ProjectMapsConfig, *, names: set[str] | None) -> ProjectMapsConfig:
    allowed_names = names or set(config.all_names())
    return ProjectMapsConfig(
        path=config.path,
        maps=tuple(item for item in config.maps if item.name in allowed_names),
    )


def _recipe_output_path(project_dir: Path, recipe_name: str) -> Path:
    return project_maps_output_dir(project_dir) / f"{recipe_name}.png"


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

    results_maps_root = project_maps_root(project_dir)
    for subdir_name in ("overview", "comparison", "observation_context"):
        legacy_family_dir = results_maps_root / subdir_name
        if legacy_family_dir.is_dir():
            logger.info("Removing legacy project map family directory {}", legacy_family_dir)
            shutil.rmtree(legacy_family_dir)


def render_project_maps(
    *,
    project_dir: Path,
    config_path: Path | None = None,
    names: set[str] | None = None,
) -> list[Path]:
    project_dir = Path(project_dir).resolve()
    _clean_legacy_project_map_outputs(project_dir)
    target_config = Path(config_path) if config_path is not None else default_project_maps_config_path(project_dir)
    config = load_project_maps_config(target_config)
    filtered = _filtered_names(config, names=names)
    if not filtered.maps:
        raise ValueError("Project maps selection resolved to no recipes")

    context = load_static_context(project_dir)
    outputs: list[Path] = []
    for recipe in filtered.maps:
        outputs.append(
            render_map_recipe(
                project_dir=project_dir,
                context=context,
                recipe=recipe,
                output_path=_recipe_output_path(project_dir, recipe.output_stem),
            )
        )
    return outputs


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oa-da-plot-project-maps",
        description="Render YAML-driven project maps from a completed openAMUNDSEN-DA project.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory")
    parser.add_argument("--config", type=Path, help="Override project maps config path (default: <project>/maps.yml)")
    parser.add_argument("--name", action="append", help="Restrict rendering to one or more recipe ids")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = parser.parse_args(argv)

    configure_cli_logger(args.log_level)
    try:
        outputs = render_project_maps(
            project_dir=Path(args.project_dir),
            config_path=Path(args.config) if args.config else None,
            names=set(args.name or ()),
        )
    except Exception as exc:
        logger.error("Project maps rendering failed: {}", exc)
        return 1

    for output in outputs:
        logger.info("Wrote {}", output)
    return 0
