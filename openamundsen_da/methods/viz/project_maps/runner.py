from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from loguru import logger

from openamundsen_da.io.paths import project_maps_output_dir, project_maps_root
from openamundsen_da.methods.viz.project_maps.config import (
    MapRecipe,
    ProjectMapsConfig,
    default_project_maps_config_path,
    load_project_maps_config,
)
from openamundsen_da.methods.viz.project_maps.data import load_static_context
from openamundsen_da.methods.viz.project_maps.render import RenderRuntimeCache, render_map_recipe
from openamundsen_da.util.loguru_utils import configure_cli_logger


@dataclass(frozen=True)
class RecipeRenderResult:
    recipe_name: str
    output_path: Path


class ProjectMapRenderError(RuntimeError):
    def __init__(self, recipe_name: str, message: str):
        self.recipe_name = recipe_name
        super().__init__(f"map '{recipe_name}' failed: {message}")


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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _resolve_effective_max_workers(max_workers: int | None, *, recipe_count: int) -> int:
    if recipe_count < 1:
        raise ValueError("recipe_count must be >= 1")
    available_cpus = os.cpu_count() or 1
    requested = available_cpus if max_workers is None else int(max_workers)
    return max(1, min(recipe_count, requested))


def _require_recipe(config: ProjectMapsConfig, recipe_name: str) -> MapRecipe:
    for recipe in config.maps:
        if recipe.name == recipe_name:
            return recipe
    raise KeyError(f"Project map recipe '{recipe_name}' not found in {config.path}")


@lru_cache(maxsize=16)
def _load_project_maps_config_cached(config_path_str: str) -> ProjectMapsConfig:
    return load_project_maps_config(Path(config_path_str))


def _render_recipe_with_cache(
    *,
    project_dir: Path,
    recipe: MapRecipe,
    context,
    runtime_cache: RenderRuntimeCache | None = None,
) -> RecipeRenderResult:
    output_path = _recipe_output_path(project_dir, recipe.output_stem)
    rendered_output = render_map_recipe(
        project_dir=project_dir,
        context=context,
        recipe=recipe,
        output_path=output_path,
        runtime_cache=runtime_cache,
    )
    return RecipeRenderResult(recipe_name=recipe.name, output_path=rendered_output)


def _render_recipe_worker(project_dir: Path, config_path: Path, recipe_name: str) -> RecipeRenderResult:
    project_dir = Path(project_dir).resolve()
    config = _load_project_maps_config_cached(str(Path(config_path).resolve()))
    recipe = _require_recipe(config, recipe_name)
    context = load_static_context(project_dir)
    return _render_recipe_with_cache(
        project_dir=project_dir,
        recipe=recipe,
        context=context,
    )


def _render_project_maps_sequential(project_dir: Path, recipes: tuple[MapRecipe, ...]) -> list[Path]:
    context = load_static_context(project_dir)
    runtime_cache = RenderRuntimeCache()
    outputs: list[Path] = []
    for recipe in recipes:
        logger.info("Starting map {}", recipe.name)
        try:
            result = _render_recipe_with_cache(
                project_dir=project_dir,
                recipe=recipe,
                context=context,
                runtime_cache=runtime_cache,
            )
        except Exception as exc:
            logger.error("Failed map {}: {}", recipe.name, exc)
            raise ProjectMapRenderError(recipe.name, str(exc)) from exc
        logger.info("Finished map {} -> {}", result.recipe_name, result.output_path)
        outputs.append(result.output_path)
    return outputs


def _render_project_maps_parallel(project_dir: Path, config_path: Path, recipes: tuple[MapRecipe, ...], *, max_workers: int) -> list[Path]:
    output_by_name: dict[str, Path] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_by_recipe = {}
        for recipe in recipes:
            logger.info("Starting map {}", recipe.name)
            future = executor.submit(_render_recipe_worker, project_dir, config_path, recipe.name)
            future_by_recipe[future] = recipe
        try:
            for future in as_completed(future_by_recipe):
                recipe = future_by_recipe[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Failed map {}: {}", recipe.name, exc)
                    raise ProjectMapRenderError(recipe.name, str(exc)) from exc
                logger.info("Finished map {} -> {}", result.recipe_name, result.output_path)
                output_by_name[result.recipe_name] = result.output_path
        except Exception:
            for future in future_by_recipe:
                future.cancel()
            raise
    return [output_by_name[recipe.name] for recipe in recipes]


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
    max_workers: int | None = None,
) -> list[Path]:
    project_dir = Path(project_dir).resolve()
    _clean_legacy_project_map_outputs(project_dir)
    target_config = (Path(config_path) if config_path is not None else default_project_maps_config_path(project_dir)).resolve()
    config = load_project_maps_config(target_config)
    filtered = _filtered_names(config, names=names)
    if not filtered.maps:
        raise ValueError("Project maps selection resolved to no recipes")
    effective_workers = _resolve_effective_max_workers(max_workers, recipe_count=len(filtered.maps))
    logger.info(
        "Rendering {} project map(s) from {} with {} worker(s) ...",
        len(filtered.maps),
        target_config,
        effective_workers,
    )
    if effective_workers == 1:
        return _render_project_maps_sequential(project_dir, filtered.maps)
    return _render_project_maps_parallel(project_dir, target_config, filtered.maps, max_workers=effective_workers)


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oa-da-plot-project-maps",
        description="Render YAML-driven project maps from a completed openAMUNDSEN-DA project.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory")
    parser.add_argument("--config", type=Path, help="Override project maps config path (default: <project>/maps.yml)")
    parser.add_argument("--name", action="append", help="Restrict rendering to one or more recipe ids")
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        help="Maximum recipe render workers (default: auto, clamped to visible CPUs and selected recipe count)",
    )
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = parser.parse_args(argv)

    configure_cli_logger(args.log_level)
    try:
        outputs = render_project_maps(
            project_dir=Path(args.project_dir),
            config_path=Path(args.config) if args.config else None,
            names=set(args.name or ()),
            max_workers=args.max_workers,
        )
    except Exception as exc:
        logger.error("Project maps rendering failed: {}", exc)
        return 1

    logger.info("Project maps rendering complete -> {} output(s)", len(outputs))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
