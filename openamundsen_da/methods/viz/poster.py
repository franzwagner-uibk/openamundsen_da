from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

from loguru import logger
import ruamel.yaml

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    project_maps_output_dir,
    project_poster_output_path,
    project_poster_root,
    project_result_overview_custom_output_path,
)
from openamundsen_da.methods.viz.maps.config import (
    LayoutSpec,
    MapPanelSpec,
    MapRecipe,
    default_project_maps_config_path,
    load_project_maps_config,
)
from openamundsen_da.methods.viz.maps.data import load_static_context
from openamundsen_da.methods.viz.maps.generated import GENERATED_DA_MAPS_SUBDIR, generated_da_map_recipes
from openamundsen_da.methods.viz.maps.render import RenderRuntimeCache, render_map_recipe
from openamundsen_da.methods.viz.maps.runner import (
    ProjectMapRenderError,
    _collect_shared_model_vmax,
    _paper_recipe,
    _resolve_effective_max_workers,
)
from openamundsen_da.methods.viz.common import PosterLinework, PosterRenderStyle, PosterTypography
from openamundsen_da.methods.viz.plots.result_overview import cli_main as plot_result_overview_cli
from openamundsen_da.util.loguru_utils import configure_cli_logger


_yaml = ruamel.yaml.YAML()
_yaml.default_flow_style = False

_SVG_NS = "{http://www.w3.org/2000/svg}"
_XLINK_NS = "{http://www.w3.org/1999/xlink}"
_POSTER_DA_EVENT_TITLE_OVERRIDES = {
    "Prior elevation band WSF": "Prior elev. band WSF",
    "Posterior elevation band WSF": "Post. elev. band WSF",
    "Observed elevation band WSF": "Obs. elev. band WSF",
    "Prior snow cover probability": "Prior snow-cover prob.",
    "Posterior snow cover probability": "Post. snow-cover prob.",
    "Satellite FSC observation": "Satellite FSC obs.",
}


@dataclass(frozen=True)
class PosterTargetSize:
    width_mm: float
    height_mm: float

    @property
    def inches(self) -> tuple[float, float]:
        return (self.width_mm / 25.4, self.height_mm / 25.4)


@dataclass(frozen=True)
class PosterThemeConfig:
    scale: float = 1.0
    typography: PosterTypography | None = None
    linework: PosterLinework | None = None

    def render_style(self) -> PosterRenderStyle:
        return PosterRenderStyle(scale=self.scale, typography=self.typography, linework=self.linework)


@dataclass(frozen=True)
class PosterSetupOverviewConfig:
    enabled: bool = False
    name: str = "setup_overview"
    keep_panel_kinds: tuple[str, ...] = ()
    drop_panel_kinds: tuple[str, ...] = ()
    ncols: int | None = None
    target_size: PosterTargetSize | None = None


@dataclass(frozen=True)
class PosterDaEventsConfig:
    enabled: bool = False
    drop_first_column: bool = True
    names: tuple[str, ...] = ()
    target_size: PosterTargetSize | None = None


@dataclass(frozen=True)
class PosterResultOverviewConfig:
    enabled: bool = False
    panels: tuple[dict[str, object], ...] = ()
    target_size: PosterTargetSize | None = None
    h_pad: float | None = None
    hspace: float | None = None
    panel_height_factor: float = 1.0
    align_first_xtick_left: bool = False


@dataclass(frozen=True)
class PosterConfig:
    path: Path
    theme: PosterThemeConfig
    setup_overview: PosterSetupOverviewConfig
    da_events: PosterDaEventsConfig
    result_overview_custom: PosterResultOverviewConfig


def default_project_poster_config_path(project_dir: Path) -> Path:
    return Path(project_dir) / "poster.yml"


def poster_profile_enabled(project_dir: Path, config_path: Path | None = None) -> bool:
    path = Path(config_path) if config_path is not None else default_project_poster_config_path(Path(project_dir))
    return path.is_file()


def default_project_poster_rerun_command(project_dir: Path, *, config_path: Path | None = None) -> str:
    parts = [
        "oa-da-plot-poster",
        "--project-dir",
        str(Path(project_dir).resolve()),
    ]
    if config_path is not None:
        parts.extend(["--config", str(Path(config_path).resolve())])
    return " ".join(parts)


def _require_mapping(value: object, *, context: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_list(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _enabled(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"enabled must be boolean-like, got {value!r}")


def _str_tuple(value: object, *, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    items = _require_list(value, context=context)
    parsed = tuple(str(item).strip() for item in items)
    if any(not item for item in parsed):
        raise ValueError(f"{context} must contain non-empty strings")
    return parsed


def _positive_int_or_none(value: object, *, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{context} must be >= 1")
    return parsed


def _positive_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a positive number") from exc
    if parsed <= 0.0:
        raise ValueError(f"{context} must be > 0")
    return parsed


def _non_negative_float_or_none(value: object, *, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a non-negative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a non-negative number") from exc
    if parsed < 0.0:
        raise ValueError(f"{context} must be >= 0")
    return parsed


def _target_size_or_none(value: object, *, context: str) -> PosterTargetSize | None:
    if value is None:
        return None
    items = _require_list(value, context=context)
    if len(items) != 2:
        raise ValueError(f"{context} must contain width and height in millimeters")
    try:
        width_mm = float(items[0])
        height_mm = float(items[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain numeric width and height in millimeters") from exc
    if width_mm <= 0.0 or height_mm <= 0.0:
        raise ValueError(f"{context} values must be > 0")
    return PosterTargetSize(width_mm=width_mm, height_mm=height_mm)


def _poster_typography_or_none(value: object, *, context: str) -> PosterTypography | None:
    if value is None:
        return None
    cfg = _require_mapping(value, context=context)
    return PosterTypography(
        title_pt=_positive_float(cfg.get("title_pt"), context=f"{context}.title_pt"),
        label_pt=_positive_float(cfg.get("label_pt"), context=f"{context}.label_pt"),
        support_pt=_positive_float(cfg.get("support_pt"), context=f"{context}.support_pt"),
    )


def _poster_linework_or_none(value: object, *, context: str) -> PosterLinework | None:
    if value is None:
        return None
    cfg = _require_mapping(value, context=context)
    return PosterLinework(
        panel_box_pt=_positive_float(cfg.get("panel_box_pt"), context=f"{context}.panel_box_pt"),
    )


def load_poster_config(config_path: Path) -> PosterConfig:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Poster config not found: {config_path}")
    cfg = _require_mapping(_read_yaml_file(config_path), context="Poster config")
    maps = _require_mapping(cfg.get("maps"), context="Poster config maps")
    plots = _require_mapping(cfg.get("plots"), context="Poster config plots")
    theme_raw = _require_mapping(cfg.get("theme"), context="Poster config theme")
    theme = PosterThemeConfig(
        scale=_positive_float(theme_raw.get("scale", 1.0), context="Poster theme.scale"),
        typography=_poster_typography_or_none(theme_raw.get("typography"), context="Poster theme.typography"),
        linework=_poster_linework_or_none(theme_raw.get("linework"), context="Poster theme.linework"),
    )

    setup_raw = _require_mapping(maps.get("setup_overview"), context="Poster setup_overview")
    setup_layout_raw = _require_mapping(setup_raw.get("layout"), context="Poster setup_overview.layout")
    da_raw = _require_mapping(maps.get("da_events"), context="Poster da_events")
    overview_raw = _require_mapping(plots.get("result_overview_custom"), context="Poster result_overview_custom")
    overview_layout_raw = _require_mapping(
        overview_raw.get("layout"),
        context="Poster result_overview_custom.layout",
    )

    setup = PosterSetupOverviewConfig(
        enabled=bool(setup_raw) and _enabled(setup_raw.get("enabled")),
        name=str(setup_raw.get("name") or "setup_overview").strip(),
        keep_panel_kinds=_str_tuple(
            setup_raw.get("keep_panel_kinds"),
            context="Poster setup_overview.keep_panel_kinds",
        ),
        drop_panel_kinds=_str_tuple(
            setup_raw.get("drop_panel_kinds"),
            context="Poster setup_overview.drop_panel_kinds",
        ),
        ncols=_positive_int_or_none(setup_layout_raw.get("ncols"), context="Poster setup_overview.layout.ncols"),
        target_size=_target_size_or_none(
            setup_raw.get("target_size_mm"),
            context="Poster setup_overview.target_size_mm",
        ),
    )
    if setup.enabled and not setup.name:
        raise ValueError("Poster setup_overview.name must be non-empty")

    da_events = PosterDaEventsConfig(
        enabled=bool(da_raw) and _enabled(da_raw.get("enabled")),
        drop_first_column=_enabled(da_raw.get("drop_first_column")) if "drop_first_column" in da_raw else True,
        names=_str_tuple(da_raw.get("names"), context="Poster da_events.names"),
        target_size=_target_size_or_none(da_raw.get("target_size_mm"), context="Poster da_events.target_size_mm"),
    )

    panels: tuple[dict[str, object], ...] = ()
    if overview_raw:
        raw_panels = _require_list(overview_raw.get("panels"), context="Poster result_overview_custom.panels")
        panels = tuple(
            _require_mapping(panel, context="Poster result_overview_custom.panels[]")
            for panel in raw_panels
        )
        if not panels:
            raise ValueError("Poster result_overview_custom.panels must not be empty")
    result_overview_custom = PosterResultOverviewConfig(
        enabled=bool(overview_raw) and _enabled(overview_raw.get("enabled")),
        panels=panels,
        target_size=_target_size_or_none(
            overview_raw.get("target_size_mm"),
            context="Poster result_overview_custom.target_size_mm",
        ),
        h_pad=_non_negative_float_or_none(
            overview_layout_raw.get("h_pad"),
            context="Poster result_overview_custom.layout.h_pad",
        ),
        hspace=_non_negative_float_or_none(
            overview_layout_raw.get("hspace"),
            context="Poster result_overview_custom.layout.hspace",
        ),
        panel_height_factor=_positive_float(
            overview_layout_raw.get("panel_height_factor", 1.0),
            context="Poster result_overview_custom.layout.panel_height_factor",
        ),
        align_first_xtick_left=_enabled(overview_layout_raw.get("align_first_xtick_left"))
        if "align_first_xtick_left" in overview_layout_raw
        else False,
    )

    if not any((setup.enabled, da_events.enabled, result_overview_custom.enabled)):
        raise ValueError(f"Poster config enables no outputs: {config_path}")

    return PosterConfig(
        path=config_path,
        theme=theme,
        setup_overview=setup,
        da_events=da_events,
        result_overview_custom=result_overview_custom,
    )


def _trim_layout_columns(recipe: MapRecipe, *, keep_cols: tuple[int, ...]) -> LayoutSpec:
    layout = recipe.layout
    width_ratios = ()
    if len(layout.width_ratios) == layout.ncols:
        width_ratios = tuple(layout.width_ratios[col] for col in keep_cols)
    return replace(layout, ncols=len(keep_cols), width_ratios=width_ratios)


def _drop_columns(recipe: MapRecipe, drop_cols: set[int]) -> MapRecipe:
    keep_cols = tuple(col for col in range(recipe.layout.ncols) if col not in drop_cols)
    if not keep_cols:
        raise ValueError(f"Cannot drop all columns from map recipe {recipe.name}")
    col_mapping = {old_col: new_col for new_col, old_col in enumerate(keep_cols)}
    panels: list[MapPanelSpec] = []
    for panel in recipe.panels:
        col = int(panel.col)
        if col in drop_cols:
            continue
        if int(panel.colspan) != 1:
            raise ValueError(f"Poster column dropping does not support colspan panels in {recipe.name}")
        panels.append(replace(panel, col=col_mapping[col]))
    return replace(
        recipe,
        layout=_trim_layout_columns(recipe, keep_cols=keep_cols),
        panels=tuple(panels),
    )


def _drop_panel_kinds(recipe: MapRecipe, kinds: tuple[str, ...]) -> MapRecipe:
    if not kinds:
        return recipe
    drop_kinds = set(kinds)
    drop_cols = {int(panel.col) for panel in recipe.panels if panel.kind in drop_kinds}
    if not drop_cols:
        return recipe
    return _drop_columns(recipe, drop_cols)


def _keep_panel_kinds(recipe: MapRecipe, kinds: tuple[str, ...]) -> MapRecipe:
    if not kinds:
        return recipe
    panels: list[MapPanelSpec] = []
    for kind in kinds:
        matched = tuple(panel for panel in recipe.panels if panel.kind == kind)
        if not matched:
            raise ValueError(f"Poster setup_overview panel kind not found: {kind}")
        panels.extend(matched)
    return replace(recipe, panels=tuple(panels))


def _reflow_recipe_panels(recipe: MapRecipe, *, ncols: int | None) -> MapRecipe:
    if ncols is None:
        return recipe
    if not recipe.panels:
        raise ValueError(f"Cannot reflow map recipe without panels: {recipe.name}")
    panels: list[MapPanelSpec] = []
    for idx, panel in enumerate(recipe.panels):
        if int(panel.rowspan) != 1 or int(panel.colspan) != 1:
            raise ValueError(f"Poster panel reflow does not support spanned panels in {recipe.name}")
        panels.append(replace(panel, row=idx // ncols, col=idx % ncols))
    return replace(
        recipe,
        layout=LayoutSpec(nrows=int(math.ceil(len(panels) / ncols)), ncols=ncols),
        panels=tuple(panels),
        row_labels=(),
        row_views=(),
    )


def poster_setup_overview_recipe(recipe: MapRecipe, config: PosterSetupOverviewConfig) -> MapRecipe:
    poster_recipe = _paper_recipe(recipe)
    poster_recipe = _keep_panel_kinds(poster_recipe, config.keep_panel_kinds)
    poster_recipe = _drop_panel_kinds(poster_recipe, config.drop_panel_kinds)
    return _reflow_recipe_panels(poster_recipe, ncols=config.ncols)


def poster_da_event_recipe(recipe: MapRecipe, config: PosterDaEventsConfig) -> MapRecipe:
    poster_recipe = _paper_recipe(recipe)
    if config.drop_first_column:
        poster_recipe = _drop_columns(poster_recipe, {0})
    return _apply_poster_da_event_titles(poster_recipe)


def _apply_poster_da_event_titles(recipe: MapRecipe) -> MapRecipe:
    panels = tuple(
        replace(panel, title=_POSTER_DA_EVENT_TITLE_OVERRIDES.get(panel.title, panel.title))
        for panel in recipe.panels
    )
    return replace(recipe, panels=panels)


def _map_recipe_output_path(project_dir: Path, recipe: MapRecipe) -> Path:
    output_dir = project_maps_output_dir(project_dir)
    if recipe.output_subdir:
        output_dir = output_dir / recipe.output_subdir
    return output_dir / f"{recipe.output_stem}.png"


def poster_map_recipes(project_dir: Path, config: PosterConfig) -> tuple[MapRecipe, ...]:
    recipes: list[MapRecipe] = []
    if config.setup_overview.enabled:
        maps_config = load_project_maps_config(default_project_maps_config_path(project_dir))
        by_name = {recipe.name: recipe for recipe in maps_config.maps}
        try:
            setup_recipe = by_name[config.setup_overview.name]
        except KeyError as exc:
            raise ValueError(f"Poster setup_overview map not found in maps.yml: {config.setup_overview.name}") from exc
        recipes.append(poster_setup_overview_recipe(setup_recipe, config.setup_overview))

    if config.da_events.enabled:
        da_recipes = tuple(generated_da_map_recipes(project_dir))
        if config.da_events.names:
            requested = set(config.da_events.names)
            available = {recipe.name for recipe in da_recipes}
            missing = sorted(requested - available)
            if missing:
                raise ValueError(f"Poster da_events.names not found: {', '.join(missing)}")
            da_recipes = tuple(recipe for recipe in da_recipes if recipe.name in requested)
        recipes.extend(poster_da_event_recipe(recipe, config.da_events) for recipe in da_recipes)

    if not recipes:
        return ()
    return tuple(recipes)


@dataclass(frozen=True)
class _PosterMapTask:
    recipe: MapRecipe
    output_path: Path
    target_size: PosterTargetSize | None = None
    style: PosterRenderStyle = PosterRenderStyle()


def _map_target_size(recipe: MapRecipe, config: PosterConfig) -> PosterTargetSize | None:
    if recipe.output_subdir == GENERATED_DA_MAPS_SUBDIR:
        return config.da_events.target_size
    if recipe.name == config.setup_overview.name:
        return config.setup_overview.target_size
    return None


def _render_poster_map_task(
    project_dir: Path,
    task: _PosterMapTask,
    shared_model_vmax: dict[str, float] | None = None,
) -> Path:
    project_dir = Path(project_dir).resolve()
    context = load_static_context(project_dir)
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    return render_map_recipe(
        project_dir=project_dir,
        context=context,
        recipe=task.recipe,
        output_path=task.output_path,
        runtime_cache=RenderRuntimeCache(shared_model_vmax=dict(shared_model_vmax or {})),
        target_size_in=task.target_size.inches if task.target_size is not None else None,
        poster_style=task.style,
    )


def _render_poster_maps_sequential(
    project_dir: Path,
    tasks: tuple[_PosterMapTask, ...],
    shared_model_vmax: dict[str, float],
) -> list[Path]:
    context = load_static_context(project_dir)
    runtime_cache = RenderRuntimeCache(shared_model_vmax=dict(shared_model_vmax))
    outputs: list[Path] = []
    for task in tasks:
        logger.info("Starting poster map {}", task.recipe.name)
        try:
            task.output_path.parent.mkdir(parents=True, exist_ok=True)
            output = render_map_recipe(
                project_dir=project_dir,
                context=context,
                recipe=task.recipe,
                output_path=task.output_path,
                runtime_cache=runtime_cache,
                target_size_in=task.target_size.inches if task.target_size is not None else None,
                poster_style=task.style,
            )
        except Exception as exc:
            raise ProjectMapRenderError(task.recipe.name, "poster", str(exc)) from exc
        logger.info("Finished poster map {} -> {}", task.recipe.name, output)
        outputs.append(output)
    return outputs


def _render_poster_maps_parallel(
    project_dir: Path,
    tasks: tuple[_PosterMapTask, ...],
    *,
    max_workers: int,
    shared_model_vmax: dict[str, float],
) -> list[Path]:
    output_by_name: dict[str, Path] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_by_task = {
            executor.submit(_render_poster_map_task, project_dir, task, shared_model_vmax): task
            for task in tasks
        }
        try:
            for future in as_completed(future_by_task):
                task = future_by_task[future]
                try:
                    output_by_name[task.recipe.name] = future.result()
                except Exception as exc:
                    raise ProjectMapRenderError(task.recipe.name, "poster", str(exc)) from exc
        except Exception:
            for future in future_by_task:
                future.cancel()
            raise
    return [output_by_name[task.recipe.name] for task in tasks]


def render_poster_maps(
    *,
    project_dir: Path,
    config: PosterConfig,
    max_workers: int | None = None,
) -> list[Path]:
    recipes = poster_map_recipes(project_dir, config)
    if not recipes:
        return []
    tasks = tuple(
        _PosterMapTask(
            recipe=recipe,
            output_path=project_poster_output_path(project_dir, _map_recipe_output_path(project_dir, recipe)),
            target_size=_map_target_size(recipe, config),
            style=config.theme.render_style(),
        )
        for recipe in recipes
    )
    effective_workers = _resolve_effective_max_workers(max_workers, recipe_count=len(tasks))
    shared_model_vmax = _collect_shared_model_vmax(
        project_dir,
        tuple(recipe for recipe in recipes if recipe.output_subdir == GENERATED_DA_MAPS_SUBDIR) or recipes,
    )
    logger.info("Rendering {} poster map(s) with {} worker(s) ...", len(tasks), effective_workers)
    if effective_workers == 1:
        return _render_poster_maps_sequential(project_dir, tasks, shared_model_vmax)
    return _render_poster_maps_parallel(
        project_dir,
        tasks,
        max_workers=effective_workers,
        shared_model_vmax=shared_model_vmax,
    )


def _write_temp_result_overview_config(panels: tuple[dict[str, object], ...]) -> Path:
    temp = NamedTemporaryFile("w", encoding="utf-8", prefix="openamundsen-da-poster-", suffix=".yml", delete=False)
    path = Path(temp.name)
    with temp:
        _yaml.dump({"panels": list(panels)}, temp)
    return path


def render_poster_result_overview(
    project_dir: Path,
    config: PosterResultOverviewConfig,
    *,
    style: PosterRenderStyle = PosterRenderStyle(),
) -> Path | None:
    if not config.enabled:
        return None
    output = project_poster_output_path(project_dir, project_result_overview_custom_output_path(project_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_config = _write_temp_result_overview_config(config.panels)
    try:
        rc = plot_result_overview_cli(
            [
                "--project-dir",
                str(project_dir),
                "--custom-config",
                str(temp_config),
                "--output",
                str(output),
                "--no-paper-mirror",
                *(
                    [
                        "--target-size-mm",
                        f"{config.target_size.width_mm:g}",
                        f"{config.target_size.height_mm:g}",
                    ]
                    if config.target_size is not None
                    else []
                ),
                *(["--poster-h-pad", f"{config.h_pad:g}"] if config.h_pad is not None else []),
                *(["--poster-hspace", f"{config.hspace:g}"] if config.hspace is not None else []),
                "--poster-panel-height-factor",
                f"{config.panel_height_factor:g}",
                *(["--poster-align-first-xtick-left"] if config.align_first_xtick_left else []),
                "--style-scale",
                f"{style.scale:g}",
                *(
                    [
                        "--poster-title-pt",
                        f"{style.typography.title_pt:g}",
                        "--poster-label-pt",
                        f"{style.typography.label_pt:g}",
                        "--poster-support-pt",
                        f"{style.typography.support_pt:g}",
                    ]
                    if style.typography is not None
                    else []
                ),
                *(
                    ["--poster-panel-box-pt", f"{style.linework.panel_box_pt:g}"]
                    if style.linework is not None
                    else []
                ),
            ],
            configure_logger=False,
        )
    finally:
        try:
            temp_config.unlink()
        except OSError:
            pass
    if isinstance(rc, int) and rc != 0:
        raise RuntimeError(f"plot_result_overview_cli returned {rc}")
    return output


def render_poster_profile(
    *,
    project_dir: Path,
    config_path: Path | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    project_dir = Path(project_dir).resolve()
    path = Path(config_path) if config_path is not None else default_project_poster_config_path(project_dir)
    config = load_poster_config(path)
    outputs: list[Path] = []
    outputs.extend(render_poster_maps(project_dir=project_dir, config=config, max_workers=max_workers))
    overview = render_poster_result_overview(project_dir, config.result_overview_custom, style=config.theme.render_style())
    if overview is not None:
        outputs.append(overview)
    return outputs


def _matmul(a: tuple[float, float, float, float, float, float], b: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
    return (
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    )


def _transform_numbers(value: str) -> list[float]:
    return [float(match) for match in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value)]


def _transform_matrix(value: str | None) -> tuple[float, float, float, float, float, float]:
    matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    if not value:
        return matrix
    for name, raw_args in re.findall(r"(matrix|translate|scale|rotate)\s*\(([^)]*)\)", value):
        args = _transform_numbers(raw_args)
        if name == "matrix" and len(args) >= 6:
            current = tuple(args[:6])
        elif name == "translate":
            current = (1.0, 0.0, 0.0, 1.0, args[0] if args else 0.0, args[1] if len(args) > 1 else 0.0)
        elif name == "scale":
            sx = args[0] if args else 1.0
            sy = args[1] if len(args) > 1 else sx
            current = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate":
            angle = math.radians(args[0] if args else 0.0)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rotation = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                current = _matmul(
                    (1.0, 0.0, 0.0, 1.0, cx, cy),
                    _matmul(rotation, (1.0, 0.0, 0.0, 1.0, -cx, -cy)),
                )
            else:
                current = rotation
        else:
            current = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        matrix = _matmul(matrix, current)
    return matrix


def _apply_matrix(matrix: tuple[float, float, float, float, float, float], x: float, y: float) -> tuple[float, float]:
    return (matrix[0] * x + matrix[2] * y + matrix[4], matrix[1] * x + matrix[3] * y + matrix[5])


def _svg_image_size_mm(
    image: ET.Element,
    matrix: tuple[float, float, float, float, float, float],
) -> PosterTargetSize:
    x = float(image.get("x") or 0.0)
    y = float(image.get("y") or 0.0)
    width = float(image.get("width") or 0.0)
    height = float(image.get("height") or 0.0)
    points = (
        _apply_matrix(matrix, x, y),
        _apply_matrix(matrix, x + width, y),
        _apply_matrix(matrix, x, y + height),
        _apply_matrix(matrix, x + width, y + height),
    )
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    return PosterTargetSize(width_mm=max(xs) - min(xs), height_mm=max(ys) - min(ys))


def _embedded_image_hash(image: ET.Element) -> str | None:
    href = image.get(f"{_XLINK_NS}href") or image.get("href") or ""
    if not href.startswith("data:image") or "," not in href:
        return None
    try:
        data = base64.b64decode(href.split(",", 1)[1], validate=False)
    except (ValueError, binascii.Error):
        return None
    return hashlib.sha256(data).hexdigest()


def _linked_image_path(image: ET.Element) -> Path | None:
    href = image.get(f"{_XLINK_NS}href") or image.get("href") or ""
    if not href or href.startswith("data:"):
        return None
    if href.startswith("file:"):
        parsed = urlparse(href)
        netloc = unquote(parsed.netloc or "")
        path = unquote(parsed.path or "")
        if netloc.lower() in {"wsl.localhost", "wsl$"}:
            parts = Path(path).parts
            if len(parts) >= 3:
                return Path("/") / Path(*parts[2:])
        if path.startswith("//wsl.localhost/") or path.startswith("//wsl$/"):
            parts = Path(path).parts
            if len(parts) >= 4:
                return Path("/") / Path(*parts[3:])
        if netloc and path:
            return Path(f"//{netloc}{path}")
        return Path(path)
    return Path(unquote(href))


def _poster_png_hashes(project_dir: Path) -> dict[str, Path]:
    root = project_poster_root(project_dir)
    out: dict[str, Path] = {}
    for path in root.rglob("*.png"):
        if path.is_file():
            out[hashlib.sha256(path.read_bytes()).hexdigest()] = path
    return out


def _poster_target_key(project_dir: Path, output_path: Path) -> str | None:
    output_path = output_path.resolve()
    poster_root = project_poster_root(project_dir).resolve()
    try:
        rel = output_path.relative_to(poster_root)
    except ValueError:
        parts = output_path.parts
        rel = None
        for idx in range(len(parts) - 1):
            if parts[idx : idx + 2] == ("results", "poster"):
                rel = Path(*parts[idx + 2 :])
                break
        if rel is None:
            return None
    parts = rel.parts
    if rel == Path("maps/setup_overview.png"):
        return "setup_overview"
    if len(parts) == 3 and parts[0] == "maps" and parts[1] == GENERATED_DA_MAPS_SUBDIR and parts[2].startswith("da_"):
        return "da_events"
    if rel == Path("plots/results/result_overview_custom.png"):
        return "result_overview_custom"
    return None


def _collect_svg_image_measurements(
    element: ET.Element,
    matrix: tuple[float, float, float, float, float, float],
    *,
    png_by_hash: dict[str, Path],
    project_dir: Path,
    measurements: dict[str, list[PosterTargetSize]],
) -> None:
    current_matrix = _matmul(matrix, _transform_matrix(element.get("transform")))
    if element.tag == f"{_SVG_NS}image":
        digest = _embedded_image_hash(element)
        output_path = png_by_hash.get(digest) if digest is not None else _linked_image_path(element)
        if output_path is not None:
            key = _poster_target_key(project_dir, Path(output_path))
            if key is not None:
                measurements.setdefault(key, []).append(_svg_image_size_mm(element, current_matrix))
    for child in list(element):
        _collect_svg_image_measurements(
            child,
            current_matrix,
            png_by_hash=png_by_hash,
            project_dir=project_dir,
            measurements=measurements,
        )


def _mean_target_size(sizes: list[PosterTargetSize]) -> PosterTargetSize:
    if not sizes:
        raise ValueError("Cannot average empty poster target size list")
    width = sum(size.width_mm for size in sizes) / len(sizes)
    height = sum(size.height_mm for size in sizes) / len(sizes)
    return PosterTargetSize(width_mm=width, height_mm=height)


def measure_poster_svg_targets(project_dir: Path, svg_path: Path) -> dict[str, PosterTargetSize]:
    project_dir = Path(project_dir).resolve()
    svg_path = Path(svg_path)
    if not svg_path.is_file():
        raise FileNotFoundError(f"Poster SVG not found: {svg_path}")
    png_by_hash = _poster_png_hashes(project_dir)
    if not png_by_hash:
        raise FileNotFoundError(f"No poster PNGs found under {project_poster_root(project_dir)}")
    root = ET.parse(svg_path).getroot()
    measurements: dict[str, list[PosterTargetSize]] = {}
    _collect_svg_image_measurements(
        root,
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        png_by_hash=png_by_hash,
        project_dir=project_dir,
        measurements=measurements,
    )
    return {key: _mean_target_size(values) for key, values in measurements.items()}


def _target_size_yaml(size: PosterTargetSize) -> list[float]:
    return [round(float(size.width_mm), 6), round(float(size.height_mm), 6)]


def write_poster_target_sizes(config_path: Path, sizes: dict[str, PosterTargetSize]) -> None:
    config_path = Path(config_path)
    cfg = _yaml.load(config_path.read_text(encoding="utf-8")) or {}
    maps = cfg.setdefault("maps", {})
    plots = cfg.setdefault("plots", {})
    if "setup_overview" in sizes:
        setup = maps.setdefault("setup_overview", {})
        setup["target_size_mm"] = _target_size_yaml(sizes["setup_overview"])
    if "da_events" in sizes:
        da_events = maps.setdefault("da_events", {})
        da_events["target_size_mm"] = _target_size_yaml(sizes["da_events"])
    if "result_overview_custom" in sizes:
        overview = plots.setdefault("result_overview_custom", {})
        overview["target_size_mm"] = _target_size_yaml(sizes["result_overview_custom"])
    with config_path.open("w", encoding="utf-8") as f:
        _yaml.dump(cfg, f)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oa-da-plot-poster",
        description="Render configured poster-profile plots and maps from existing project outputs.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory")
    parser.add_argument("--config", type=Path, help="Poster config path (default: <project>/poster.yml)")
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        help="Maximum poster map render workers (default: auto, clamped to CPUs and selected map count)",
    )
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = parser.parse_args(argv)

    configure_cli_logger(args.log_level)
    try:
        outputs = render_poster_profile(
            project_dir=args.project_dir,
            config_path=args.config,
            max_workers=args.max_workers,
        )
    except Exception as exc:
        logger.error("Poster rendering failed: {}", exc)
        logger.error(
            "Rerun poster rendering with: {}",
            default_project_poster_rerun_command(args.project_dir, config_path=args.config),
        )
        return 1
    logger.info("Poster rendering complete -> {} output(s)", len(outputs))
    return 0


def measure_cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oa-da-poster-measure",
        description="Measure final poster asset sizes from an Inkscape SVG and optionally write poster.yml.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory")
    parser.add_argument("--svg", required=True, type=Path, help="Poster SVG containing embedded current poster PNGs")
    parser.add_argument("--config", type=Path, help="Poster config path (default: <project>/poster.yml)")
    parser.add_argument("--write", action="store_true", help="Write measured target_size_mm values to poster.yml")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = parser.parse_args(argv)

    configure_cli_logger(args.log_level)
    project_dir = Path(args.project_dir).resolve()
    config_path = Path(args.config) if args.config is not None else default_project_poster_config_path(project_dir)
    try:
        sizes = measure_poster_svg_targets(project_dir, args.svg)
        if not sizes:
            raise ValueError("No matching embedded poster images found")
        if args.write:
            write_poster_target_sizes(config_path, sizes)
            logger.info("Wrote poster target sizes to {}", config_path)
        for key in sorted(sizes):
            size = sizes[key]
            logger.info("{} target_size_mm: [{:.6f}, {:.6f}]", key, size.width_mm, size.height_mm)
    except Exception as exc:
        logger.error("Poster SVG measurement failed: {}", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
