from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openamundsen_da.core.env import _read_yaml_file


@dataclass(frozen=True)
class DateSelector:
    explicit: tuple[str, ...] = ()
    assimilation_variables: tuple[str, ...] = ()
    include_first: bool = False
    include_last: bool = False


@dataclass(frozen=True)
class OverviewMapRecipe:
    name: str
    title: str | None = None


@dataclass(frozen=True)
class ComparisonMapRecipe:
    name: str
    variable: str
    title: str | None = None
    dates: DateSelector = DateSelector()


@dataclass(frozen=True)
class ObservationContextMapRecipe:
    name: str
    model_variable: str
    observation: str
    title: str | None = None
    dates: DateSelector = DateSelector()


@dataclass(frozen=True)
class ProjectMapsConfig:
    path: Path
    overview_maps: tuple[OverviewMapRecipe, ...]
    comparison_maps: tuple[ComparisonMapRecipe, ...]
    observation_context_maps: tuple[ObservationContextMapRecipe, ...]

    def all_names(self) -> set[str]:
        return {
            *[item.name for item in self.overview_maps],
            *[item.name for item in self.comparison_maps],
            *[item.name for item in self.observation_context_maps],
        }


def default_project_maps_config_path(project_dir: Path) -> Path:
    return Path(project_dir) / "project_maps.yml"


def _require_mapping(value: object, *, context: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_str(value: object, *, context: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{context} must be a non-empty string")
    return text


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def _coerce_str_list(value: object, *, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings")
    return tuple(_require_str(item, context=f"{context}[]") for item in value)


def _parse_date_selector(value: object, *, context: str) -> DateSelector:
    mapping = _require_mapping(value, context=context)
    return DateSelector(
        explicit=_coerce_str_list(mapping.get("explicit"), context=f"{context}.explicit"),
        assimilation_variables=_coerce_str_list(
            mapping.get("assimilation_variables"),
            context=f"{context}.assimilation_variables",
        ),
        include_first=_coerce_bool(mapping.get("include_first"), default=False),
        include_last=_coerce_bool(mapping.get("include_last"), default=False),
    )


def _parse_overview_maps(raw: object) -> tuple[OverviewMapRecipe, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("overview_maps must be a list")
    items: list[OverviewMapRecipe] = []
    for idx, item in enumerate(raw):
        mapping = _require_mapping(item, context=f"overview_maps[{idx}]")
        items.append(
            OverviewMapRecipe(
                name=_require_str(mapping.get("name"), context=f"overview_maps[{idx}].name"),
                title=_optional_str(mapping.get("title")),
            )
        )
    return tuple(items)


def _parse_comparison_maps(raw: object) -> tuple[ComparisonMapRecipe, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("comparison_maps must be a list")
    items: list[ComparisonMapRecipe] = []
    for idx, item in enumerate(raw):
        mapping = _require_mapping(item, context=f"comparison_maps[{idx}]")
        items.append(
            ComparisonMapRecipe(
                name=_require_str(mapping.get("name"), context=f"comparison_maps[{idx}].name"),
                variable=_require_str(mapping.get("variable"), context=f"comparison_maps[{idx}].variable"),
                title=_optional_str(mapping.get("title")),
                dates=_parse_date_selector(mapping.get("dates"), context=f"comparison_maps[{idx}].dates"),
            )
        )
    return tuple(items)


def _parse_observation_context_maps(raw: object) -> tuple[ObservationContextMapRecipe, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("observation_context_maps must be a list")
    items: list[ObservationContextMapRecipe] = []
    for idx, item in enumerate(raw):
        mapping = _require_mapping(item, context=f"observation_context_maps[{idx}]")
        observation = _require_str(
            mapping.get("observation"),
            context=f"observation_context_maps[{idx}].observation",
        ).lower()
        if observation not in {"scf", "wet_snow"}:
            raise ValueError(
                "observation_context_maps[].observation must be one of: scf, wet_snow"
            )
        items.append(
            ObservationContextMapRecipe(
                name=_require_str(
                    mapping.get("name"),
                    context=f"observation_context_maps[{idx}].name",
                ),
                model_variable=_require_str(
                    mapping.get("model_variable"),
                    context=f"observation_context_maps[{idx}].model_variable",
                ),
                observation=observation,
                title=_optional_str(mapping.get("title")),
                dates=_parse_date_selector(
                    mapping.get("dates"),
                    context=f"observation_context_maps[{idx}].dates",
                ),
            )
        )
    return tuple(items)


def load_project_maps_config(config_path: Path) -> ProjectMapsConfig:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Project maps config not found: {config_path}")
    cfg = _read_yaml_file(config_path) or {}
    root = _require_mapping(cfg, context=str(config_path))
    loaded = ProjectMapsConfig(
        path=config_path,
        overview_maps=_parse_overview_maps(root.get("overview_maps")),
        comparison_maps=_parse_comparison_maps(root.get("comparison_maps")),
        observation_context_maps=_parse_observation_context_maps(root.get("observation_context_maps")),
    )
    names = list(loaded.all_names())
    if len(names) != (
        len(loaded.overview_maps) + len(loaded.comparison_maps) + len(loaded.observation_context_maps)
    ):
        raise ValueError(f"Map recipe names must be unique in {config_path}")
    if not (loaded.overview_maps or loaded.comparison_maps or loaded.observation_context_maps):
        raise ValueError(f"Project maps config is empty: {config_path}")
    return loaded
