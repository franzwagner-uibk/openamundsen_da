from __future__ import annotations

from pathlib import Path


def cli_main(argv: list[str] | None = None) -> int:
    from .runner import cli_main as _cli_main

    return _cli_main(argv)


def project_maps_enabled(project_dir: Path, config_path: Path | None = None) -> bool:
    from .runner import project_maps_enabled as _project_maps_enabled

    return _project_maps_enabled(project_dir, config_path)


def render_project_maps(
    *,
    project_dir: Path,
    config_path: Path | None = None,
    names: set[str] | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    from .runner import render_project_maps as _render_project_maps

    return _render_project_maps(
        project_dir=project_dir,
        config_path=config_path,
        names=names,
        max_workers=max_workers,
    )


__all__ = ["cli_main", "project_maps_enabled", "render_project_maps"]
