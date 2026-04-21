"""Structured plot subpackage for non-map visualization CLIs and helpers."""

from __future__ import annotations

from pathlib import Path


def cli_main(argv: list[str] | None = None) -> int:
    from .runner import cli_main as _cli_main

    return _cli_main(argv)


def render_project_plots(
    *,
    project_dir: Path,
    plot_workers: int | None = None,
    max_workers: int | None = None,
) -> list[str]:
    from .runner import render_project_plots as _render_project_plots

    return _render_project_plots(
        project_dir=project_dir,
        plot_workers=plot_workers,
        max_workers=max_workers,
    )


__all__ = ["cli_main", "render_project_plots"]
