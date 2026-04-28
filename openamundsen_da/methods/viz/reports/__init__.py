"""Report assembly helpers for project visualization outputs."""

from __future__ import annotations

from pathlib import Path


def cli_main(argv: list[str] | None = None) -> int:
    from .project_collection_pdf import cli_main as _cli_main

    return _cli_main(argv)


def build_project_collection_pdf(*, project_dir: Path, output: Path | None = None) -> Path:
    from .project_collection_pdf import build_project_collection_pdf as _build_project_collection_pdf

    return _build_project_collection_pdf(project_dir=project_dir, output=output)


__all__ = ["build_project_collection_pdf", "cli_main"]
