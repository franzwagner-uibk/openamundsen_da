#!/usr/bin/env python3
"""Render the minimal title-free profile used by the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

from openamundsen_da.io.paths import project_paper_root
from openamundsen_da.methods.viz.maps.runner import render_project_map_profile
from openamundsen_da.methods.viz.plots.assimilation.weights import plot_setup_weights_overview


PROJECT_NAME = "project_2022_2023"
MAP_RECIPE_NAMES = {"da_6", "da_8"}
PROFILE_RELATIVE_OUTPUTS = (
    Path("maps/da_events/da_6.png"),
    Path("maps/da_events/da_8.png"),
    Path("plots/assim/weights/setup_weights_overview_2022_2023.png"),
)


class ManuscriptProfileError(RuntimeError):
    """Raised when the manuscript render profile is incomplete or expands."""


def _existing_profile_files(paper_root: Path) -> tuple[Path, ...]:
    if not paper_root.is_dir():
        return ()
    return tuple(sorted(path.relative_to(paper_root) for path in paper_root.rglob("*") if path.is_file()))


def _replace_profile_tree(temporary_root: Path, paper_root: Path) -> None:
    """Install a rendered profile and restore the previous tree on swap failure."""
    backup_root: Path | None = None
    if paper_root.exists():
        backup_root = Path(
            tempfile.mkdtemp(prefix=".manuscript-profile-backup-", dir=paper_root.parent)
        )
        backup_root.rmdir()
        paper_root.replace(backup_root)
    try:
        temporary_root.replace(paper_root)
    except OSError as replace_error:
        if backup_root is not None:
            try:
                backup_root.replace(paper_root)
            except OSError as restore_error:
                raise ManuscriptProfileError(
                    "Failed to install the manuscript profile and restore the previous tree: "
                    f"install error={replace_error}; restore error={restore_error}"
                ) from restore_error
        raise
    if backup_root is not None:
        shutil.rmtree(backup_root, ignore_errors=True)


def render_manuscript_profile(root: Path) -> tuple[Path, ...]:
    """Replace ``results/paper`` with exactly the declared profile outputs."""
    root = Path(root).resolve(strict=True)
    project_dir = root / "projects" / PROJECT_NAME
    if not project_dir.is_dir():
        raise ManuscriptProfileError(f"Missing manuscript reference project: {project_dir}")

    paper_root = project_paper_root(project_dir)
    paper_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".manuscript-profile-", dir=paper_root.parent))
    expected = tuple(sorted(PROFILE_RELATIVE_OUTPUTS))
    try:
        weights_output = (
            temporary_root
            / "plots"
            / "assim"
            / "weights"
            / "setup_weights_overview_2022_2023.png"
        )
        plot_setup_weights_overview(
            project_dir,
            output=weights_output,
            show_figure_title=False,
        )
        render_project_map_profile(
            project_dir=project_dir,
            output_root=temporary_root / "maps",
            names=MAP_RECIPE_NAMES,
            strip_figure_titles=True,
        )

        actual = _existing_profile_files(temporary_root)
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            unexpected = sorted(set(actual) - set(expected))
            details: list[str] = []
            if missing:
                details.append("missing: " + ", ".join(map(str, missing)))
            if unexpected:
                details.append("unexpected: " + ", ".join(map(str, unexpected)))
            raise ManuscriptProfileError(
                "Manuscript profile output mismatch (" + "; ".join(details) + ")"
            )
        _replace_profile_tree(temporary_root, paper_root)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
    return tuple(paper_root / relative for relative in expected)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Completed manuscript reference setup root")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace the generated results/paper tree with the declared profile",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        project_dir = root / "projects" / PROJECT_NAME
        paper_root = project_paper_root(project_dir)
        if not args.apply:
            print(f"Manuscript profile preview for {project_dir}")
            for relative in sorted(PROFILE_RELATIVE_OUTPUTS):
                print(f"WRITE {paper_root / relative}")
            for relative in _existing_profile_files(paper_root):
                if relative not in PROFILE_RELATIVE_OUTPUTS:
                    print(f"REMOVE {paper_root / relative}")
            return 0
        outputs = render_manuscript_profile(root)
    except (ManuscriptProfileError, OSError, RuntimeError, ValueError) as exc:
        print(f"manuscript profile error: {exc}")
        return 2
    print(f"Rendered {len(outputs)} manuscript profile output(s) under {paper_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
