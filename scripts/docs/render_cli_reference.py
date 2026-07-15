#!/usr/bin/env python3
"""Render the supported command tree from the package's argparse parser."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "reference" / "cli.md"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _command_parsers(
    parser: argparse.ArgumentParser,
    command: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], argparse.ArgumentParser]]:
    yield command, parser
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, child in action.choices.items():
            yield from _command_parsers(child, (*command, name))


def render_cli_reference() -> str:
    """Return the complete supported CLI reference as Markdown."""
    from openamundsen_da.cli import build_parser

    lines = [
        "---",
        "layout: default",
        "title: Command-Line Interface",
        "parent: Reference",
        "nav_order: 1",
        "---",
        "",
        "# Command-Line Interface",
        "",
        "This page is generated from `openamundsen_da.cli.build_parser`. It documents the",
        "single supported command tree installed by the `openamundsen-da` package.",
        "",
        "Run `python scripts/docs/render_cli_reference.py` after changing the parser.",
        "The documentation gate fails if this file is stale.",
        "",
    ]
    for command, parser in _command_parsers(build_parser()):
        display = "openamundsen-da" if not command else f"openamundsen-da {' '.join(command)}"
        heading_level = min(2 + len(command), 5)
        lines.extend(
            [
                f"{'#' * heading_level} `{display}`",
                "",
                "```text",
                parser.format_help().rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail instead of updating a stale output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    expected = render_cli_reference()
    output = args.output.resolve()
    current = output.read_text(encoding="utf-8") if output.is_file() else None
    if args.check:
        if current != expected:
            print(f"Generated CLI reference is stale: {output}", file=sys.stderr)
            return 1
        print(f"Generated CLI reference is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    print(f"Rendered CLI reference: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
