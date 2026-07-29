#!/usr/bin/env python3
"""Materialize the shipped Rofental setup used by the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "examples" / "rofental"


class ManuscriptSetupError(ValueError):
    """Raised when the shipped manuscript setup cannot be materialized safely."""


def materialize_setup(
    *,
    base_setup: Path,
    target: Path,
    overwrite: bool = False,
) -> Path:
    """Copy the shipped setup without applying scientific-input overlays."""
    base_setup = base_setup.resolve(strict=True)
    target = target.expanduser().resolve()
    if target == base_setup or base_setup in target.parents or target in base_setup.parents:
        raise ManuscriptSetupError(f"Target must be outside the shipped setup tree: {target}")
    if target.exists():
        if not overwrite:
            raise ManuscriptSetupError(f"Target already exists; pass --overwrite to replace it: {target}")
        if target == Path(target.anchor):
            raise ManuscriptSetupError(f"Refusing to remove filesystem root: {target}")
        shutil.rmtree(target)

    shutil.copytree(base_setup, target)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="New setup directory to create")
    parser.add_argument("--base-setup", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        target = materialize_setup(
            base_setup=args.base_setup,
            target=args.target,
            overwrite=bool(args.overwrite),
        )
    except (OSError, ManuscriptSetupError) as exc:
        print(f"manuscript setup error: {exc}")
        return 1
    print(f"Materialized shipped manuscript setup at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
