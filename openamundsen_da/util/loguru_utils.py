from __future__ import annotations

import sys
from typing import TextIO

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT


def configure_cli_logger(
    level: str,
    *,
    stream: TextIO | None = None,
    enqueue: bool = True,
    colorize: bool = True,
    fmt: str = LOGURU_FORMAT,
) -> None:
    """Configure a single Loguru sink for CLI entry points."""
    logger.remove()
    logger.add(
        stream or sys.stdout,
        level=str(level).upper(),
        colorize=colorize,
        enqueue=enqueue,
        format=fmt,
    )
