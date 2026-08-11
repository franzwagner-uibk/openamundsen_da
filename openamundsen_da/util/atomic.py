"""Crash-durable promotion helpers for accepted package outputs."""

from __future__ import annotations

import os
from pathlib import Path


def fsync_file(path: str | Path) -> None:
    """Flush one completed regular file to stable storage."""
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: str | Path) -> None:
    """Flush directory metadata after an atomic rename."""
    directory_fd = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def durable_replace(temp: str | Path, target: str | Path) -> Path:
    """Durably replace ``target`` with a fully written same-directory file."""
    temp = Path(temp)
    target = Path(target)
    if temp.parent.resolve() != target.parent.resolve():
        raise ValueError("Durable replacement requires a same-directory temporary")
    fsync_file(temp)
    os.replace(temp, target)
    fsync_directory(target.parent)
    return target


__all__ = ["durable_replace", "fsync_directory", "fsync_file"]
