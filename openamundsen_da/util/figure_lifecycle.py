"""Fail-safe Matplotlib figure lifecycle helpers."""

from __future__ import annotations

from functools import wraps
import sys
from typing import Callable, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")


def close_created_figures(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Close every figure created by one plotting call, including on failure."""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pyplot = sys.modules.get("matplotlib.pyplot")
        before = set(pyplot.get_fignums()) if pyplot is not None else set()
        try:
            return function(*args, **kwargs)
        finally:
            pyplot = sys.modules.get("matplotlib.pyplot")
            if pyplot is not None:
                for figure_number in set(pyplot.get_fignums()) - before:
                    pyplot.close(figure_number)

    return wrapped


__all__ = ["close_created_figures"]
