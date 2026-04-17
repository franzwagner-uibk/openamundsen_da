"""Compatibility shim for the structured ``viz.plots.common`` module."""

from __future__ import annotations

from openamundsen_da.methods.viz.plots import common as _impl

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})

