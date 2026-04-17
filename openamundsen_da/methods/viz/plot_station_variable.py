"""Compatibility shim for ``viz.plots.station_variable``."""

from __future__ import annotations

from openamundsen_da.methods.viz.plots import station_variable as _impl

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
