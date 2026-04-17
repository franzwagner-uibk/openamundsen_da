"""Compatibility wrapper for benchmark rendering/writing helpers."""

from openamundsen_da.methods.viz.plots.benchmark import write_plots
from .render.tables import (
    write_case_tables,
    write_manifest,
    write_score_tables,
    write_summary_markdown,
    write_summary_tables,
)

__all__ = [
    "write_case_tables",
    "write_manifest",
    "write_plots",
    "write_score_tables",
    "write_summary_markdown",
    "write_summary_tables",
]
