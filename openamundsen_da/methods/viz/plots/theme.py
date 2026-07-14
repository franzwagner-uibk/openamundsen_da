"""Shared visualization style constants for ensemble plot modules."""

from __future__ import annotations

from openamundsen_da.methods.viz.theme import (
    DA_VARIABLE_STYLES,
    EXPORT_DPI,
    FIGHEIGHT_OVERVIEW_ROW,
    FIGWIDTH_OVERVIEW_PAPER,
    TEXT_COLOR as COLOR_TEXT,
    canonical_da_variable,
    da_variable_fill_color,
    da_variable_line_color,
    da_variable_style,
)

COLOR_MEAN = "#3c4f8a"
COLOR_OPEN_LOOP = "black"
COLOR_MEMBER = "#9a9a9a"
# One shared ensemble-band opacity across plot modules and legend swatches.
BAND_ALPHA = 0.32
# Use one shared linewidth across all plotted data lines.
LW_MEMBER = 1.8
LW_MEAN = 1.8
LW_OPEN = 1.8
LEGEND_NCOL = 4
LEGEND_NCOL_SETUP = 6

# Observation markers (e.g., SCF DA points and station obs)
# Use a distinct color from COLOR_OPEN_LOOP ("black") so open-loop and
# observations are clearly distinguishable in results plots.
COLOR_DA_OBS = "#d62728"
SIZE_DA_OBS = 80
LW_DA_OBS = 1.8
LS_STATION_OBS = "--"
COLOR_OBS_SCF = COLOR_DA_OBS
SIZE_OBS_SCF = 10

# Grid style
GRID_LS = ":"
GRID_LW = 0.7
GRID_ALPHA = 0.72

# Titles / text
FS_TITLE = 11.4
FS_SUBTITLE = 10
OVERVIEW_AXIS_LABEL_SIZE = 7.6
OVERVIEW_YTICK_SIZE = 7.0
OVERVIEW_XTICK_SIZE = 7.2
COLOR_SUBTITLE = COLOR_TEXT
FS_ASSIM_LABEL = 9
ASSIM_LABEL_ROT = 45
TITLE_PAD_WITH_ASSIM_LABELS = 16.0
TITLE_PAD_DEFAULT = 9.0
FIGURE_TITLE_CLEARANCE_PTS = 2.0
FIGURE_TITLE_TOP_MARGIN_PTS = 0.5
METRIC_AXIS_MIN_INTERVALS = 2
METRIC_AXIS_MAX_INTERVALS = 6
CRPSS_AXIS_STEP_CANDIDATES = (0.25, 0.5, 1.0, 2.0)
CRPSS_AXIS_UPPER_CAP = 1.0

# Figure sizes
FIGSIZE_FORCING = (12.0, 6.0)
FIGSIZE_RESULTS = (10.2, 5.2)
OVERVIEW_STANDARD_PANEL_HEIGHT_FACTOR = 0.64
OVERVIEW_SCORE_PANEL_HEIGHT_FACTOR = 1.428
STANDALONE_SCORE_PANEL_HEIGHT_FACTOR = 0.748

__all__ = [
    "ASSIM_LABEL_ROT",
    "BAND_ALPHA",
    "COLOR_DA_OBS",
    "COLOR_MEAN",
    "COLOR_MEMBER",
    "COLOR_OBS_SCF",
    "COLOR_OPEN_LOOP",
    "COLOR_SUBTITLE",
    "COLOR_TEXT",
    "CRPSS_AXIS_STEP_CANDIDATES",
    "CRPSS_AXIS_UPPER_CAP",
    "DA_VARIABLE_STYLES",
    "EXPORT_DPI",
    "FIGHEIGHT_OVERVIEW_ROW",
    "FIGSIZE_FORCING",
    "FIGSIZE_RESULTS",
    "FIGURE_TITLE_CLEARANCE_PTS",
    "FIGURE_TITLE_TOP_MARGIN_PTS",
    "FIGWIDTH_OVERVIEW_PAPER",
    "FS_ASSIM_LABEL",
    "FS_SUBTITLE",
    "FS_TITLE",
    "GRID_ALPHA",
    "GRID_LS",
    "GRID_LW",
    "LEGEND_NCOL",
    "LEGEND_NCOL_SETUP",
    "LS_STATION_OBS",
    "LW_DA_OBS",
    "LW_MEAN",
    "LW_MEMBER",
    "LW_OPEN",
    "METRIC_AXIS_MAX_INTERVALS",
    "METRIC_AXIS_MIN_INTERVALS",
    "OVERVIEW_SCORE_PANEL_HEIGHT_FACTOR",
    "OVERVIEW_AXIS_LABEL_SIZE",
    "OVERVIEW_STANDARD_PANEL_HEIGHT_FACTOR",
    "OVERVIEW_XTICK_SIZE",
    "OVERVIEW_YTICK_SIZE",
    "SIZE_DA_OBS",
    "SIZE_OBS_SCF",
    "STANDALONE_SCORE_PANEL_HEIGHT_FACTOR",
    "TITLE_PAD_DEFAULT",
    "TITLE_PAD_WITH_ASSIM_LABELS",
    "canonical_da_variable",
    "da_variable_fill_color",
    "da_variable_line_color",
    "da_variable_style",
]
