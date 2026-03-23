"""Shared visualization style constants for ensemble plots."""

COLOR_MEAN = "#120fb6"
COLOR_OPEN_LOOP = "black"
COLOR_MEMBER = "#9a9a9a"
BAND_ALPHA = 0.18
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
SIZE_DA_OBS = 100
LW_DA_OBS = 1.8
COLOR_OBS_SCF = "#d62728"
SIZE_OBS_SCF = 10
COLOR_DA_STATION_HS = "#2b6cb0"
COLOR_DA_STATION_SWE = "#2f855a"
LS_DA_STATION_HS = "-."
LS_DA_STATION_SWE = ":"
LW_DA_STATION = 1.4

# Grid style
GRID_LS = ":"
GRID_LW = 0.6
GRID_ALPHA = 0.7

# Titles / text
FS_TITLE = 12
FS_SUBTITLE = 10
COLOR_SUBTITLE = "#555555"
FS_ASSIM_LABEL = 9
ASSIM_LABEL_ROT = 45

# Figure sizes
FIGSIZE_FORCING = (12.0, 6.0)
FIGSIZE_RESULTS = (10.2, 5.2)
