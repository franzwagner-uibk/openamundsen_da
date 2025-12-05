"""Shared visualization style constants for ensemble plots."""

COLOR_MEAN = "#120fb6"
COLOR_OPEN_LOOP = "black"
BAND_ALPHA = 0.18
LW_MEMBER = 0.9
# Use a common, clearly visible linewidth for key summary lines
# (ensemble mean, open loop, and station observations).
LW_MEAN = 2.0
LW_OPEN = 2.0
LEGEND_NCOL = 4
LEGEND_NCOL_SEASON = 6

# Observation markers (e.g., SCF DA points and station obs)
# Use a distinct color from COLOR_OPEN_LOOP ("black") so open-loop and
# observations are clearly distinguishable in results plots.
COLOR_DA_OBS = "#d62728"
SIZE_DA_OBS = 100
LW_DA_OBS = 2.0
COLOR_OBS_SCF = "#d62728"
SIZE_OBS_SCF = 10

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
FIGSIZE_RESULTS = (12.0, 5.2)
