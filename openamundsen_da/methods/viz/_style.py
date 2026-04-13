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

# Grid style
GRID_LS = ":"
GRID_LW = 0.6
GRID_ALPHA = 0.7

# Titles / text
COLOR_TEXT = "#000000"
FS_TITLE = 12
FS_SUBTITLE = 10
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
FIGWIDTH_OVERVIEW_PAPER = 7.2876875
FIGHEIGHT_OVERVIEW_ROW = 1.71236835
OVERVIEW_SCORE_PANEL_HEIGHT_FACTOR = 1.68
STANDALONE_SCORE_FIGURE_ROW_UNITS = 2.91

# Export
EXPORT_DPI = 600

_DA_VARIABLE_ALIASES = {
    "scf": "scf",
    "fsc": "scf",
    "wet_snow": "wet_snow",
    "fws": "wet_snow",
    "station_hs": "station_hs",
    "station_sd": "station_hs",
    "sd": "station_hs",
    "snow_depth": "station_hs",
    "snowdepth": "station_hs",
    "hs": "station_hs",
    "station_swe": "station_swe",
    "swe": "station_swe",
}

DA_VARIABLE_STYLES = {
    "scf": {"fill": "#9ec5ff", "line": "#2f6fb5"},
    "wet_snow": {"fill": "#9bd8bf", "line": "#2c8a64"},
    "station_hs": {"fill": "#f3c38e", "line": "#ff7f0e"},
    "station_swe": {"fill": "#ccb8f2", "line": "#9467bd"},
}


def canonical_da_variable(variable: str) -> str:
    token = str(variable or "").strip().lower().replace("-", "_")
    return _DA_VARIABLE_ALIASES.get(token, token)


def da_variable_style(variable: str) -> dict[str, str]:
    return DA_VARIABLE_STYLES[canonical_da_variable(variable)]


def da_variable_line_color(variable: str) -> str:
    return da_variable_style(variable)["line"]


def da_variable_fill_color(variable: str) -> str:
    return da_variable_style(variable)["fill"]
