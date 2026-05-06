from __future__ import annotations


_FIGURE_HEIGHT_MIN = 2.9
_FIGURE_HEIGHT_MAX = 16.5
_BUFFER_RATIO = 0.03
_STATION_LABEL_RATIO = 0.04
_GRID_COLOR = "#666666"
_GRID_STYLE = (0, (4, 4))
_GRID_WIDTH = 0.74
_SPINE_WIDTH = 0.85
_TICK_SIZE = 6.6
_TICK_LABEL_MIN_GAP_IN = 0.42
_COLORBAR_TICK_SIZE = 6.0
_COLORBAR_TITLE_SIZE = 6.6
_SUPPORT_PANEL_KINDS = {"colorbar", "legend"}
_STATION_COLOR = "#d94801"
_ROI_FILL = "#efefef"
_OVERVIEW_ROI_COLOR = "#c21f24"
_SUBDOMAIN_BOUNDARY_COLOR = "#222222"
_SUBDOMAIN_BOUNDARY_HALO_COLOR = "white"
_SUBDOMAIN_BOUNDARY_WIDTH = 0.58
_SUBDOMAIN_BOUNDARY_HALO_WIDTH = 1.55
_GRID_ZORDER = 120
_ANNOTATION_ZORDER = 130
_HILLSHADE_INTERPOLATION = "bilinear"
_SNOW_DEPTH_PANEL_ALPHA = 0.70
_LAYOUT_COL_GAP = 0.07
_LAYOUT_ROW_GAP = 0.02
_LEFT_MARGIN = 0.03
_RIGHT_MARGIN = 0.992
_BOTTOM_MARGIN = 0.028
_TOP_MARGIN = 0.992
_DATE_CALLOUT_ALPHA = 1.0
_SCALEBAR_TARGET_FRACTION = 0.27
_SCALEBAR_RIGHT_PAD_FRACTION = 0.04
_SCALEBAR_BOTTOM_FRACTION = 0.060
_OVERVIEW_FRAGMENT_RATIO = 0.018
_VERTICAL_COLORBAR_GAP_EXTRA = 0.11
_VERTICAL_COLORBAR_OUTER_EXTRA = 0.09
_VERTICAL_COLORBAR_XOFFSET_AXES = 0.038
_VERTICAL_COLORBAR_WIDTH_AXES = 0.060
_VERTICAL_COLORBAR_BOTTOM_AXES = 0.035
_VERTICAL_COLORBAR_HEIGHT_AXES = 0.89
_HORIZONTAL_COLORBAR_GAP_AXES = 0.10
_HORIZONTAL_ANNOTATION_MIN_GAP_MAX_ASPECT = 0.85
_HORIZONTAL_COLORBAR_MIN_GAP_IN = 0.26
_HORIZONTAL_COLORBAR_HEIGHT_AXES = 0.050
_HORIZONTAL_COLORBAR_BOTTOM_PAD_AXES = 0.02
_HORIZONTAL_COLORBAR_EXTRA = _HORIZONTAL_COLORBAR_GAP_AXES + _HORIZONTAL_COLORBAR_HEIGHT_AXES + _HORIZONTAL_COLORBAR_BOTTOM_PAD_AXES
_HORIZONTAL_LEGEND_GAP_AXES = 0.095
_HORIZONTAL_LEGEND_MIN_GAP_IN = 0.26
_HORIZONTAL_LEGEND_ROW_HEIGHT_AXES = 0.058
_HORIZONTAL_LEGEND_BOTTOM_PAD_AXES = 0.14
_HORIZONTAL_LEGEND_ITEM_GAP_IN = 0.05
_HORIZONTAL_LEGEND_MIN_ITEM_GAP_IN = 0.022
_HORIZONTAL_LEGEND_HANDLE_WIDTH_IN = 0.145
_HORIZONTAL_LEGEND_HANDLE_TEXT_PAD_IN = 0.025
_HORIZONTAL_LEGEND_MIN_TEXT_WIDTH_IN = 0.12
_HORIZONTAL_LEGEND_SIDE_PAD_IN = 0.018
_HORIZONTAL_LEGEND_TEXT_SIZE = 5.5
_HORIZONTAL_LEGEND_PATCH_HEIGHT_IN = 0.055
_PANEL_BELOW_ITEMS_GAP_AXES = 0.000
_PANEL_BELOW_ITEMS_DRAW_GAP_BASE_AXES = 0.090
_PANEL_BELOW_ITEMS_DRAW_GAP_PER_HEIGHT_AXES = 0.15
_PANEL_BELOW_ITEMS_MIN_GAP_IN = 0.26
_PANEL_BELOW_ITEMS_ROW_HEIGHT_AXES = 0.040
_PANEL_BELOW_ITEMS_BOTTOM_PAD_AXES = 0.000
_OVERVIEW_LABEL_SIZE = 6.2
_OVERVIEW_LABEL_DX_RATIO = 0.09
_OVERVIEW_LABEL_DY_RATIO = 0.07
_OVERVIEW_LABEL_BOX_PAD_EM = 0.10
_OVERVIEW_LABEL_BOX_SAFETY_IN = 0.02
_OVERLAY_LABEL_HALO_COLOR = "white"
_OVERLAY_LABEL_HALO_WIDTH = 2.0
_OVERLAY_LABEL_BBOX_HALO_WIDTH = 2.4
_OVERVIEW_ROI_LABEL_SIZE = 6.4
_OVERVIEW_ROI_LABEL_DX_RATIO = 0.04

_MODEL_KIND_TO_VARIABLE = {
    "snow_depth": "snowdepth_daily",
    "swe": "swe_daily",
    "liquid_water_content": "liquid_water_content",
}
_OBSERVATION_KIND_TO_NAME = {
    "fsc": "scf",
    "wet_snow": "wet_snow",
}
_STATIC_FIELD_KIND_TO_FIELD = {
    "dem": "dem",
    "svf": "svf",
    "srf": "srf",
    "landcover": "landcover",
}
_CLASSIFIED_PANEL_KINDS = {"landcover", "wet_snow", "wet_snow_line"}
_CONTINUOUS_COLORBAR_PANEL_KINDS = {
    "dem",
    "svf",
    "srf",
    "snow_depth",
    "swe",
    "liquid_water_content",
    "uncertainty",
    "fsc",
    "wet_snow_elevation_fraction",
}
_AUTO_TITLE_SOURCE = {
    "open_loop": "Open loop",
    "ensemble_mean": "Ensemble mean",
    "analysis_mean": "Posterior mean",
    "posterior": "Posterior",
    "prior_probability": "Prior probability",
    "posterior_probability": "Posterior probability",
    "increment": "Increment",
    "analysis_increment": "posterior - prior",
    "open_loop_binary": "Open-loop snow cover",
}
_AUTO_TITLE_KIND = {
    "overview": "Overview map",
    "roi": "Region of interest",
    "hillshade": "Hillshade",
    "dem": "Digital elevation model",
    "svf": "Sky view factor",
    "srf": "Snow redistribution factor",
    "landcover": "Landcover",
    "snow_depth": "snow depth",
    "swe": "SWE",
    "liquid_water_content": "liquid water content",
    "fsc": "Sentinel-2 FSC",
    "uncertainty": "observation uncertainty",
    "wet_snow": "Sentinel-1 wet snow fraction (WSF)",
    "wet_snow_line": "Wet snow line altitude (WSLA)",
    "wet_snow_elevation_fraction": "elevation-band WSF",
}
