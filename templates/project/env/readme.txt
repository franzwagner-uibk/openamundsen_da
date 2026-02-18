expected files:
- roi.gpkg: ROI polygon(s) in model CRS (preferred default vector source).
- subdomains.gpkg: optional multi-feature regions for sub-domain mode.

openAMUNDSEN-DA uses `grids/roi_<domain>_<resolution>.asc` as canonical ROI mask and auto-generates it from the vector source when missing.

