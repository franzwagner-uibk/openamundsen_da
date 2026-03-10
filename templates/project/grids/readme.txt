expected files:
- dem.tif (or similar): domain DEM in model CRS.
- roi_<domain>_<resolution>.asc: canonical ROI mask on the setup grid (required by openAMUNDSEN-data assimilation runs).
- lc_<domain>_<resolution>.asc: land-cover classes used by the model and data assimilation masking.
- other static grids referenced from setup YAML (e.g. soil, glacier/ice IDs if available).

