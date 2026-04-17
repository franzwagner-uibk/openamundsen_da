# Project Map Benchmark Inputs

The image-regression test for the shipped Rofental project maps can render the
example end-to-end when these local benchmark inputs are present:

- `da_output_grids.nc`
- `CNTR_BN_01M_2020_3857.geojson`
- `CNTR_RG_01M_2020_3857.geojson`
- `CNTR_LB_2020_3857.geojson`

They are intentionally not committed because they are large generated/downloaded
artifacts. To refresh them locally, copy:

- `examples/rofental/projects/project_2022_2023/results/grids/da_output_grids.nc`
- the overview GeoJSONs downloaded into `examples/rofental/env/`

into this directory before running the image-regression test.
