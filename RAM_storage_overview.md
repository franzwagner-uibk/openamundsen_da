# Memory and Storage Sizing for openAMUNDSEN (energy-balance, multilayer snow)

## What drives RAM
- State arrays allocated once per grid cell at init: `openamundsen/statevars.py` (per-category `add_variable`, created in `StateVariableManager.initialize`). Defaults use float64, some ints/bools.
- Snow multilayer (4 layers here): allocations in `openamundsen/modules/snow/multilayermodel.py` lines ~15-45 (thickness, density, ice/liquid water, temp, thermal conductivity, heat capacity per layer) plus shared snow fields.
- Soil (default 4 layers): `openamundsen/statevars.py` soil section lines ~310-360.
- Optional modules that add arrays: canopy (`modules/canopy/canopy.py` adds 24h temp stack with shape [86400/timestep, rows, cols] + 4 snow fields), evapotranspiration (`modules/evapotranspiration/evapotranspiration.py` adds ~22 2-D fields), glaciers (`modules/glacier/glaciermodel.py` adds glacier ID mask), land cover (`openamundsen/landcover.py` adds LAI/land cover grids).
- Gridded outputs can preallocate full in-memory datasets if using NetCDF without Dask (`openamundsen/fileio/griddedoutput.py` lines ~315-487); warning for >4 GiB at line ~435. GeoTIFF writes stream directly, minimal extra RAM.
- Forcing data kept as xarray in memory (`openamundsen/forcing.py`), sized stations × timesteps × vars × 8 bytes. Snow updates temporarily copy 3-D snow arrays (`multilayermodel.py` lines ~572-579), so peak RAM can be ~1.5–2× steady state.

## Per-cell footprint (energy balance, multilayer snow=4, soil=4, canopy/ET off)
- ~124 float64 fields + 1 int per cell ≈ 1000 bytes/cell for state.
- Grid metadata (X/Y coords, ROI, masks) adds ~2 float grids + 1 bool grid ≈ ~20–30 bytes/cell. Rule of thumb: 1.02 kB/cell steady.
- Additions if enabled: canopy adds `(86400/timestep_seconds + 4)` floats/cell (at 3h: ~12 floats ⇒ +96 bytes); ET adds ~22 floats (+176 bytes); each extra snow layer adds 7 floats (+56 bytes); each extra soil layer adds 7 floats (+56 bytes); cryolayers add ~11 floats vs default.

## Example A: Euregio setup (area 260,000 km², 3h timestep, multilayer 4, energy balance)
Cell counts (A = 2.6e11 m²):
- 50 m (2,500 m²/cell): ~1.04e8 cells → state ≈ 97 GiB; with overhead/temps plan ~110–130 GiB.
- 100 m (10,000 m²/cell): ~2.6e7 cells → state ≈ 24 GiB; plan ~28–35 GiB.
- 500 m (250,000 m²/cell): ~1.04e6 cells → state ≈ 1 GiB; plan ~1.2–1.6 GiB.
Forcing (350 stations, 6 vars, 1 year @3h): ~50 MB (negligible). NetCDF gridded outputs can dominate RAM if Dask is off (e.g., daily 3-D write over 50 m grid ≈ 152 GB in memory); prefer Dask or coarse outputs.

## Example B: AOI 1,500 km², 100 m, 3h outputs as GeoTIFF per timestep (lwc, SWE, snow depth)
- Cells: 1.5e9 / 1e4 ≈ 150,000.
- State RAM: ~150 MB steady; allow ~200 MB with temp copies.
- Per timestep disk write (float32 GeoTIFF): ~0.57 MB per grid → ~1.7 MB for 3 vars. Daily (8 steps): ~14 MB; 1 year: ~5.2 GB. Compression (LZW/DEFLATE) can roughly halve.
- RAM overhead for writing is negligible because arrays already in memory; I/O bound.

## Ensemble sizing (season pipeline, 30 members)
- If runs are sequential: RAM remains per-member numbers above; disk multiplies by 30. Example B outputs: ~5.2 GB/year → ~156 GB for 30 members; plus point/diagnostic outputs as configured.
- If runs are concurrent: multiply RAM by concurrency. Example B with 5 members in parallel: ~5 × 200 MB ≈ 1 GB; Euregio 100 m with 2 in parallel: ~2 × 30 GiB ≈ 60 GiB.
- For gridded NetCDF outputs without Dask, avoid parallel runs or enable Dask to keep memory bounded.

## How to estimate for other cases
1) Cells = AOI_area_m² / resolution². 
2) Base RAM ≈ cells × 1.02 kB (adjust +56 bytes per added snow/soil layer; +~176 bytes if ET; canopy add ~96 bytes at 3h).
3) Peak RAM ≈ 1.5–2× base to cover temporary copies and small buffers.
4) Forcing RAM ≈ stations × timesteps × vars × 8 bytes (float64); usually small.
5) Output storage: 
   - GeoTIFF per timestep: vars × cells × 4 bytes × steps. 
   - NetCDF (no Dask): same as above but may allocate full time dimension in RAM; enable Dask or write coarser/aggregated.

## Practical guidance
- For large domains (≥20 GiB RAM), prefer 100–500 m or tile the domain; use ROI to skip ocean/void but note arrays still allocate full grid.
- Install Dask to avoid in-memory NetCDF assembly; otherwise limit gridded outputs or write GeoTIFFs per timestep.
- For ensembles, run members sequentially or limit concurrency; monitor disk (outputs × ensemble size) and clean intermediates.
- Keep timestep ≥1h if canopy is on to avoid large `last_24h_temps` stacks; at 10 min this stack alone adds ~148 floats/cell.
