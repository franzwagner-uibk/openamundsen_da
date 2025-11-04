INPUT_DATA = "input_data"
GRIDS = "grids"
METEO = "meteo"
DIR = "dir"
RESULTS_DIR = "results_dir"
START_DATE = "start_date"
END_DATE = "end_date"
LOG_LEVEL = "log_level"
ENVIRONMENT = "environment"

# Environment variables we export/apply from project.yml or sensible defaults
ENV_VARS_EXPORT = (
    "GDAL_DATA",
    "PROJ_LIB",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
