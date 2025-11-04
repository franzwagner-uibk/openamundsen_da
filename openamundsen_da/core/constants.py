INPUT_DATA = "input_data"
GRIDS = "grids"
METEO = "meteo"
DIR = "dir"
RESULTS_DIR = "results_dir"
START_DATE = "start_date"
END_DATE = "end_date"
LOG_LEVEL = "log_level"
ENVIRONMENT = "environment"

# Environment variables to export/apply from project.yml or sensible defaults
ENV_VARS_EXPORT = (
    "GDAL_DATA",
    "PROJ_LIB",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Member artifacts
MEMBER_LOG_REL = ("logs", "member.log")
MEMBER_MANIFEST = "member_run.json"

# Ensemble naming and layout (used by prior builder and launcher)
ENSEMBLE_PRIOR = "prior"
MEMBER_PREFIX = "member_"
OPEN_LOOP = "open_loop"

# Default meteo column names (openAMUNDSEN station CSV schema)
DEFAULT_TIME_COL = "date"
DEFAULT_TEMP_COL = "temp"
DEFAULT_PRECIP_COL = "precip"

# Data assimilation config blocks and keys (project.yml)
DA_BLOCK = "data_assimilation"
DA_PRIOR_BLOCK = "prior_forcing"
DA_ENSEMBLE_SIZE = "ensemble_size"
DA_RANDOM_SEED = "random_seed"
DA_SIGMA_T = "sigma_t"
DA_MU_P = "mu_p"
DA_SIGMA_P = "sigma_p"
