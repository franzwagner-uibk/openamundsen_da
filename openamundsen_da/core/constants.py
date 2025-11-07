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

# Station metadata filename (CSV schema used by openAMUNDSEN)
STATIONS_CSV = "stations.csv"

# Data assimilation config blocks and keys (project.yml)
DA_BLOCK = "data_assimilation"
DA_PRIOR_BLOCK = "prior_forcing"
DA_ENSEMBLE_SIZE = "ensemble_size"
DA_RANDOM_SEED = "random_seed"
DA_SIGMA_T = "sigma_t"
DA_MU_P = "mu_p"
DA_SIGMA_P = "sigma_p"

# Observation processing (satellite SCF)
OBS_DIR_NAME = "obs"
SCF_BLOCK = "scf"
SCF_NDSI_THRESHOLD = "ndsi_threshold"
SCF_REGION_ID_FIELD = "region_id_field"

# MODIS MOD10A1 (Collection 6/6.1) processing
MOD10A1_PRODUCT = "MOD10A1"
MOD10A1_SDS_NAME = "NDSI_Snow_Cover"

# H(x) model SCF operator config
HOFX_BLOCK = "h_of_x"
HOFX_METHOD = "method"
HOFX_VARIABLE = "variable"
HOFX_PARAMS = "params"
HOFX_PARAM_H0 = "h0"
HOFX_PARAM_K = "k"

# Variable identifiers
VAR_HS = "hs"
VAR_SWE = "swe"

# Likelihood / Resampling config blocks (project.yml)
LIKELIHOOD_BLOCK = "likelihood"
LIK_OBS_SIGMA = "obs_sigma"
LIK_USE_BINOMIAL = "use_binomial"
LIK_SIGMA_FLOOR = "sigma_floor"
LIK_SIGMA_CLOUD_SCALE = "sigma_cloud_scale"
LIK_MIN_SIGMA = "min_sigma"

RESAMPLING_BLOCK = "resampling"
RESAMPLING_ALGORITHM = "algorithm"  # multinomial|systematic|stratified (we implement systematic)
RESAMPLING_ESS_THRESHOLD = "ess_threshold"

REJUVENATION_BLOCK = "rejuvenation"
REJ_SIGMA_T = "sigma_t"
REJ_SIGMA_P = "sigma_p"

# Logging format (green timestamp | level | message)
LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}"
