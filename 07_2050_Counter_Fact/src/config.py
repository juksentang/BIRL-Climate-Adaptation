"""
Configuration for Step 07: 2050 Climate Counterfactual Analysis.

All paths, constants, feature lists, growing seasons, scenarios.
"""

from pathlib import Path

# ── Root directories ──
STEP07_DIR = Path(__file__).resolve().parent.parent
STEP04_DIR = STEP07_DIR.parent / "04_Env_Model"
STEP06_DIR = STEP07_DIR.parent / "06_BIRL_MCMC"
NIGERIA_DIR = STEP07_DIR.parent.parent / "Nigeria"

# ── Upstream files (read-only) ──
MODEL_MU_PATH = STEP04_DIR / "model_mu.txt"
MODEL_SIGMA_PATH = STEP04_DIR / "model_sigma.txt"
ENV_METRICS_PATH = STEP04_DIR / "env_model_metrics.json"
ENV_OUTPUT_PATH = STEP04_DIR / "env_model_output.npz"

POSTERIOR_PATH = STEP06_DIR / "outputs" / "hier_noalpha" / "posterior.pkl"
BIRL_SAMPLE_PATH = STEP06_DIR / "data" / "birl_sample.parquet"
ACTION_CONFIG_PATH = STEP06_DIR / "data" / "action_space_config.json"

# ── Output directories ──
DATA_DIR = STEP07_DIR / "data"
CMIP6_RAW_DIR = DATA_DIR / "cmip6_raw"
CMIP6_PROC_DIR = DATA_DIR / "cmip6_processed"
RESULTS_DIR = STEP07_DIR / "results"
FIGURES_DIR = STEP07_DIR / "figures"

# ── Environment model constants ──
SIGMA_SCALE = 1.625636    # from env_model_metrics.json (calibration kappa)
Z10 = -1.2816             # norm.ppf(0.10)
Z90 = 1.2816              # norm.ppf(0.90)

# ── BIRL parameter bounds (sigmoid transform) ──
RHO_LO, RHO_HI = 0.1, 5.0
GAMMA_LO, GAMMA_HI = 0.1, 30.0

# ── 36 feature columns (exact order from env_model_metrics.json) ──
FEATURE_COLS = [
    "action_crop",
    "input_intensity",
    "plot_area_GPS",
    "intercropped",
    "plot_owned",
    "irrigated",
    "rainfall_growing_sum_final",
    "rainfall_10yr_mean_final",
    "rainfall_10yr_cv_final",
    "ndvi_preseason_mean_final",
    "ndvi_growing_mean_final",
    "era5_tmean_growing_final",
    "era5_tmax_growing_final",
    "clay_0-5cm",
    "sand_0-5cm",
    "soc_0-5cm",
    "nitrogen_0-5cm",
    "phh2o_0-5cm",
    "elevation_m",
    "slope_deg",
    "hh_size",
    "hh_asset_index",
    "hh_dependency_ratio",
    "age_manager",
    "female_manager",
    "formal_education_manager",
    "livestock",
    "nonfarm_enterprise",
    "hh_electricity_access",
    "travel_time_city_min",
    "urban",
    "conflict_events_25km_12m",
    "conflict_nearest_event_km",
    "country",
    "year",
    "season",
]

CAT_FEATURES = ["action_crop", "country", "season"]

# ── 7 climate features to replace with 2050 values ──
CLIMATE_FEATURES = [
    "rainfall_growing_sum_final",
    "rainfall_10yr_mean_final",
    "rainfall_10yr_cv_final",
    "ndvi_preseason_mean_final",
    "ndvi_growing_mean_final",
    "era5_tmean_growing_final",
    "era5_tmax_growing_final",
]

# Features with multiplicative delta
MULTIPLICATIVE_FEATURES = [
    "rainfall_growing_sum_final",
    "rainfall_10yr_mean_final",
    "rainfall_10yr_cv_final",
]

# Features with additive delta
ADDITIVE_FEATURES = [
    "era5_tmean_growing_final",
    "era5_tmax_growing_final",
]

# Features predicted by NDVI model
NDVI_FEATURES = [
    "ndvi_growing_mean_final",
    "ndvi_preseason_mean_final",
]

# ── Crop order (action_id = crop_idx * 3 + intensity_idx) ──
CROP_ORDER = [
    "maize", "tree_crops", "tubers", "legumes", "sorghum_millet",
    "teff", "other", "rice", "wheat_barley",
]
INTENSITY_ORDER = ["low", "medium", "high"]
N_ACTIONS = 27

# ── GCMs and scenarios ──
GCMS = ["ACCESS-CM2", "MIROC6", "MRI-ESM2-0", "INM-CM5-0", "IPSL-CM6A-LR"]
SSPS = ["ssp245", "ssp585"]
PERIOD_BASELINE = ("2005-01-01", "2023-12-31")
PERIOD_FUTURE = ("2040-01-01", "2060-12-31")

# ── Growing seasons per country (months) ──
GROWING_SEASONS = {
    "Ethiopia":  {"months": [6, 7, 8, 9], "cross_year": False},
    "Malawi":    {"months": [11, 12, 1, 2, 3, 4], "cross_year": True},
    "Mali":      {"months": [6, 7, 8, 9, 10], "cross_year": False},
    "Nigeria":   {"months": [4, 5, 6, 7, 8, 9, 10], "cross_year": False},
    "Tanzania":  {"months": [11, 12, 1, 2, 3, 4, 5], "cross_year": True},
    "Uganda":    {"months": [3, 4, 5, 6], "cross_year": False},
}

# ── 6 countries (alphabetical, matching MCMC posterior order) ──
COUNTRIES = sorted(GROWING_SEASONS.keys())

# ── Policy scenarios ──
# Safety net: floor = gamma + buffer. The government guarantees income at
#   gamma + (median_income - gamma) * coverage, ensuring positive surplus.
#   floor_coverage = 0.5 means the floor is halfway between gamma and median income.
#   Y_effective = max(Y, floor). gamma itself is UNCHANGED.
# Insurance: rho_factor scales effective risk aversion.
SCENARIOS = {
    "S0_current":    {"climate": "baseline", "rho_factor": 1.0, "floor_coverage": 0.0},
    "S1_ssp245":     {"climate": "ssp245",   "rho_factor": 1.0, "floor_coverage": 0.0},
    "S2_ssp585":     {"climate": "ssp585",   "rho_factor": 1.0, "floor_coverage": 0.0},
    "S3_insurance":  {"climate": "ssp585",   "rho_factor": 0.5, "floor_coverage": 0.0},
    "S4_safety_net": {"climate": "ssp585",   "rho_factor": 1.0, "floor_coverage": 0.5},
    "S5_combined":   {"climate": "ssp585",   "rho_factor": 0.5, "floor_coverage": 0.5},
}

# ── Posterior thinning ──
THIN_FACTOR = 4  # use every 4th sample → 3,000 from 12,000
