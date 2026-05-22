import os

# Project root directory configuration
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Dynamic directory creation to guarantee directory structure
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Datasets
MASTER_FUSED_FILE = os.path.join(DATA_PROCESSED_DIR, "delhi_2024_master_fused.csv")
ENGINEERED_DATA_FILE = os.path.join(DATA_PROCESSED_DIR, "delhi_2024_engineered.csv")
SAFETY_METRICS_FILE = os.path.join(DATA_PROCESSED_DIR, "safety_evaluation_metrics.csv")

# Model Serialized Paths
INPUT_SCALER_PATH = os.path.join(MODELS_DIR, "input_scaler.joblib")
TARGET_SCALER_PATH = os.path.join(MODELS_DIR, "target_scaler.joblib")
BEST_GRU_MODEL_PATH = os.path.join(MODELS_DIR, "best_gru_model.pt")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

# Features to use
FEATURE_COLS = [
    "airport_temp", "airport_dew", "airport_wind_speed", "airport_wind_dir", "airport_slp",
    "urban_temp", "urban_dew", "urban_wind_speed", "urban_wind_dir", "urban_slp", "urban_visibility",
    "rural_temp", "rural_dew", "rural_wind_speed", "rural_wind_dir", "rural_slp", "rural_visibility",
    "AOD_500nm", "AOD_440nm", "AOD_675nm", "440-870_Angstrom_Exponent",
    "airport_dpd", "airport_rh", "airport_wsi", "airport_wind_u", "airport_wind_v",
    "urban_dpd", "urban_rh", "urban_wsi", "urban_wind_u", "urban_wind_v",
    "rural_dpd", "rural_rh", "rural_wsi", "rural_wind_u", "rural_wind_v",
    "asep", "spatial_grad_urban_airport", "spatial_grad_rural_airport",
    "hour", "month", "day_of_year", "hour_sin", "hour_cos", "month_sin", "month_cos", "day_sin", "day_cos"
]

TARGET_COL = "airport_visibility"

# Modeling Hyperparameters
WINDOW_SIZE = 24
HORIZON = 6

# Random Forest parameters
RF_PARAMS = {
    "n_estimators": 30,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": 1
}

# XGBoost parameters
XGB_PARAMS = {
    "n_estimators": 50,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": 1,
    "objective": "reg:squarederror"
}

# PyTorch GRU Neural parameters
GRU_PARAMS = {
    "hidden_dim": 32,
    "num_layers": 1,
    "epochs": 15,
    "batch_size": 32,
    "lr": 0.001
}
