import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import traceback

# Ensure project root is in the path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import (
    ENGINEERED_DATA_FILE, FEATURE_COLS, TARGET_COL,
    INPUT_SCALER_PATH, TARGET_SCALER_PATH, BEST_GRU_MODEL_PATH,
    WINDOW_SIZE, HORIZON, MODELS_DIR
)
from src.utils import log_prediction_to_mongo, download_from_s3, pipeline_logger

logger = pipeline_logger

class PredictionPipeline:
    def __init__(self):
        self.engineered_file = ENGINEERED_DATA_FILE
        self.models_dir = MODELS_DIR
        self._ensure_models_loaded()
        
    def _ensure_models_loaded(self):
        """
        Loads the inputs scalers, target scalers, and all multi-horizon and sequential models.
        Downloads from S3 if missing and S3 credentials are set.
        """
        logger.info("🎬 [PredictionPipeline] Loading model assets and neural weights...")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Check and download engineered dataset if missing
        if not os.path.exists(self.engineered_file):
            logger.warning(f"[PredictionPipeline] Engineered dataset {self.engineered_file} not found locally. Attempting remote AWS S3 download...")
            try:
                os.makedirs(os.path.dirname(self.engineered_file), exist_ok=True)
                download_from_s3("data/delhi_2024_engineered.csv", self.engineered_file)
            except Exception as data_dl_err:
                logger.error(f"[PredictionPipeline] Failed to download engineered dataset from S3: {data_dl_err}", exc_info=True)
        
        # Check and download if necessary
        model_files = [
            ("input_scaler.joblib", INPUT_SCALER_PATH),
            ("target_scaler.joblib", TARGET_SCALER_PATH),
            ("best_gru_model.pt", BEST_GRU_MODEL_PATH)
        ]
        for h in range(HORIZON):
            model_files.append((f"rf_model_h{h}.joblib", os.path.join(self.models_dir, f"rf_model_h{h}.joblib")))
            model_files.append((f"xgb_model_h{h}.joblib", os.path.join(self.models_dir, f"xgb_model_h{h}.joblib")))
            
        try:
            for s3_key, local_path in model_files:
                if not os.path.exists(local_path):
                    logger.warning(f"[PredictionPipeline] Asset {local_path} not found locally. Attempting remote AWS S3 download...")
                    download_from_s3(f"models/{s3_key}", local_path)
        except Exception as dl_err:
            logger.error(f"[PredictionPipeline] Error downloading assets from remote storage: {dl_err}", exc_info=True)
                
        # Load standardizers
        try:
            if not os.path.exists(INPUT_SCALER_PATH) or not os.path.exists(TARGET_SCALER_PATH):
                logger.error("[PredictionPipeline] ❌ Input or target standardizer is missing. Validation block failed.")
                raise FileNotFoundError("🚨 Failed to load input/target scalers. Please run the training pipeline first.")
                
            self.scaler = joblib.load(INPUT_SCALER_PATH)
            self.target_scaler = joblib.load(TARGET_SCALER_PATH)
        except Exception as sc_err:
            logger.error(f"[PredictionPipeline] Failed to load scaler standardizers: {sc_err}", exc_info=True)
            raise sc_err
        
        # Load RF models
        try:
            self.rf_models = {}
            for h in range(HORIZON):
                rf_path = os.path.join(self.models_dir, f"rf_model_h{h}.joblib")
                if os.path.exists(rf_path):
                    self.rf_models[h] = joblib.load(rf_path)
                else:
                    raise FileNotFoundError(f"🚨 Missing Random Forest model for horizon t+{h+1}h: {rf_path}")
        except Exception as rf_load_ex:
            logger.error(f"[PredictionPipeline] RF model loading failed: {rf_load_ex}", exc_info=True)
            raise rf_load_ex
                 
        # Load XGBoost models
        try:
            self.xgb_models = {}
            for h in range(HORIZON):
                xgb_path = os.path.join(self.models_dir, f"xgb_model_h{h}.joblib")
                if os.path.exists(xgb_path):
                    self.xgb_models[h] = joblib.load(xgb_path)
                else:
                    raise FileNotFoundError(f"🚨 Missing XGBoost model for horizon t+{h+1}h: {xgb_path}")
        except Exception as xgb_load_ex:
            logger.error(f"[PredictionPipeline] XGBoost model loading failed: {xgb_load_ex}", exc_info=True)
            raise xgb_load_ex

        # Dynamic imports to avoid macOS OpenMP / Z3 / Torch threading segmentation fault.
        # Importing torch/z3 after joblib.load of XGBoost avoids the crash.
        try:
            import torch
            from src.components.model_training import VisibilityGRUForecaster
            from scripts.z3_verification import SymbolicGuardrail
                    
            # Load GRU Sequence Model
            input_dim = len(FEATURE_COLS)
            self.gru_model = VisibilityGRUForecaster(input_dim, 32, num_layers=1, output_dim=HORIZON)
            if os.path.exists(BEST_GRU_MODEL_PATH):
                self.gru_model.load_state_dict(torch.load(BEST_GRU_MODEL_PATH, map_location=torch.device('cpu')))
                self.gru_model.eval()
            else:
                raise FileNotFoundError(f"🚨 Missing GRU sequence model: {BEST_GRU_MODEL_PATH}")
                
            # Initialize Z3 Guardrail
            self.guardrail = SymbolicGuardrail()
            logger.info("[PredictionPipeline] ✅ All models, standardizers, and formal solvers loaded successfully!")
        except Exception as dynamic_load_ex:
            logger.error(f"[PredictionPipeline] Failed to load PyTorch GRU / Z3 verification solvers dynamically: {dynamic_load_ex}", exc_info=True)
            raise dynamic_load_ex
        
    def get_test_timestamps(self, limit=100):
        """
        Fetches a subset of winter timestamps from the test set for the dropdown UI.
        Prioritizes hours with low visibility/fog.
        """
        logger.info("[PredictionPipeline] Fetching foggy test timestamps for dropdown UI...")
        try:
            if not os.path.exists(self.engineered_file):
                logger.warning(f"Engineered dataset not found at {self.engineered_file}. Returning empty dropdown list.")
                return []
            df = pd.read_csv(self.engineered_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter for test partition (Nov-Dec 2024)
            test_df = df[df['timestamp'] >= pd.Timestamp('2024-11-01 00:00:00')].copy()
            
            # Sort by visibility ascending to show challenging fog conditions first, then sample
            foggy_df = test_df[test_df['airport_visibility'] < 1500].sort_values('timestamp').head(limit // 2)
            random_df = test_df.sample(n=min(limit // 2, len(test_df) - len(foggy_df)), random_state=42)
            
            combined_df = pd.concat([foggy_df, random_df]).drop_duplicates().sort_values('timestamp')
            return [t.strftime('%Y-%m-%d %H:%M:%S') for t in combined_df['timestamp']]
        except Exception as list_ex:
            logger.error(f"[PredictionPipeline] Failed to extract list of test timestamps: {list_ex}", exc_info=True)
            return []

    def get_sequence_by_timestamp(self, timestamp_str):
        """
        Loads the 24-hour sequence ending at the selected timestamp.
        """
        logger.info(f"[PredictionPipeline] Constructing historical input slice ending at: {timestamp_str}")
        if not os.path.exists(self.engineered_file):
            logger.error(f"Engineered dataset missing during query: {self.engineered_file}")
            raise FileNotFoundError(f"Engineered dataset not found: {self.engineered_file}")
            
        try:
            df = pd.read_csv(self.engineered_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            target_time = pd.to_datetime(timestamp_str)
            idx_list = df[df['timestamp'] == target_time].index
            if len(idx_list) == 0:
                logger.error(f"Selected timestamp {timestamp_str} not found in the master engineered dataset.")
                raise ValueError(f"Timestamp '{timestamp_str}' not found in the dataset.")
                
            idx = idx_list[0]
            if idx < WINDOW_SIZE - 1:
                logger.error(f"Insufficient history before timestamp {timestamp_str}. Available idx: {idx}, required window size: {WINDOW_SIZE}")
                raise ValueError(f"Insufficient historical sequence before {timestamp_str}. Requires at least 24 hours.")
                
            # Get 24-hour slice
            seq_df = df.iloc[idx - WINDOW_SIZE + 1 : idx + 1].copy()
            return seq_df
        except Exception as seq_ex:
            logger.error(f"[PredictionPipeline] Error fetching sequence slice: {seq_ex}", exc_info=True)
            raise seq_ex

    def re_engineer_last_step(self, seq_df, overrides):
        """
        Applies manual parameter modifications to the last hour of the sequence (index 23)
        and re-calculates all thermodynamic, kinematic, and cyclic aerosol features.
        """
        logger.info("[PredictionPipeline] Applying manual parameter overrides to the final sequence hour...")
        try:
            df_mod = seq_df.copy().reset_index(drop=True)
            last_idx = len(df_mod) - 1
            
            # Apply overrides
            for col, val in overrides.items():
                if col in df_mod.columns:
                    df_mod.at[last_idx, col] = float(val)
                    
            # Recompute dependent meteorological features for the modified step
            # 1. Dew Point Depression (DPD)
            df_mod.at[last_idx, 'airport_dpd'] = max(
                df_mod.at[last_idx, 'airport_temp'] - df_mod.at[last_idx, 'airport_dew'], 0.0
            )
            # 2. Relative Humidity (Magnus-Tetens)
            t = df_mod.at[last_idx, 'airport_temp']
            td = df_mod.at[last_idx, 'airport_dew']
            rh = 100.0 * np.exp((17.625 * td) / (243.04 + td) - (17.625 * t) / (243.04 + t))
            df_mod.at[last_idx, 'airport_rh'] = np.clip(rh, 0.0, 100.0)
            
            # 3. Wind Stagnation Index (WSI)
            df_mod.at[last_idx, 'airport_wsi'] = int(df_mod.at[last_idx, 'airport_wind_speed'] < 1.5)
            
            # 4. Wind Vectors (U/V)
            rad = df_mod.at[last_idx, 'airport_wind_dir'] * np.pi / 180.0
            df_mod.at[last_idx, 'airport_wind_u'] = df_mod.at[last_idx, 'airport_wind_speed'] * np.sin(rad)
            df_mod.at[last_idx, 'airport_wind_v'] = df_mod.at[last_idx, 'airport_wind_speed'] * np.cos(rad)
            
            # 5. Aerosol Scattering Extinction Proxy (ASEP)
            if df_mod.at[last_idx, 'AOD_440nm'] != 0.0:
                df_mod.at[last_idx, 'asep'] = df_mod.at[last_idx, 'AOD_500nm'] / df_mod.at[last_idx, 'AOD_440nm']
                
            # 6. Spatial Gradients
            df_mod.at[last_idx, 'spatial_grad_urban_airport'] = (
                df_mod.at[last_idx, 'urban_temp'] - df_mod.at[last_idx, 'airport_temp']
            )
            df_mod.at[last_idx, 'spatial_grad_rural_airport'] = (
                df_mod.at[last_idx, 'rural_temp'] - df_mod.at[last_idx, 'airport_temp']
            )
            
            return df_mod
        except Exception as override_ex:
            logger.error(f"[PredictionPipeline] Override re-engineering calculations failed: {override_ex}", exc_info=True)
            raise override_ex

    def run_inference(self, timestamp_str, overrides=None):
        """
        Retrieves sequence, overrides last observation, scales, runs RF/XGB/GRU models,
        audits with Z3 guardrails, and logs telemetry.
        """
        logger.info(f"🔮 [PredictionPipeline] Initializing inference run for timestamp: {timestamp_str}")
        
        try:
            # Load historical sequence
            seq_df = self.get_sequence_by_timestamp(timestamp_str)
            
            # Apply user overrides
            if overrides:
                seq_df = self.re_engineer_last_step(seq_df, overrides)
                
            # Get raw weather values of the current (last) hour for Z3 audit
            last_row = seq_df.iloc[-1]
            rh_val = float(last_row['airport_rh'])
            dpd_val = float(last_row['airport_dpd'])
            wsi_val = float(last_row['airport_wsi'])
            aod_val = float(last_row['AOD_500nm'])
            actual_visibility = float(last_row['airport_visibility'])
            
            # Prepare feature matrix
            X_seq = seq_df[FEATURE_COLS].values  # shape: (24, 48)
            
            # Scale input features
            X_seq_scaled = self.scaler.transform(X_seq)
            
            # Flatten for Tree Regressors
            X_seq_flat = X_seq_scaled.reshape(1, -1)  # shape: (1, 1152)
            
            # 1. Random Forest Predictions
            rf_forecasts = []
            for h in range(HORIZON):
                pred = self.rf_models[h].predict(X_seq_flat)[0]
                rf_forecasts.append(float(pred))
                
            # 2. XGBoost Predictions
            xgb_forecasts = []
            for h in range(HORIZON):
                pred = self.xgb_models[h].predict(X_seq_flat)[0]
                xgb_forecasts.append(float(pred))
                
            # 3. Deep GRU Prediction
            import torch
            X_tensor = torch.tensor(X_seq_scaled, dtype=torch.float32).unsqueeze(0)  # shape: (1, 24, 48)
            with torch.no_grad():
                gru_scaled_pred = self.gru_model(X_tensor).numpy()[0]  # shape: (6,)
                
            gru_forecasts = []
            for h in range(HORIZON):
                pred = self.target_scaler.inverse_transform([[gru_scaled_pred[h]]])[0, 0]
                gru_forecasts.append(float(pred))
                
            # 4. Z3 Formal Verification audit on Random Forest Predictions
            logger.info("Executing MS Research Z3 verification filter checks across forecast horizons...")
            verified_forecasts = []
            violation_counts = 0
            all_violated_rules = []
            z3_status = "SAT"
            
            for h in range(HORIZON):
                raw_pred = rf_forecasts[h]
                is_sat, verified_val, status, rules = self.guardrail.verify_prediction(
                    raw_pred=raw_pred, rh_val=rh_val, dpd_val=dpd_val, wsi_val=wsi_val, aod_val=aod_val
                )
                verified_forecasts.append(float(verified_val))
                if not is_sat:
                    violation_counts += 1
                    z3_status = "UNSAT"
                    for rule in rules:
                        if rule not in all_violated_rules:
                            all_violated_rules.append(rule)
                            
            logger.info(f"Verification check completed. Status={z3_status}, Violated axioms counts: {violation_counts}")
                            
            # Save telemetry to MongoDB (with local file fallback)
            input_summary = {
                "timestamp": timestamp_str,
                "airport_temp": float(last_row["airport_temp"]),
                "airport_dew": float(last_row["airport_dew"]),
                "airport_wind_speed": float(last_row["airport_wind_speed"]),
                "airport_wind_dir": float(last_row["airport_wind_dir"]),
                "airport_rh": rh_val,
                "airport_dpd": dpd_val,
                "airport_wsi": wsi_val,
                "AOD_500nm": aod_val,
                "AOD_440nm": float(last_row["AOD_440nm"])
            }
            
            try:
                log_prediction_to_mongo(
                    input_data=input_summary,
                    raw_forecasts=rf_forecasts,
                    verified_forecasts=verified_forecasts,
                    status=z3_status,
                    violations=all_violated_rules
                )
            except Exception as telemetry_ex:
                logger.warning(f"Telemetry persistence encountered an exception (continuing execution): {telemetry_ex}")
            
            return {
                "timestamp": timestamp_str,
                "actual_visibility": actual_visibility,
                "meteorological_conditions": {
                    "relative_humidity": round(rh_val, 2),
                    "dew_point_depression": round(dpd_val, 2),
                    "wind_stagnation_index": int(wsi_val),
                    "aerosol_optical_depth": round(aod_val, 3)
                },
                "predictions": {
                    "random_forest": [round(v, 1) for v in rf_forecasts],
                    "xgboost": [round(v, 1) for v in xgb_forecasts],
                    "deep_gru": [round(v, 1) for v in gru_forecasts],
                    "z3_verified": [round(v, 1) for v in verified_forecasts]
                },
                "z3_audit": {
                    "status": z3_status,
                    "violations_count": violation_counts,
                    "violated_rules": all_violated_rules
                }
            }
        except Exception as inf_ex:
            logger.error(f"❌ [PredictionPipeline] Real-time inference failed: {inf_ex}", exc_info=True)
            raise inf_ex

if __name__ == "__main__":
    pipeline = PredictionPipeline()
    timestamps = pipeline.get_test_timestamps(5)
    if len(timestamps) > 0:
        test_ts = timestamps[0]
        logger.info(f"🔬 Testing inference for timestamp: {test_ts}")
        res = pipeline.run_inference(test_ts)
        logger.info(json.dumps(res, indent=4))
        
        # Test an UNSAT condition by overriding values:
        logger.info(f"\n🔬 Testing UNSAT condition override (Dry air but manually forcing low visibility)...")
        overrides = {
            "airport_temp": 25.0,
            "airport_dew": 5.0,
            "airport_wind_speed": 3.0,
            "airport_wind_dir": 180.0,
            "AOD_500nm": 0.5,
            "AOD_440nm": 0.4
        }
        res_unsat = pipeline.run_inference(test_ts, overrides=overrides)
        logger.info(json.dumps(res_unsat, indent=4))
    else:
        logger.warning("⚠️ No timestamps available for inference testing.")
