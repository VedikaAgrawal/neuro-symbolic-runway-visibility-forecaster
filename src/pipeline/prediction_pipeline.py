import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
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
from src.utils import log_prediction_to_mongo, download_from_s3


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
        # Ensure model folder exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Check and download if necessary
        model_files = [
            ("input_scaler.joblib", INPUT_SCALER_PATH),
            ("target_scaler.joblib", TARGET_SCALER_PATH),
            ("best_gru_model.pt", BEST_GRU_MODEL_PATH)
        ]
        for h in range(HORIZON):
            model_files.append((f"rf_model_h{h}.joblib", os.path.join(self.models_dir, f"rf_model_h{h}.joblib")))
            model_files.append((f"xgb_model_h{h}.joblib", os.path.join(self.models_dir, f"xgb_model_h{h}.joblib")))
            
        for s3_key, local_path in model_files:
            if not os.path.exists(local_path):
                print(f"[PredictionPipeline] 📥 Model file {local_path} not found. Attempting S3 download...")
                download_from_s3(f"models/{s3_key}", local_path)
                
        # Load standardizers
        if not os.path.exists(INPUT_SCALER_PATH) or not os.path.exists(TARGET_SCALER_PATH):
            raise FileNotFoundError("🚨 Failed to load input/target scalers. Please run the training pipeline first.")
            
        self.scaler = joblib.load(INPUT_SCALER_PATH)
        self.target_scaler = joblib.load(TARGET_SCALER_PATH)
        
        # Load RF models
        self.rf_models = {}
        for h in range(HORIZON):
            rf_path = os.path.join(self.models_dir, f"rf_model_h{h}.joblib")
            if os.path.exists(rf_path):
                self.rf_models[h] = joblib.load(rf_path)
            else:
                raise FileNotFoundError(f"🚨 Missing Random Forest model for horizon t+{h+1}h: {rf_path}")
                
        # Load XGBoost models
        self.xgb_models = {}
        for h in range(HORIZON):
            xgb_path = os.path.join(self.models_dir, f"xgb_model_h{h}.joblib")
            if os.path.exists(xgb_path):
                self.xgb_models[h] = joblib.load(xgb_path)
            else:
                raise FileNotFoundError(f"🚨 Missing XGBoost model for horizon t+{h+1}h: {xgb_path}")

        # Dynamic imports to avoid macOS OpenMP / Z3 / Torch threading segmentation fault.
        # Importing torch/z3 after joblib.load of XGBoost avoids the crash.
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
        print("[PredictionPipeline] ✅ All models, standardizers, and formal solvers loaded successfully!")
        
    def get_test_timestamps(self, limit=100):
        """
        Fetches a subset of winter timestamps from the test set for the dropdown UI.
        Prioritizes hours with low visibility/fog.
        """
        if not os.path.exists(self.engineered_file):
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

    def get_sequence_by_timestamp(self, timestamp_str):
        """
        Loads the 24-hour sequence ending at the selected timestamp.
        """
        if not os.path.exists(self.engineered_file):
            raise FileNotFoundError(f"Engineered dataset not found: {self.engineered_file}")
            
        df = pd.read_csv(self.engineered_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        target_time = pd.to_datetime(timestamp_str)
        idx_list = df[df['timestamp'] == target_time].index
        if len(idx_list) == 0:
            raise ValueError(f"Timestamp '{timestamp_str}' not found in the dataset.")
            
        idx = idx_list[0]
        if idx < WINDOW_SIZE - 1:
            raise ValueError(f"Insufficient historical sequence before {timestamp_str}. Requires at least 24 hours.")
            
        # Get 24-hour slice
        seq_df = df.iloc[idx - WINDOW_SIZE + 1 : idx + 1].copy()
        return seq_df

    def re_engineer_last_step(self, seq_df, overrides):
        """
        Applies manual parameter modifications to the last hour of the sequence (index 23)
        and re-calculates all thermodynamic, kinematic, and cyclic aerosol features.
        """
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

    def run_inference(self, timestamp_str, overrides=None):
        """
        Retrieves sequence, overrides last observation, scales, runs RF/XGB/GRU models,
        audits with Z3 guardrails, and logs telemetry.
        """
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
        
        log_prediction_to_mongo(
            input_data=input_summary,
            raw_forecasts=rf_forecasts,
            verified_forecasts=verified_forecasts,
            status=z3_status,
            violations=all_violated_rules
        )
        
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

if __name__ == "__main__":
    pipeline = PredictionPipeline()
    timestamps = pipeline.get_test_timestamps(5)
    if len(timestamps) > 0:
        test_ts = timestamps[0]
        print(f"🔬 Testing inference for timestamp: {test_ts}")
        res = pipeline.run_inference(test_ts)
        print(json.dumps(res, indent=4))
        
        # Test an UNSAT condition by overriding values:
        # Dry air (RH < 45% or DPD > 12°C) but we predict visibility of 400m (which is impossible)
        print(f"\n🔬 Testing UNSAT condition override (Dry air but manually forcing low visibility)...")
        # Let's set a dry air scenario (Temp=25, Dew=5, so DPD = 20, which triggers Rule 1)
        # We override the variables in the last step
        overrides = {
            "airport_temp": 25.0,
            "airport_dew": 5.0,
            "airport_wind_speed": 3.0,
            "airport_wind_dir": 180.0,
            "AOD_500nm": 0.5,
            "AOD_440nm": 0.4
        }
        res_unsat = pipeline.run_inference(test_ts, overrides=overrides)
        # We manually modify the raw RF prediction in res_unsat output so we can verify if the guardrail catches it.
        # But run_inference does it internally, because RF models might predict whatever. If we force Temp=25 and Dew=5,
        # RH is 27.5% and DPD is 20.0. The guardrail asserts that visibility MUST be >= 800m.
        # Let's print the actual result:
        print(json.dumps(res_unsat, indent=4))
    else:
        print("⚠️ No timestamps available for inference testing.")
