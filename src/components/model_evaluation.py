import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib
# Use non-interactive backend for server environment
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup sys path to find scripts
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import (
    ENGINEERED_DATA_FILE, FEATURE_COLS, TARGET_COL,
    INPUT_SCALER_PATH, TARGET_SCALER_PATH, BEST_GRU_MODEL_PATH,
    SAFETY_METRICS_FILE, WINDOW_SIZE, HORIZON, MODELS_DIR
)
from src.utils import upload_to_s3, pipeline_logger
from src.components.model_training import VisibilityGRUForecaster
from scripts.z3_verification import SymbolicGuardrail

logger = pipeline_logger

class ModelEvaluation:
    def __init__(self):
        self.engineered_file = ENGINEERED_DATA_FILE
        self.models_dir = MODELS_DIR
        self.metrics_file = SAFETY_METRICS_FILE
        
    def evaluate_classification(self, y_true, y_pred, threshold):
        try:
            y_true_bin = (y_true < threshold).astype(int)
            y_pred_bin = (y_pred < threshold).astype(int)
            
            tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
            fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
            tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
            fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            far_ratio = fp / (tp + fp) if (tp + fp) > 0 else 0.0
            far_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
                'Precision': float(precision),
                'Recall (POD)': float(recall),
                'False Alarm Ratio (FAR)': float(far_ratio),
                'False Alarm Rate (FPR)': float(far_rate),
                'F1-Score': float(f1)
            }
        except Exception as eval_cls_ex:
            logger.error(f"Error evaluating classification performance metrics for threshold {threshold}: {eval_cls_ex}")
            raise eval_cls_ex

    def run_evaluation(self):
        logger.info(f"📊 [ModelEvaluation] Commencing scientific validation metrics and safety evaluations on: {self.engineered_file}")
        if not os.path.exists(self.engineered_file):
            logger.error(f"[ModelEvaluation] Engineered data not found at path: {self.engineered_file}")
            raise FileNotFoundError(f"🚨 Engineered data not found: {self.engineered_file}")
            
        try:
            df = pd.read_csv(self.engineered_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 1. Chronological Splits
            val_mask = (df['timestamp'] >= '2024-09-01 00:00:00') & (df['timestamp'] < '2024-11-01 00:00:00')
            test_mask = df['timestamp'] >= '2024-11-01 00:00:00'
            
            df_val = df[val_mask].reset_index(drop=True)
            df_test = df[test_mask].reset_index(drop=True)
            
            logger.info(f"[ModelEvaluation] Splits: Validation={len(df_val)} records, Test={len(df_test)} records")
            
            # Load Scalers
            if not os.path.exists(INPUT_SCALER_PATH) or not os.path.exists(TARGET_SCALER_PATH):
                logger.error("Standardizer scalers not found. Run model training stage first.")
                raise FileNotFoundError("🚨 Scalers not found. Run training pipeline first.")
                
            scaler = joblib.load(INPUT_SCALER_PATH)
            target_scaler = joblib.load(TARGET_SCALER_PATH)
            
            # Reconstruct sliding windows
            def create_sliding_windows(df_scaled, feature_cols, target_col, window_size=24, horizon=6):
                X_list, y_list, t_list = [], [], []
                max_idx = len(df_scaled) - horizon
                for i in range(window_size, max_idx):
                    t_last_input = df_scaled.iloc[i-1]['timestamp']
                    t_first_target = df_scaled.iloc[i]['timestamp']
                    time_delta = (t_first_target - t_last_input).total_seconds() / 3600.0
                    
                    if abs(time_delta - 1.0) < 0.01:
                        X_slice = df_scaled.iloc[i-window_size:i][feature_cols].values
                        y_slice = df_scaled.iloc[i:i+horizon][target_col].values
                        t_list.append(t_first_target)
                        X_list.append(X_slice)
                        y_list.append(y_slice)
                        
                return np.array(X_list), np.array(y_list), np.array(t_list)

            # Scale features
            df_val_scaled_feat = pd.DataFrame(scaler.transform(df_val[FEATURE_COLS]), columns=FEATURE_COLS)
            df_val_scaled = df_val_scaled_feat.copy()
            df_val_scaled['timestamp'] = df_val['timestamp'].values
            df_val_scaled['airport_visibility'] = df_val['airport_visibility'].values
            
            df_test_scaled_feat = pd.DataFrame(scaler.transform(df_test[FEATURE_COLS]), columns=FEATURE_COLS)
            df_test_scaled = df_test_scaled_feat.copy()
            df_test_scaled['timestamp'] = df_test['timestamp'].values
            df_test_scaled['airport_visibility'] = df_test['airport_visibility'].values
            
            X_val, y_val, t_val = create_sliding_windows(df_val_scaled, FEATURE_COLS, TARGET_COL, WINDOW_SIZE, HORIZON)
            X_test, y_test, t_test = create_sliding_windows(df_test_scaled, FEATURE_COLS, TARGET_COL, WINDOW_SIZE, HORIZON)
            
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            _, y_val_raw, _ = create_sliding_windows(df_val, FEATURE_COLS, TARGET_COL, WINDOW_SIZE, HORIZON)
            _, y_test_raw, _ = create_sliding_windows(df_test, FEATURE_COLS, TARGET_COL, WINDOW_SIZE, HORIZON)
            
            # 2. Reload trained RF models
            logger.info("Deserializing Multi-Horizon Random Forest Regressors...")
            rf_preds_val = np.zeros_like(y_val_raw)
            rf_preds_test = np.zeros_like(y_test_raw)
            for h in range(HORIZON):
                rf_path = os.path.join(self.models_dir, f"rf_model_h{h}.joblib")
                if not os.path.exists(rf_path):
                    raise FileNotFoundError(f"Missing RF model file for horizon {h}: {rf_path}")
                rf = joblib.load(rf_path)
                rf_preds_val[:, h] = rf.predict(X_val_flat)
                rf_preds_test[:, h] = rf.predict(X_test_flat)
                
            # 3. Reload trained XGBoost models
            logger.info("Deserializing Multi-Horizon XGBoost Regressors...")
            xgb_preds_val = np.zeros_like(y_val_raw)
            xgb_preds_test = np.zeros_like(y_test_raw)
            for h in range(HORIZON):
                xgb_path = os.path.join(self.models_dir, f"xgb_model_h{h}.joblib")
                if not os.path.exists(xgb_path):
                    raise FileNotFoundError(f"Missing XGBoost model file for horizon {h}: {xgb_path}")
                xgb_reg = joblib.load(xgb_path)
                xgb_preds_val[:, h] = xgb_reg.predict(X_val_flat)
                xgb_preds_test[:, h] = xgb_reg.predict(X_test_flat)
                
            # 4. Reload GRU model and evaluate
            logger.info("Deserializing best sequential Deep PyTorch GRU neural model weights...")
            if not os.path.exists(BEST_GRU_MODEL_PATH):
                raise FileNotFoundError(f"Missing PyTorch GRU weights: {BEST_GRU_MODEL_PATH}")
            input_dim = len(FEATURE_COLS)
            gru_model = VisibilityGRUForecaster(input_dim, 32, num_layers=1, output_dim=HORIZON)
            gru_model.load_state_dict(torch.load(BEST_GRU_MODEL_PATH, map_location=torch.device('cpu')))
            gru_model.eval()
            
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            
            with torch.no_grad():
                gru_preds_val_scaled = gru_model(X_val_tensor).numpy()
                gru_preds_test_scaled = gru_model(X_test_tensor).numpy()
                
            gru_preds_val = np.zeros_like(gru_preds_val_scaled)
            gru_preds_test = np.zeros_like(gru_preds_test_scaled)
            for h in range(HORIZON):
                gru_preds_val[:, h] = target_scaler.inverse_transform(gru_preds_val_scaled[:, h:h+1]).flatten()
                gru_preds_test[:, h] = target_scaler.inverse_transform(gru_preds_test_scaled[:, h:h+1]).flatten()
                
            # 5. Apply Z3 Symbolic Guardrails to RF predictions
            logger.info("Routing raw Random Forest predictions through Z3 SMT Symbolic Guardrail Filter...")
            guard = SymbolicGuardrail()
            
            def apply_z3_guardrail(df_raw, X_windows, raw_preds):
                output_preds = np.zeros_like(raw_preds)
                for i in range(len(X_windows)):
                    orig_idx = WINDOW_SIZE + i
                    rh_val = df_raw.iloc[orig_idx-1]['airport_rh']
                    dpd_val = df_raw.iloc[orig_idx-1]['airport_dpd']
                    wsi_val = df_raw.iloc[orig_idx-1]['airport_wsi']
                    aod_val = df_raw.iloc[orig_idx-1]['AOD_500nm']
                    
                    for h in range(HORIZON):
                        raw_pred = raw_preds[i, h]
                        _, verified_val, _, _ = guard.verify_prediction(
                            raw_pred=raw_pred, rh_val=rh_val, dpd_val=dpd_val, wsi_val=wsi_val, aod_val=aod_val
                        )
                        output_preds[i, h] = verified_val
                return output_preds

            verified_preds_val = apply_z3_guardrail(df_val, X_val, rf_preds_val)
            verified_preds_test = apply_z3_guardrail(df_test, X_test, rf_preds_test)
            
            # 6. Set up calibration and compute uncertainty metrics
            models = {
                'Random Forest': (rf_preds_val, rf_preds_test),
                'XGBoost': (xgb_preds_val, xgb_preds_test),
                'Deep GRU': (gru_preds_val, gru_preds_test),
                'Z3-Verified': (verified_preds_val, verified_preds_test)
            }
            
            model_rmse = {}
            for name, (val_preds, _) in models.items():
                model_rmse[name] = {}
                for h in range(HORIZON):
                    rmse = np.sqrt(mean_squared_error(y_val_raw[:, h], val_preds[:, h]))
                    model_rmse[name][h] = max(rmse, 1e-3)
                    
            # 7. Evaluate classifications at safety thresholds <800m and <500m
            logger.info("Computing F1-Scores, False Alarm Ratios, and Brier probabilistic metrics at critical safety thresholds...")
            eval_horizons = [0, 5] # t+1 and t+6
            thresholds = [800.0, 500.0]
            summary_records = []
            
            for thresh in thresholds:
                for h_idx in eval_horizons:
                    h_label = f"t+{h_idx+1}h"
                    for name, (_, test_preds) in models.items():
                        metrics = self.evaluate_classification(y_test_raw[:, h_idx], test_preds[:, h_idx], thresh)
                        
                        # Convert deterministic forecasts to probabilities via normal dressing
                        sigma = model_rmse[name][h_idx]
                        preds = test_preds[:, h_idx]
                        probs = norm.cdf(thresh, loc=preds, scale=sigma)
                        actual_bin = (y_test_raw[:, h_idx] < thresh).astype(int)
                        brier_score = np.mean((probs - actual_bin) ** 2)
                        
                        record = {
                            'Model': name,
                            'Threshold': thresh,
                            'Horizon': h_label,
                            'Precision': metrics['Precision'],
                            'Recall (POD)': metrics['Recall (POD)'],
                            'False Alarm Ratio (FAR)': metrics['False Alarm Ratio (FAR)'],
                            'False Alarm Rate (FPR)': metrics['False Alarm Rate (FPR)'],
                            'F1-Score': metrics['F1-Score'],
                            'Brier Score': float(brier_score),
                            'TP': metrics['TP'], 'FP': metrics['FP'], 'TN': metrics['TN'], 'FN': metrics['FN']
                        }
                        summary_records.append(record)
                        
            # Export metrics
            df_metrics = pd.DataFrame(summary_records)
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            df_metrics.to_csv(self.metrics_file, index=False)
            logger.info(f"[ModelEvaluation] Fused safety verification summary exported to: {self.metrics_file}")
            upload_to_s3(self.metrics_file, "data/safety_evaluation_metrics.csv")
            
            # 8. Generate and save plots to processed data directory AND Flask static/img folder
            logger.info("Generating and rendering Reliability Diagrams and ROC curves using Matplotlib...")
            static_img_dir = os.path.join(SRC_DIR, "app", "static", "img")
            os.makedirs(static_img_dir, exist_ok=True)
            
            colors = {
                'Random Forest': '#7f8c8d',
                'XGBoost': '#e67e22',
                'Deep GRU': '#9b59b6',
                'Z3-Verified': '#27ae60'
            }
            
            # Plot 1: Reliability Diagram
            h_idx = 5  # t+6h
            thresh = 800.0
            y_true_bin = (y_test_raw[:, h_idx] < thresh).astype(int)
            
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            for name, (_, test_preds) in models.items():
                sigma = model_rmse[name][h_idx]
                preds = test_preds[:, h_idx]
                probs = norm.cdf(thresh, loc=preds, scale=sigma)
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true_bin, probs, n_bins=10, strategy='uniform'
                )
                ax1.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f"{name}", color=colors[name], linewidth=2)
                
            ax1.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect Calibration')
            ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
            ax1.set_ylabel('Observed Relative Frequency', fontsize=12)
            ax1.set_title(f'Reliability Diagram: Fog Collapse Events (<800m) at t+6h Horizon', fontsize=14, fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.legend(fontsize=11, loc='upper left')
            plt.tight_layout()
            
            plot1_proc_path = os.path.join(os.path.dirname(self.metrics_file), "reliability_diagram.png")
            plot1_static_path = os.path.join(static_img_dir, "reliability_diagram.png")
            
            fig1.savefig(plot1_proc_path, dpi=300)
            fig1.savefig(plot1_static_path, dpi=300)
            plt.close(fig1)
            upload_to_s3(plot1_proc_path, "data/reliability_diagram.png")
            logger.info("[ModelEvaluation] Reliability diagram saved and uploaded successfully.")
            
            # Plot 2: ROC-AUC Curves
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            for name, (_, test_preds) in models.items():
                sigma = model_rmse[name][h_idx]
                preds = test_preds[:, h_idx]
                probs = norm.cdf(thresh, loc=preds, scale=sigma)
                fpr, tpr, _ = roc_curve(y_true_bin, probs)
                auc_val = roc_auc_score(y_true_bin, probs)
                ax2.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", color=colors[name], linewidth=2)
                
            ax2.plot([0, 1], [0, 1], linestyle='--', color='black')
            ax2.set_xlabel('False Positive Rate (FPR)', fontsize=12)
            ax2.set_ylabel('True Positive Rate (TPR)', fontsize=12)
            ax2.set_title(f'ROC-AUC Curves: Fog Collapse Events (<800m) at t+6h Horizon', fontsize=14, fontweight='bold')
            ax2.grid(True, linestyle=':', alpha=0.6)
            ax2.legend(fontsize=11, loc='lower right')
            plt.tight_layout()
            
            plot2_proc_path = os.path.join(os.path.dirname(self.metrics_file), "roc_auc_curve.png")
            plot2_static_path = os.path.join(static_img_dir, "roc_auc_curve.png")
            
            fig2.savefig(plot2_proc_path, dpi=300)
            fig2.savefig(plot2_static_path, dpi=300)
            plt.close(fig2)
            upload_to_s3(plot2_proc_path, "data/roc_auc_curve.png")
            logger.info("[ModelEvaluation] ROC-AUC curves successfully exported and archived.")
            
            # Return RMSE stats for use in runtime predictions
            rmse_record_path = os.path.join(self.models_dir, "model_rmse.json")
            with open(rmse_record_path, "w") as f:
                json.dump(model_rmse, f, indent=4)
            upload_to_s3(rmse_record_path, "models/model_rmse.json")
            
            logger.info("✅ [ModelEvaluation] Evaluation suite successfully generated all statistics!")
        except Exception as eval_ex:
            logger.error(f"❌ [ModelEvaluation] Evaluation stage encountered a fatal exception: {eval_ex}", exc_info=True)
            raise eval_ex

if __name__ == "__main__":
    me = ModelEvaluation()
    me.run_evaluation()
