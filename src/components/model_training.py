import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from src.config import (
    ENGINEERED_DATA_FILE, FEATURE_COLS, TARGET_COL,
    INPUT_SCALER_PATH, TARGET_SCALER_PATH, BEST_GRU_MODEL_PATH, FEATURE_NAMES_PATH,
    WINDOW_SIZE, HORIZON, RF_PARAMS, XGB_PARAMS, GRU_PARAMS, MODELS_DIR
)
from src.utils import upload_to_s3, pipeline_logger

logger = pipeline_logger

# PyTorch Dataset
class VisibilitySequenceDataset(Dataset):
    def __init__(self, X, y_scaled):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_scaled, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]

# PyTorch GRU Forecaster Model
class VisibilityGRUForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=6):
        super(VisibilityGRUForecaster, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sliding_windows(df_scaled, feature_cols, target_col, window_size=24, horizon=6):
    X_list, y_list = [], []
    max_idx = len(df_scaled) - horizon
    for i in range(window_size, max_idx):
        t_last_input = df_scaled.iloc[i-1]['timestamp']
        t_first_target = df_scaled.iloc[i]['timestamp']
        time_delta = (t_first_target - t_last_input).total_seconds() / 3600.0
        
        if abs(time_delta - 1.0) < 0.01:
            X_slice = df_scaled.iloc[i-window_size:i][feature_cols].values
            y_slice = df_scaled.iloc[i:i+horizon][target_col].values
            X_list.append(X_slice)
            y_list.append(y_slice)
            
    return np.array(X_list), np.array(y_list)

class ModelTraining:
    def __init__(self):
        self.data_file = ENGINEERED_DATA_FILE
        self.models_dir = MODELS_DIR
        
    def train_models(self):
        """
        Executes chronological splits, target scaling, sequence window building,
        and trains Random Forest, XGBoost and PyTorch GRU models.
        """
        logger.info(f"🏋️ [ModelTraining] Loading engineered dataset for training: {self.data_file}")
        if not os.path.exists(self.data_file):
            logger.error(f"[ModelTraining] Engineered data file not found: {self.data_file}")
            raise FileNotFoundError(f"🚨 Engineered data not found: {self.data_file}")
            
        try:
            df = pd.read_csv(self.data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 1. Chronological Splits
            logger.info("Splitting dataset chronologically into Train, Validation, and Test sets...")
            train_mask = df['timestamp'] < '2024-09-01 00:00:00'
            val_mask = (df['timestamp'] >= '2024-09-01 00:00:00') & (df['timestamp'] < '2024-11-01 00:00:00')
            test_mask = df['timestamp'] >= '2024-11-01 00:00:00'
            
            df_train = df[train_mask].reset_index(drop=True)
            df_val = df[val_mask].reset_index(drop=True)
            df_test = df[test_mask].reset_index(drop=True)
            
            logger.info(f"[ModelTraining] Split results: Train={len(df_train)} rows, Val={len(df_val)} rows, Test={len(df_test)} rows")
            if len(df_train) == 0 or len(df_val) == 0:
                raise ValueError("🚨 Chronological splits produced empty subsets. Validate dates in dataset.")
            
            # 2. Input and Target Scaling
            logger.info("Initializing and fitting StandardScalers on inputs and targets...")
            input_scaler = StandardScaler()
            target_scaler = StandardScaler()
            
            df_train_scaled_features = pd.DataFrame(
                input_scaler.fit_transform(df_train[FEATURE_COLS]), columns=FEATURE_COLS
            )
            df_val_scaled_features = pd.DataFrame(
                input_scaler.transform(df_val[FEATURE_COLS]), columns=FEATURE_COLS
            )
            df_test_scaled_features = pd.DataFrame(
                input_scaler.transform(df_test[FEATURE_COLS]), columns=FEATURE_COLS
            )
            
            y_train_scaled_col = target_scaler.fit_transform(df_train[[TARGET_COL]].values).flatten()
            y_val_scaled_col = target_scaler.transform(df_val[[TARGET_COL]].values).flatten()
            y_test_scaled_col = target_scaler.transform(df_test[[TARGET_COL]].values).flatten()
            
            def reconstruct_df(df_orig, df_scaled_feat, y_scaled):
                df_new = df_scaled_feat.copy()
                df_new['timestamp'] = df_orig['timestamp'].values
                df_new['airport_visibility'] = y_scaled
                return df_new
                
            df_train_scaled = reconstruct_df(df_train, df_train_scaled_features, y_train_scaled_col)
            df_val_scaled = reconstruct_df(df_val, df_val_scaled_features, y_val_scaled_col)
            df_test_scaled = reconstruct_df(df_test, df_test_scaled_features, y_test_scaled_col)
            
            # 3. Create sliding windows
            logger.info(f"Extracting temporal sequence windows (Window Size={WINDOW_SIZE}, Horizon={HORIZON})...")
            X_train, y_train_scaled = create_sliding_windows(df_train_scaled, FEATURE_COLS, TARGET_COL, WINDOW_SIZE, HORIZON)
            X_val, y_val_scaled = create_sliding_windows(df_val_scaled, FEATURE_COLS, TARGET_COL, WINDOW_SIZE, HORIZON)
            X_test, y_test_scaled = create_sliding_windows(df_test_scaled, FEATURE_COLS, TARGET_COL, WINDOW_SIZE, HORIZON)
            
            # Raw targets for Tree Regressors and metrics evaluations
            _, y_train_raw = create_sliding_windows(df_train, FEATURE_COLS, TARGET_COL, WINDOW_SIZE, HORIZON)
            _, y_val_raw = create_sliding_windows(df_val, FEATURE_COLS, TARGET_COL, WINDOW_SIZE, HORIZON)
            _, y_test_raw = create_sliding_windows(df_test, FEATURE_COLS, TARGET_COL, WINDOW_SIZE, HORIZON)
            
            logger.info(f"[ModelTraining] Generated Sliding Windows: Train={len(X_train)} samples, Val={len(X_val)} samples, Test={len(X_test)} samples")
            
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            # 4. Save Scalers and Feature Metadata
            os.makedirs(self.models_dir, exist_ok=True)
            joblib.dump(input_scaler, INPUT_SCALER_PATH)
            joblib.dump(target_scaler, TARGET_SCALER_PATH)
            with open(FEATURE_NAMES_PATH, "w") as f:
                json.dump(FEATURE_COLS, f)
                
            # Upload Scalers to S3
            logger.info("Uploading scalers and feature metadata to AWS S3 storage...")
            upload_to_s3(INPUT_SCALER_PATH, "models/input_scaler.joblib")
            upload_to_s3(TARGET_SCALER_PATH, "models/target_scaler.joblib")
            upload_to_s3(FEATURE_NAMES_PATH, "models/feature_names.json")
            
            # 5. Train Random Forest models (one per horizon)
            logger.info("Training Multi-Horizon Random Forest Regressors...")
            for h in range(HORIZON):
                try:
                    logger.info(f"  Training RF Regressor for Horizon t+{h+1}h...")
                    rf = RandomForestRegressor(**RF_PARAMS)
                    rf.fit(X_train_flat, y_train_raw[:, h])
                    rf_path = os.path.join(self.models_dir, f"rf_model_h{h}.joblib")
                    joblib.dump(rf, rf_path)
                    upload_to_s3(rf_path, f"models/rf_model_h{h}.joblib")
                except Exception as rf_err:
                    logger.error(f"Random Forest training failed at horizon index {h}: {rf_err}", exc_info=True)
                    raise rf_err
                
            # 6. Train XGBoost models (one per horizon)
            logger.info("Training Multi-Horizon XGBoost Regressors...")
            for h in range(HORIZON):
                try:
                    logger.info(f"  Training XGBoost Regressor for Horizon t+{h+1}h...")
                    xgb_reg = xgb.XGBRegressor(**XGB_PARAMS)
                    xgb_reg.fit(X_train_flat, y_train_raw[:, h])
                    xgb_path = os.path.join(self.models_dir, f"xgb_model_h{h}.joblib")
                    joblib.dump(xgb_reg, xgb_path)
                    upload_to_s3(xgb_path, f"models/xgb_model_h{h}.joblib")
                except Exception as xgb_err:
                    logger.error(f"XGBoost training failed at horizon index {h}: {xgb_err}", exc_info=True)
                    raise xgb_err
                
            # 7. Train PyTorch GRU Model
            logger.info("Training Deep PyTorch GRU Sequence Model on CPU...")
            try:
                train_dataset = VisibilitySequenceDataset(X_train, y_train_scaled)
                val_dataset = VisibilitySequenceDataset(X_val, y_val_scaled)
                
                train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
                
                input_dim = X_train.shape[2]
                gru_model = VisibilityGRUForecaster(
                    input_dim=input_dim,
                    hidden_dim=GRU_PARAMS["hidden_dim"],
                    num_layers=GRU_PARAMS["num_layers"],
                    output_dim=HORIZON
                )
                
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(gru_model.parameters(), lr=GRU_PARAMS["lr"])
                
                best_val_loss = float('inf')
                best_weights = gru_model.state_dict().copy()
                
                epochs = GRU_PARAMS["epochs"]
                for epoch in range(epochs):
                    gru_model.train()
                    train_loss = 0.0
                    for X_batch, y_batch in train_loader:
                        optimizer.zero_grad()
                        outputs = gru_model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * X_batch.size(0)
                    train_loss /= len(train_dataset)
                    
                    # Validation
                    gru_model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            outputs = gru_model(X_batch)
                            loss = criterion(outputs, y_batch)
                            val_loss += loss.item() * X_batch.size(0)
                    val_loss /= len(val_dataset)
                    
                    logger.info(f"  Epoch {epoch+1:02d}/{epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = gru_model.state_dict().copy()
                        
                # Load best weights and save PyTorch state dictionary
                gru_model.load_state_dict(best_weights)
                torch.save(gru_model.state_dict(), BEST_GRU_MODEL_PATH)
                upload_to_s3(BEST_GRU_MODEL_PATH, "models/best_gru_model.pt")
                logger.info(f"PyTorch GRU trained successfully! Best validation loss (MSE): {best_val_loss:.4f}")
            except Exception as gru_err:
                logger.error(f"PyTorch GRU network optimization/compilation loop failed: {gru_err}", exc_info=True)
                raise gru_err
                
            logger.info(f"✅ [ModelTraining] All regression and sequence architectures successfully trained and serialized to: {self.models_dir}")
        except Exception as mt_err:
            logger.error(f"❌ [ModelTraining] Chronological splits or training loops encountered a fatal exception: {mt_err}", exc_info=True)
            raise mt_err

if __name__ == "__main__":
    mt = ModelTraining()
    mt.train_models()
