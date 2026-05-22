import os
import numpy as np
import pandas as pd
from src.config import MASTER_FUSED_FILE, ENGINEERED_DATA_FILE, FEATURE_COLS
from src.utils import upload_to_s3

class FeatureEngineering:
    def __init__(self):
        self.input_file = MASTER_FUSED_FILE
        self.output_file = ENGINEERED_DATA_FILE
        
    def run_feature_engineering(self):
        """
        Executes the feature engineering pipeline from the master fused dataset.
        """
        print(f"[FeatureEngineering] Ingesting master fused data from: {self.input_file}")
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"🚨 Fused dataset not found at: {self.input_file}")
            
        df = pd.read_csv(self.input_file)
        
        # 1. Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 2. Drop rows with missing targets (airport_visibility)
        initial_len = len(df)
        df = df.dropna(subset=['airport_visibility']).reset_index(drop=True)
        print(f"[FeatureEngineering] Dropped {initial_len - len(df)} rows with missing target. Count is now: {len(df)}")
        
        # 3. Spatial Interpolation of 3-hourly/6-hourly weather variables
        interpolate_cols = [
            'urban_temp', 'urban_dew', 'urban_wind_speed', 'urban_wind_dir', 'urban_slp', 'urban_visibility',
            'rural_temp', 'rural_dew', 'rural_wind_speed', 'rural_wind_dir', 'rural_slp', 'rural_visibility'
        ]
        print("[FeatureEngineering] Interpolating urban and rural observation gaps...")
        df[interpolate_cols] = df[interpolate_cols].interpolate(method='linear', limit_direction='both')
        
        # 4. Domain Feature Calculations
        print("[FeatureEngineering] Extrapolating meteorological domain features...")
        stations = ['airport', 'urban', 'rural']
        for st in stations:
            # Dew Point Depression (DPD)
            df[f'{st}_dpd'] = df[f'{st}_temp'] - df[f'{st}_dew']
            df[f'{st}_dpd'] = df[f'{st}_dpd'].clip(lower=0.0)
            
            # Relative Humidity (RH) - Magnus-Tetens
            t = df[f'{st}_temp']
            td = df[f'{st}_dew']
            df[f'{st}_rh'] = 100.0 * np.exp((17.625 * td) / (243.04 + td) - (17.625 * t) / (243.04 + t))
            df[f'{st}_rh'] = df[f'{st}_rh'].clip(0.0, 100.0)
            
            # Wind Stagnation Index (WSI)
            df[f'{st}_wsi'] = (df[f'{st}_wind_speed'] < 1.5).astype(int)
            
            # Wind Vectors (U/V)
            rad = df[f'{st}_wind_dir'] * np.pi / 180.0
            df[f'{st}_wind_u'] = df[f'{st}_wind_speed'] * np.sin(rad)
            df[f'{st}_wind_v'] = df[f'{st}_wind_speed'] * np.cos(rad)
            
        # Aerosol Scattering Extinction Proxy (ASEP)
        df['asep'] = df['AOD_500nm'] / df['AOD_440nm']
        
        # Spatial Gradients
        df['spatial_grad_urban_airport'] = df['urban_temp'] - df['airport_temp']
        df['spatial_grad_rural_airport'] = df['rural_temp'] - df['airport_temp']
        
        # Cyclic Temporal Attributes
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        df['hour_sin'] = np.sin(2.0 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2.0 * np.pi * df['hour'] / 24.0)
        
        df['month_sin'] = np.sin(2.0 * np.pi * (df['month'] - 1) / 12.0)
        df['month_cos'] = np.cos(2.0 * np.pi * (df['month'] - 1) / 12.0)
        
        df['day_sin'] = np.sin(2.0 * np.pi * df['day_of_year'] / 366.0)
        df['day_cos'] = np.cos(2.0 * np.pi * df['day_of_year'] / 366.0)
        
        # 5. Clean final missing values for robustness (e.g. AOD during cloudy weeks) using forward-backward persistence fill
        df = df.ffill().bfill()
        
        # Check that all features in FEATURE_COLS are present and have 0 NaNs
        missing_features = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_features:
            raise ValueError(f"🚨 Features missing after engineering stage: {missing_features}")
            
        nan_counts = df[FEATURE_COLS].isna().sum().sum()
        if nan_counts > 0:
            print(f"[FeatureEngineering] ⚠️ Warning: {nan_counts} NaN values remain in feature columns. Forcing fill...")
            df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
            
        # Export processed file
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        df.to_csv(self.output_file, index=False)
        print(f"[FeatureEngineering] ✅ Completed! Exported file to: {self.output_file}")
        
        # 6. Upload to S3 if online
        upload_to_s3(self.output_file, "data/delhi_2024_engineered.csv")
        
        return self.output_file

if __name__ == "__main__":
    fe = FeatureEngineering()
    fe.run_feature_engineering()
