import os
import pandas as pd
import numpy as np

class DataValidation:
    def __init__(self):
        pass
        
    def validate_dataset(self, file_path):
        """
        Validates the schema, columns, and meteorological range boundaries of the fused dataset.
        Returns (is_valid: bool, validation_report: dict)
        """
        print(f"[DataValidation] Validating dataset from: {file_path}")
        if not os.path.exists(file_path):
            return False, {"error": f"Dataset file not found: {file_path}"}
            
        df = pd.read_csv(file_path)
        report = {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": list(df.columns),
            "range_violations": {}
        }
        
        # Check standard required raw column names
        required_cols = [
            "timestamp", 
            "airport_temp", "airport_dew", "airport_wind_speed", "airport_wind_dir", "airport_slp", "airport_visibility",
            "urban_temp", "urban_dew", "urban_wind_speed", "urban_wind_dir", "urban_slp", "urban_visibility",
            "rural_temp", "rural_dew", "rural_wind_speed", "rural_wind_dir", "rural_slp", "rural_visibility",
            "AOD_500nm", "AOD_440nm"
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            report["missing_columns"] = missing_cols
            print(f"[DataValidation] ❌ Validation failed due to missing columns: {missing_cols}")
            return False, report
            
        # Range validations
        violations = []
        
        # 1. Temperature limits (-10°C to 60°C)
        temp_cols = ["airport_temp", "urban_temp", "rural_temp"]
        for col in temp_cols:
            outliers = df[(df[col] < -10.0) | (df[col] > 60.0)][col]
            if len(outliers) > 0:
                violations.append(f"{col} has {len(outliers)} rows violating bounds [-10, 60]°C")
                
        # 2. Wind speed (0 to 100 m/s)
        wind_cols = ["airport_wind_speed", "urban_wind_speed", "rural_wind_speed"]
        for col in wind_cols:
            outliers = df[(df[col] < 0.0) | (df[col] > 100.0)][col]
            if len(outliers) > 0:
                violations.append(f"{col} has {len(outliers)} rows violating bounds [0, 100] m/s")
                
        # 3. Sea level pressure (800 to 1200 hPa)
        slp_cols = ["airport_slp", "urban_slp", "rural_slp"]
        for col in slp_cols:
            outliers = df[(df[col] < 800.0) | (df[col] > 1200.0)][col]
            if len(outliers) > 0:
                violations.append(f"{col} has {len(outliers)} rows violating bounds [800, 1200] hPa")
                
        # 4. Visibility (0 to 20000 m)
        vis_cols = ["airport_visibility", "urban_visibility", "rural_visibility"]
        for col in vis_cols:
            outliers = df[(df[col] < 0.0) | (df[col] > 20000.0)][col]
            if len(outliers) > 0:
                violations.append(f"{col} has {len(outliers)} rows violating bounds [0, 20000] meters")
                
        # 5. AOD limits (0 to 10)
        aod_cols = ["AOD_500nm", "AOD_440nm"]
        for col in aod_cols:
            outliers = df[(df[col] < 0.0) | (df[col] > 10.0)][col]
            if len(outliers) > 0:
                violations.append(f"{col} has {len(outliers)} rows violating bounds [0, 10] indices")
                
        if len(violations) > 0:
            report["range_violations"] = violations
            print(f"[DataValidation] ⚠️ Meteorological range alerts detected:\n" + "\n".join(violations))
            # Note: We return True here as these range alerts could represent extreme weather events (not necessarily pipeline failures)
            # but we log them. If columns are fine, it is structurally valid.
            
        print("[DataValidation] ✅ Dataset structure successfully validated!")
        return True, report

if __name__ == "__main__":
    from src.config import MASTER_FUSED_FILE
    val = DataValidation()
    val.validate_dataset(MASTER_FUSED_FILE)
