import os
import numpy as np
import pandas as pd
from src.config import DATA_RAW_DIR, MASTER_FUSED_FILE
from src.utils import upload_to_s3, pipeline_logger

logger = pipeline_logger

def parse_noaa_isd_fixed_width(file_path, station_name):
    """
    Parses raw NOAA ISD fixed-width ASCII weather observation records using stable physical byte offsets.
    This reads from the mandatory control and meteorological section (always 105 characters).
    """
    logger.info(f"Ingesting {station_name} dataset from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Raw weather file not found at: {file_path}")
        raise FileNotFoundError(f"🚨 Raw weather file not found at: {file_path}")
        
    parsed_rows = []
    skipped_rows_count = 0
    total_rows_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                total_rows_count += 1
                if len(line) < 105:
                    skipped_rows_count += 1
                    continue
                    
                try:
                    # The 4-digit year '2024' always starts exactly at index 15 in NOAA ISD
                    year = line[15:19]
                    if year != "2024":
                        continue
                        
                    month = line[19:21]
                    day = line[21:23]
                    hour = line[23:25]
                    minute = line[25:27]
                    
                    timestamp = pd.to_datetime(f"{year}-{month}-{day} {hour}:{minute}:00")
                    
                    # Wind Features: Direction (60:63), Speed (65:69)
                    wind_dir_raw = line[60:63]
                    wind_dir = float(wind_dir_raw) if wind_dir_raw != "999" else np.nan
                    
                    wind_speed_raw = line[65:69]
                    wind_speed = float(wind_speed_raw) / 10.0 if wind_speed_raw != "9999" else np.nan
                    
                    # Visibility Range (78:84)
                    vis_raw = line[78:84]
                    visibility = float(vis_raw) if vis_raw != "999999" else np.nan
                    
                    # Air Temperature (87:92)
                    temp_raw = line[87:92]
                    if temp_raw in ("+9999", "99999"):
                        temp = np.nan
                    else:
                        temp = float(temp_raw[1:5]) / 10.0
                        if temp_raw[0] == '-':
                            temp = -temp
                            
                    # Dew Point Temperature (93:98)
                    dew_raw = line[93:98]
                    if dew_raw in ("+9999", "99999"):
                        dew = np.nan
                    else:
                        dew = float(dew_raw[1:5]) / 10.0
                        if dew_raw[0] == '-':
                            dew = -dew
                            
                    # Sea Level Pressure (99:104)
                    slp_raw = line[99:104]
                    slp = float(slp_raw) / 10.0 if slp_raw != "99999" else np.nan
                    
                    parsed_rows.append({
                        'timestamp': timestamp,
                        f'{station_name}_temp': temp,
                        f'{station_name}_dew': dew,
                        f'{station_name}_wind_speed': wind_speed,
                        f'{station_name}_wind_dir': wind_dir,
                        f'{station_name}_slp': slp,
                        f'{station_name}_visibility': visibility
                    })
                except Exception as row_err:
                    skipped_rows_count += 1
                    # Avoid spamming log file, write as debug/low-priority or keep simple count
                    continue
    except Exception as io_err:
        logger.error(f"Critical I/O error occurred while reading weather file {file_path}: {io_err}", exc_info=True)
        raise io_err
        
    logger.info(f"Finished weather file scan. Total rows evaluated: {total_rows_count}, Invalid/Skipped rows: {skipped_rows_count}")
    
    try:
        df = pd.DataFrame(parsed_rows)
        logger.info(f"Parsed {len(df)} lines successfully into DataFrame for station: {station_name}")
        
        # Aggregate into strict hourly windows and reindex onto the complete 2024 timeline
        df = df.set_index('timestamp').resample('1h').mean()
        
        # Complete 2024 hourly calendar (8784 uniform slots)
        time_index = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:00:00', freq='h')
        df = df.reindex(time_index)
        df.index.name = 'timestamp'
        df = df.reset_index()
        
        return df
    except Exception as df_err:
        logger.error(f"DataFrame aggregation/reindexing failed for station {station_name}: {df_err}", exc_info=True)
        raise df_err

def fuse_aerosols_and_export(df_air, df_urb, df_rur, aeronet_file, output_file):
    """
    Standardizes daily NASA AERONET AOD data and left-joins it onto the uniform hourly meteorological matrix.
    """
    aeronet_path = os.path.join(DATA_RAW_DIR, aeronet_file)
    output_path = output_file
    
    logger.info(f"Loading NASA Aerosol AOD data from: {aeronet_path}")
    if not os.path.exists(aeronet_path):
        logger.error(f"NASA AERONET CSV not found at: {aeronet_path}")
        raise FileNotFoundError(f"🚨 NASA AERONET CSV not found at: {aeronet_path}")
        
    try:
        df_aero = pd.read_csv(aeronet_path)
        df_aero.columns = df_aero.columns.str.strip()
        
        # Date parsing
        df_aero['date_parsed'] = pd.to_datetime(df_aero['Date(dd:mm:yyyy)'], format='%d:%m:%Y')
        
        # Target aerosol properties to track
        target_cols = ['AOD_500nm', 'AOD_440nm', 'AOD_675nm', '440-870_Angstrom_Exponent']
        for col in target_cols:
            if col in df_aero.columns:
                df_aero[col] = df_aero[col].replace(-999.0, np.nan)
                
        # Calculate daily average AOD values
        df_daily_aero = df_aero.groupby('date_parsed')[target_cols].mean().reset_index()
        df_daily_aero = df_daily_aero.rename(columns={'date_parsed': 'date_key'})
        
        logger.info("Fusing weather station datasets...")
        master_df = pd.merge(df_air, df_urb, on='timestamp', how='outer')
        master_df = pd.merge(master_df, df_rur, on='timestamp', how='outer')
        
        # Normalize hourly timestamps to daily date keys for AOD merging
        master_df['date_key'] = pd.DatetimeIndex(master_df['timestamp']).normalize()
        
        logger.info("Merging spatial observations with daily atmospheric aerosol indexes...")
        final_fused_df = pd.merge(master_df, df_daily_aero, on='date_key', how='left')
        final_fused_df = final_fused_df.drop(columns=['date_key'])
        final_fused_df = final_fused_df.sort_values('timestamp').reset_index(drop=True)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_fused_df.to_csv(output_path, index=False)
        
        logger.info(f"SUCCESS: Unified Delhi-NCR 2024 meteorological spatial matrix created at: {output_path}")
        return final_fused_df
    except Exception as fuse_err:
        logger.error(f"Dataset fusion/export step failed: {fuse_err}", exc_info=True)
        raise fuse_err

class DataIngestion:
    def __init__(self):
        self.raw_dir = DATA_RAW_DIR
        self.output_file = MASTER_FUSED_FILE
        
    def initiate_data_ingestion(self):
        """
        Runs the full ingestion pipeline.
        """
        logger.info("🎬 [DataIngestion] Initiating weather observation ingestion stages...")
        
        airport_file = os.path.join(self.raw_dir, "421810-99999-2024.txt")
        urban_file = os.path.join(self.raw_dir, "421820-99999-2024.txt")
        rural_file = os.path.join(self.raw_dir, "421390-99999-2024.txt")
        aeronet_file = "aeronet_aerosols_raw.csv"
        
        try:
            # 1. Parse weather stations robustly from fixed-width ASCII formats
            logger.info("Parsing airport observed records...")
            df_airport = parse_noaa_isd_fixed_width(airport_file, "airport")
            
            logger.info("Parsing Safdarjung urban observed records...")
            df_safdarjung = parse_noaa_isd_fixed_width(urban_file, "urban")
            
            logger.info("Parsing Rohtak rural observed records...")
            df_rohtak = parse_noaa_isd_fixed_width(rural_file, "rural")
            
            # 2. Fuse with aerosols and export master matrix
            logger.info("Fusing meteorological stations with NASA AERONET AOD CSV...")
            df_fused = fuse_aerosols_and_export(
                df_airport, df_safdarjung, df_rohtak,
                aeronet_file, self.output_file
            )
            
            # 3. Upload to S3 if online
            logger.info("Attempting model artifacts backup to AWS S3 storage...")
            upload_to_s3(self.output_file, "data/delhi_2024_master_fused.csv")
            
            logger.info("✅ [DataIngestion] Stage completed successfully!")
            return self.output_file
        except Exception as ingest_ex:
            logger.error(f"❌ [DataIngestion] Pipeline ingestion failed: {ingest_ex}", exc_info=True)
            raise ingest_ex

if __name__ == "__main__":
    ingest = DataIngestion()
    ingest.initiate_data_ingestion()
