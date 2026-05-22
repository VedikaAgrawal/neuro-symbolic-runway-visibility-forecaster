1. Activating your Environment via Terminal
To execute scripts or run terminal-bound pipelines natively inside your locked dependency boundaries, use the poetry run wrapper:

Bash
poetry run python scripts/your_script.py
Alternatively, spawn a dedicated terminal sub-shell pinned directly inside your environment layer:

Bash
poetry shell
2. Launching notebooks directly inside VS Code
You do not need to bounce out to a web browser layer. Work directly inside your native VS Code environment:

Open the file explorer sidebar and click into notebooks/01_data_fusion_lake.ipynb.

Look at the upper right-hand header of the open workbook tab and click "Select Kernel".

Choose Python Environments... from the central dropdown.

Select the environment tagged explicitly with your path signature and virtual environment tag:
Python 3.12.x (~/Documents/climate-visibility-new/.venv/bin/python)

Run your notebook cells cleanly via Shift + Enter.


---

### 📊 Metric Reference: `docs/dataset_reference.md` (Save in `docs/` Folder)

```markdown
# Data Dictionary & Spatial Fusion Reference Manual
This document serves as an exhaustive structural ledger detailing the data schemas, raw fixed-width byte offsets, physical unit metrics, and spatial integration logic utilized to engineer the consolidated hourly meteorological/aerosol matrix for Delhi-NCR.

---

## 🌐 Spatial Configuration Map
The master execution matrix dynamically fuses data across three highly distinct geographic coordinates to account for spatial variations across urban microclimates, transport corridors, and regional baselines:

1. **`airport` (Delhi IGI Airport - Station ID: 421810)**: High-frequency terminal meteorological hub acting as our primary visibility target point. Highly prone to severe localized radiation fog layers.
2. **`urban` (Safdarjung Observatory - Station ID: 421820)**: Dense urban central core baseline capturing anthropogenically driven microclimate variations, heat island metrics, and high-concentration city center stagnation.
3. **`rural` (Rohtak Regional Baseline - Station ID: 421390)**: Upwind regional background anchor situated in a rural setting, critical for separating local pollution clusters from macro-scale transboundary agricultural biomass smoke plumes.
4. **`NASA AERONET Hub` (Kanpur/Regional Column Baseline)**: High-accuracy ground-based sun photometer column tracking real-time Aerosol Optical Depth (AOD) characteristics to provide true physical optical mass loads.

---

## 📑 Core Ingestion Blueprints (NOAA ISD Decoding)
Raw source files from NOAA consist of variable-length prefixes (`0100`, `0156`, `0218`) denoting individual row string lengths. To eliminate indexing failures, the data layer utilizes a **Dynamic Year Anchor Validation Engine** that locks onto the 4-digit chronological marker `2024` on each row line and applies the following calibrated offsets:

| Target Property | Relative Year Offset | Format Width | Raw Sample | Physical Unit Metric & Conversion Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Observation Timestamp** | `y_idx + 0` to `y_idx + 12` | 12 chars | `202401010000` | Converted to unified datetime index string: `YYYY-MM-DD HH:MM:SS` |
| **Wind Direction** | `y_idx + 45` to `y_idx + 48` | 3 chars | `270` | Wind bearing in degrees (0° to 360°). `999` indicates missing values. |
| **Wind Speed** | `y_idx + 50` to `y_idx + 54` | 4 chars | `0021` | Meters per second (m/s). **Scale factor: divide by 10.0** (`0021` → 2.1 m/s). |
| **Air Temperature** | `y_idx + 72` to `y_idx + 77` | 5 chars | `+0122` | Degrees Celsius (°C). Char 0 acts as sign tracker. **Scale factor: divide by 10.0** (`+0122` → 12.2°C). |
| **Dew Point Temp** | `y_idx + 78` to `y_idx + 83` | 5 chars | `+0090` | Degrees Celsius (°C). Critical moisture proxy. **Scale factor: divide by 10.0** (`+0090` → 9.0°C). |
| **Sea Level Pressure** | `y_idx + 84` to `y_idx + 89` | 5 chars | `10194` | Hectopascals (hPa) / Millibars. **Scale factor: divide by 10.0** (`10194` → 1019.4 hPa). |
| **Visibility Range** | `y_idx + 91` to `y_idx + 97` | 6 chars | `000500` | **Core Machine Learning Target Variable**. Stored in meters (m). `999999` represents missing records. |

---

## 📊 Processed Unified Feature Columns (`delhi_2024_master_fused.csv`)
The generated dataset contains **23 explicit engineering columns** mapped out chronologically into strict 1-hour uniform buckets ($n=8784$ records):

```text
 #   Column                       Type     Data Source         Description / Scaling Unit
---  ------                       ----     -----------         --------------------------
 0   timestamp                    datetime Time Axis           Unified temporal tracking key index [YYYY-MM-DD HH:MM:SS]
 1   airport_temp                 float    NOAA ISD 421810     Air Temperature at IGI Airport in degrees Celsius
 2   airport_dew                  float    NOAA ISD 421810     Dew Point Temperature at IGI Airport in degrees Celsius
 3   airport_wind_speed           float    NOAA ISD 421810     Wind Speed at IGI Airport in meters per second (m/s)
 4   airport_wind_dir             float    NOAA ISD 421810     Wind Bearing Direction at IGI Airport in degrees (0-360)
 5   airport_slp                  float    NOAA ISD 421810     Sea Level Atmospheric Pressure at IGI Airport in hPa
 6   airport_visibility           float    NOAA ISD 421810     TARGET VARIABLE: Visibility depth range in meters (m)
 7   urban_temp                   float    NOAA ISD 421820     Air Temperature at Safdarjung (Urban Core) in Celsius
 8   urban_dew                    float    NOAA ISD 421820     Dew Point Temperature at Safdarjung in Celsius
 9   urban_wind_speed             float    NOAA ISD 421820     Wind Speed at Safdarjung in meters per second (m/s)
 10  urban_wind_dir               float    NOAA ISD 421820     Wind Bearing Direction at Safdarjung in degrees (0-360)
 11  urban_slp                    float    NOAA ISD 421820     Sea Level Atmospheric Pressure at Safdarjung in hPa
 12  urban_visibility             float    NOAA ISD 421820     Visibility depth range at Safdarjung in meters (m)
 13  rural_temp                   float    NOAA ISD 421390     Air Temperature at Rohtak (Rural Background) in Celsius
 14  rural_dew                    float    NOAA ISD 421390     Dew Point Temperature at Rohtak in Celsius
 15  rural_wind_speed             float    NOAA ISD 421390     Wind Speed at Rohtak in meters per second (m/s)
 16  rural_wind_dir               float    NOAA ISD 421390     Wind Bearing Direction at Rohtak in degrees (0-360)
 17  rural_slp                    float    NOAA ISD 421390     Sea Level Atmospheric Pressure at Rohtak in hPa
 18  rural_visibility             float    NOAA ISD 421390     Visibility depth range at Rohtak in meters (m)
 19  AOD_500nm                    float    NASA AERONET        Aerosol Optical Depth at 500nm (Critical visibility attenuation channel)
 20  AOD_440nm                    float    NASA AERONET        Aerosol Optical Depth at 440nm channel
 21  AOD_675nm                    float    NASA AERONET        Aerosol Optical Depth at 675nm channel
 22  440-870_Angstrom_Exponent    float    NASA AERONET        Angstrom Exponent proxy representing predominant particle sizing distributions