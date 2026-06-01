import os
import sys
import pandas as pd
from flask import Flask, jsonify, render_template, request

# Ensure project root is in the path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.config import SAFETY_METRICS_FILE

app = Flask(__name__)

# Initialize Prediction Pipeline once at startup
try:
    pipeline = PredictionPipeline()
except Exception as e:
    print(f"⚠️ Warning: Could not initialize prediction pipeline at startup: {e}")
    pipeline = None

@app.route("/")
def index():
    """Serves the premium dashboard landing page."""
    return render_template("index.html")

@app.route("/api/timestamps", methods=["GET"])
def get_timestamps():
    """Returns a list of valid historical winter timestamps for testing."""
    if not pipeline:
        return jsonify({"error": "Prediction pipeline not initialized."}), 500
    try:
        limit = request.args.get("limit", default=100, type=int)
        timestamps = pipeline.get_test_timestamps(limit=limit)
        return jsonify({"timestamps": timestamps})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/sequence", methods=["GET"])
def get_sequence():
    """Loads the 24-hour meteorological sequence ending at the selected timestamp."""
    if not pipeline:
        return jsonify({"error": "Prediction pipeline not initialized."}), 500
    timestamp = request.args.get("timestamp")
    if not timestamp:
        return jsonify({"error": "Missing required query parameter: timestamp"}), 400
        
    try:
        seq_df = pipeline.get_sequence_by_timestamp(timestamp)
        # Extract the last row as the current hour's default values
        last_row = seq_df.iloc[-1].to_dict()
        
        # Convert timestamp to string
        last_row["timestamp"] = last_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        
        # Select key METAR/AOD values that the user can override
        form_defaults = {
            "timestamp": last_row["timestamp"],
            "airport_temp": float(last_row["airport_temp"]),
            "airport_dew": float(last_row["airport_dew"]),
            "airport_wind_speed": float(last_row["airport_wind_speed"]),
            "airport_wind_dir": float(last_row["airport_wind_dir"]),
            "urban_temp": float(last_row["urban_temp"]),
            "rural_temp": float(last_row["rural_temp"]),
            "AOD_500nm": float(last_row["AOD_500nm"]),
            "AOD_440nm": float(last_row["AOD_440nm"]),
            "airport_visibility": float(last_row["airport_visibility"])
        }
        
        return jsonify({
            "timestamp": timestamp,
            "defaults": form_defaults,
            "history_length": len(seq_df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    """Receives meteorological parameters, runs prediction models, and audits with Z3 solver."""
    global pipeline
    if not pipeline:
        # Retry initialization in case models were trained after startup
        try:
            pipeline = PredictionPipeline()
        except Exception as e:
            return jsonify({"error": f"Prediction pipeline not available: {e}"}), 500
            
    data = request.get_json() or {}
    timestamp = data.get("timestamp")
    if not timestamp:
        return jsonify({"error": "Missing required parameter: timestamp"}), 400
        
    overrides = data.get("overrides", None)
    
    try:
        inference_result = pipeline.run_inference(timestamp, overrides)
        return jsonify(inference_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Reads safety metrics from the processed evaluation file."""
    if not os.path.exists(SAFETY_METRICS_FILE):
        return jsonify({"error": "Metrics file not found. Run training/evaluation pipeline first."}), 404
        
    try:
        df = pd.read_csv(SAFETY_METRICS_FILE)
        metrics_list = df.to_dict(orient="records")
        return jsonify({"metrics": metrics_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render dynamically binds the port specified in the PORT environment variable
    port = int(os.environ.get("PORT", 5050))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
