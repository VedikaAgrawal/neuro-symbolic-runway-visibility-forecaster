import os
import json
import logging
from datetime import datetime, timezone

# Setup robust centralized logging to console and a persistent file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_log_dir = os.path.join(project_root, "logs")
os.makedirs(local_log_dir, exist_ok=True)
log_file_path = os.path.join(local_log_dir, "pipeline_execution.log")

# Create logger
pipeline_logger = logging.getLogger("AeroVerifyPipeline")
pipeline_logger.setLevel(logging.INFO)

# Avoid adding duplicate handlers if logger is imported multiple times
if not pipeline_logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    pipeline_logger.addHandler(console_handler)
    
    # File Handler
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    pipeline_logger.addHandler(file_handler)

# Keep the module's alias for backwards compatibility
logger = pipeline_logger

# Manual dotenv loader to keep poetry environment light and zero-dependency
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        logger.info(f"🔑 Loading environment variables from {env_path}")
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    # Strip quotes if present
                    val = val.strip().strip("'").strip('"')
                    os.environ[key.strip()] = val
    else:
        logger.warning("⚠️ No .env file detected in project root. Running in local fallback mode.")

load_env()

# --- MongoDB Connector & Logging ---
def get_mongo_client():
    mongo_uri = os.environ.get("MONGO_DB_URI")
    if not mongo_uri:
        logger.info("ℹ️ MONGO_DB_URI not set. Prediction telemetry logging will fall back to local JSON file.")
        return None
        
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        # Test connection
        client.admin.command('ping')
        logger.info("✅ Successfully connected to MongoDB!")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}. Telemetry will use local fallback logs.")
        return None

def log_prediction_to_mongo(input_data, raw_forecasts, verified_forecasts, status, violations):
    """
    Logs prediction logs, user inputs, and Z3 symbolic verification audit logs.
    Saves to MongoDB if online, otherwise appends to a local JSON lines file.
    """
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_features": input_data,
        "raw_forecasts_meters": [float(v) for v in raw_forecasts],
        "verified_forecasts_meters": [float(v) for v in verified_forecasts],
        "z3_status": status,  # SAT or UNSAT
        "z3_violations": violations
    }
    
    # Try logging to MongoDB
    client = get_mongo_client()
    if client is not None:
        try:
            db = client["climate_visibility"]
            collection = db["predictions_audit"]
            collection.insert_one(log_record)
            logger.info("📡 Telemetry logged to MongoDB successfully.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to log to MongoDB: {e}. Falling back to local logging.")
            
    # Local fallback file logging
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_log_dir = os.path.join(project_root, "logs")
    os.makedirs(local_log_dir, exist_ok=True)
    local_log_file = os.path.join(local_log_dir, "predictions_telemetry.jsonl")
    
    try:
        # Convert ObjectId or complex types to serialize safely
        with open(local_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record) + "\n")
        logger.info(f"📂 Telemetry logged locally to {local_log_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write local fallback telemetry: {e}")
        return False


# --- AWS S3 Connectors ---
def get_s3_client():
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("AWS_S3_BUCKET")
    
    if not (aws_key and aws_secret and bucket_name):
        logger.info("ℹ️ AWS S3 credentials not fully set. Model artifacts will be managed locally.")
        return None, None
        
    try:
        import boto3
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret
        )
        return s3, bucket_name
    except Exception as e:
        logger.error(f"❌ Failed to initialize AWS S3 Client: {e}.")
        return None, None

def upload_to_s3(local_path, s3_key):
    """
    Uploads a local file to S3 if online, else log warning.
    """
    s3, bucket = get_s3_client()
    if s3 is None:
        logger.info(f"Local storage mode: artifact saved at {local_path} (not uploaded to S3).")
        return False
        
    try:
        logger.info(f"📤 Uploading {local_path} to S3 bucket '{bucket}' as '{s3_key}'...")
        s3.upload_file(local_path, bucket, s3_key)
        logger.info("✅ Upload complete!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to upload artifact to S3: {e}")
        return False

def download_from_s3(s3_key, local_path):
    """
    Downloads a file from S3 to local_path. Returns True if successful.
    """
    s3, bucket = get_s3_client()
    if s3 is None:
        logger.info("Local storage mode: skipping S3 download, reading local files.")
        return False
        
    try:
        logger.info(f"📥 Downloading S3 key '{s3_key}' from bucket '{bucket}' to '{local_path}'...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, s3_key, local_path)
        logger.info("✅ Download complete!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download artifact from S3: {e}. Falling back to local files.")
        return False
