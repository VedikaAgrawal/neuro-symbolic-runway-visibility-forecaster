import os
import sys
import time
import traceback

# Ensure project root is in the path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Enforce absolute single-threaded execution to prevent macOS sandboxed parallel execution warnings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation
from src.utils import pipeline_logger

logger = pipeline_logger

class TrainingPipeline:
    def __init__(self):
        pass
        
    def run_pipeline(self):
        start_time = time.time()
        logger.info("================================================================================")
        logger.info("🎬 STARTING END-TO-END CLIMATE VISIBILITY FORECASTING TRAINING PIPELINE")
        logger.info("================================================================================")
        
        # Step 1: Data Ingestion
        try:
            logger.info("--- STAGE 1: DATA INGESTION ---")
            ingestion = DataIngestion()
            fused_file = ingestion.initiate_data_ingestion()
            logger.info(f"✅ Data Ingestion complete! Master fused file at: {fused_file}")
        except Exception as e:
            logger.error("🚨 PIPELINE CRASHED AT STAGE 1: DATA INGESTION!")
            logger.error(f"Error Details: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
        # Step 2: Data Validation
        try:
            logger.info("--- STAGE 2: DATA VALIDATION ---")
            validation = DataValidation()
            is_valid, report = validation.validate_dataset(fused_file)
            if not is_valid:
                logger.error("🚨 Ingested data schema validation failed! Schema structure does not conform to specifications.")
                sys.exit(1)
            logger.info("✅ Data Validation complete!")
        except Exception as e:
            logger.error("🚨 PIPELINE CRASHED AT STAGE 2: DATA VALIDATION!")
            logger.error(f"Error Details: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
        # Step 3: Feature Engineering
        try:
            logger.info("--- STAGE 3: FEATURE ENGINEERING ---")
            engineering = FeatureEngineering()
            engineered_file = engineering.run_feature_engineering()
            logger.info(f"✅ Feature Engineering complete! Engineered dataset at: {engineered_file}")
        except Exception as e:
            logger.error("🚨 PIPELINE CRASHED AT STAGE 3: FEATURE ENGINEERING!")
            logger.error(f"Error Details: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
        # Step 4: Model Training
        try:
            logger.info("--- STAGE 4: MODEL TRAINING ---")
            training = ModelTraining()
            training.train_models()
            logger.info("✅ Model Training complete!")
        except Exception as e:
            logger.error("🚨 PIPELINE CRASHED AT STAGE 4: MODEL TRAINING!")
            logger.error(f"Error Details: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
        # Step 5: Model Evaluation
        try:
            logger.info("--- STAGE 5: SAFETY EVALUATION ---")
            evaluation = ModelEvaluation()
            evaluation.run_evaluation()
            logger.info("✅ Model Evaluation complete!")
        except Exception as e:
            logger.error("🚨 PIPELINE CRASHED AT STAGE 5: SAFETY EVALUATION!")
            logger.error(f"Error Details: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
        elapsed = time.time() - start_time
        logger.info("================================================================================")
        logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
        logger.info("================================================================================")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
