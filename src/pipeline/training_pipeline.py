import os
import sys
import time

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

class TrainingPipeline:
    def __init__(self):
        pass
        
    def run_pipeline(self):
        start_time = time.time()
        print("================================================================================")
        print("🎬 STARTING END-TO-END CLIMATE VISIBILITY FORECASTING TRAINING PIPELINE")
        print("================================================================================")
        
        # Step 1: Data Ingestion
        print("\n--- STAGE 1: DATA INGESTION ---")
        ingestion = DataIngestion()
        fused_file = ingestion.initiate_data_ingestion()
        print(f"✅ Data Ingestion complete! Master fused file at: {fused_file}")
        
        # Step 2: Data Validation
        print("\n--- STAGE 2: DATA VALIDATION ---")
        validation = DataValidation()
        is_valid, report = validation.validate_dataset(fused_file)
        if not is_valid:
            print("🚨 Ingested data schema validation failed! Terminating training pipeline.")
            sys.exit(1)
        print("✅ Data Validation complete!")
        
        # Step 3: Feature Engineering
        print("\n--- STAGE 3: FEATURE ENGINEERING ---")
        engineering = FeatureEngineering()
        engineered_file = engineering.run_feature_engineering()
        print(f"✅ Feature Engineering complete! Engineered dataset at: {engineered_file}")
        
        # Step 4: Model Training
        print("\n--- STAGE 4: MODEL TRAINING ---")
        training = ModelTraining()
        training.train_models()
        print("✅ Model Training complete!")
        
        # Step 5: Model Evaluation
        print("\n--- STAGE 5: SAFETY EVALUATION ---")
        evaluation = ModelEvaluation()
        evaluation.run_evaluation()
        print("✅ Model Evaluation complete!")
        
        elapsed = time.time() - start_time
        print("\n================================================================================")
        print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
        print("================================================================================")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
