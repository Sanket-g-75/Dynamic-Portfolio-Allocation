import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

import os

import glob

def load_model():
    # Detect if we are running inside Docker
    in_docker = os.path.exists("/app/mlruns")
    
    # Base directory to search for models
    base_dir = "/app/mlruns" if in_docker else "mlruns"
    
    if not os.path.exists(base_dir):
        raise ValueError(f"Directory {base_dir} does not exist. Please train the model first.")
        
    # Find all MLmodel files
    mlmodels = glob.glob(f"{base_dir}/**/MLmodel", recursive=True)
    if not mlmodels:
        raise ValueError(f"No MLmodel found in {base_dir}. Please train the model first.")
        
    # Get the latest one by modification time
    latest_mlmodel = max(mlmodels, key=os.path.getmtime)
    model_dir = os.path.dirname(latest_mlmodel)
    
    print(f"Loading latest model dynamically from: {model_dir}")
    return mlflow.keras.load_model(model_dir)