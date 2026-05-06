import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

def load_model():
    client = MlflowClient()
    experiment_name = "e2e-mlops-project"
    
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found. Please train the model first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")

    latest_run_id = runs[0].info.run_id
    model_uri = f"runs:/{latest_run_id}/final_model"
    
    print(f"Loading latest model dynamically from run ID: {latest_run_id}")
    return mlflow.keras.load_model(model_uri)