import mlflow
import os

# Start an MLflow run
with mlflow.start_run():
    print("Running MLflow experiment...")

    # Log a parameter (e.g., a hyperparameter for a model)
    mlflow.log_param("alpha", 0.5)
    print("Logged parameter 'alpha': 0.5")

    # Log a metric (e.g., a model performance metric)
    mlflow.log_metric("rmse", 0.87)
    print("Logged metric 'rmse': 0.87")

    # Create a dummy file to log as an artifact
    artifact_path = "outputs"
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)

    with open(os.path.join(artifact_path, "model.txt"), "w") as f:
        f.write("This is a dummy model file.")

    # Log the artifact
    mlflow.log_artifacts(artifact_path)
    print(f"Logged artifact from: '{artifact_path}'")

    print("\nMLflow run completed.")
    print("A new directory 'mlruns' should now exist in this folder.")
