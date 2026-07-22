from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import os

# Set MLflow tracking URI
#mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns"))

# Initialise app
app = FastAPI(title="Iris Prediction API")

# Global model variable
model = None

@app.on_event("startup")
async def load_model_on_startup():
    global model
    print("Finding the best model from MLFlow library...")
    try:
        runs = mlflow.search_runs(experiment_ids=["0"])
        if len(runs) == 0:
            raise Exception("No runs found")
        
        # Try to sort by accuracy if available, otherwise use the first run
        if "metrics.accuracy" in runs.columns:
            best_run_id = runs.sort_values("metrics.accuracy", ascending=False).iloc[0].run_id
        else:
            print("metrics.accuracy not found, using latest run")
            best_run_id = runs.iloc[0].run_id
            
        model_uri = f"runs:/{best_run_id}/random-forest-model"
        
        print("Loading model...")
        model = mlflow.sklearn.load_model(model_uri)
        print("API is ready.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using dummy model for now...")
        # Load any available model as fallback
        runs = mlflow.search_runs(experiment_ids=["0"])
        if len(runs) > 0:
            best_run_id = runs.iloc[0].run_id
            model_uri = f"runs:/{best_run_id}/random-forest-model"
            model = mlflow.sklearn.load_model(model_uri)

# Define the required form of incoming data 
class FlowerFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create the endpoint
@app.post("/predict")
def predict_flower(features: FlowerFeatures):
    # Convert the incoming JSON data into 2D array
    data = [[
        features.sepal_length, 
        features.sepal_width, 
        features.petal_length, 
        features.petal_width
    ]]

    prediction = model.predict(data)
    
    return {"predicted_class": int(prediction[0])}
