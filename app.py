from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn

# Initialise app
app = FastAPI(title="Iris Prediction API")

# Select the best model on startup 
print("Finding the best model from MLFlow library...")
runs = mlflow.search_runs(experiment_ids=["0"])
best_run_id = runs.sort_values("metrics.accuracy", ascending=False).iloc[0].run_id
model_uri = f"runs:/{best_run_id}/random-forest-model"

print("Loading model...")
model = mlflow.sklearn.load_model(model_uri)
print("API is ready.")

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