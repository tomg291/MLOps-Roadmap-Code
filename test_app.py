import os

# 1. Tell the local Windows MLflow to allow folder-based tracking
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# 2. Delete any lingering Docker paths from the Windows terminal memory
if "MLFLOW_TRACKING_URI" in os.environ:
    del os.environ["MLFLOW_TRACKING_URI"]


from fastapi.testclient import TestClient
from app import app

def test_predict_endpoint():
    """
    Test that /predict endpoint returns a 200 success code and a predicted_class value
    """

    with TestClient(app) as client:
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }

        response = client.post("/predict", json=payload)

        # Assertions must all be passed for the test to pass

        # Check server is not crashed 
        assert response.status_code == 200

        # Extract response
        data = response.json()

        # Check API has returned the predicted class 
        assert "predicted_class" in data

        # Check the prediction is one of the valid classes (i.e. 0, 1, or 2)
        assert isinstance(data["predicted_class"], int)
        assert data["predicted_class"] in [0,1,2]