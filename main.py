# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load("linear_regression_model.pkl")

# Set MLflow Tracking URI (Make sure MLflow server is running)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Titanic_Salary_Prediction")

class Experience(BaseModel):
    years_experience: float

@app.get("/")
def read_root():
    return {"message": "Post only"}

@app.post("/predict")
def predict_salary(experience: Experience):
    try:
        # Prepare input for prediction
        input_data = np.array([[experience.years_experience]])

        with mlflow.start_run():
            # Make prediction
            predicted_salary = model.predict(input_data)[0]

            # Log input and output in MLflow
            mlflow.log_param("years_experience", experience.years_experience)
            mlflow.log_metric("predicted_salary", predicted_salary)

            return {"predicted_salary": predicted_salary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
