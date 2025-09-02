from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# loading the saved model
model = joblib.load("model.pkl")

# defining the structure of the input data using pydantic


class ModelInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


app = FastAPI()

# loading the model at startup
model = joblib.load("model.pkl")


@app.post('/predict')
def predict(data: ModelInput):
    # converting the input data to a numpy array for sklearn
    input_data = np.array([[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]])

    # using the model to predict whether its heart disease or not
    prediction = model.predict(input_data)

    # returniong the prediction result as JSON
    return {'prediction': int(prediction[0])}
