import numpy as np
from fastapi import FastAPI
from app.train import train_model
from pydantic import BaseModel

app = FastAPI()

MODEL = None
SCALER = None

@app.get("/train")
def train_endpoint():
    global MODEL, SCALER
    MODEL, SCALER, metrics = train_model()
    return metrics

@app.get("/predict")
def predict(
        hydraulic_pressure: float,
        coolant_pressure: float,
        air_system_pressure: float,
        coolant_temperature: float,
        hydraulic_oil_temperature: float,
        spindle_bearing_temperature: float,
        spindle_vibration: float,
        tool_vibration: float,
        spindle_speed: int,
        voltage: int,
        torque: float,
        cutting: float
):
    input_values = np.array([
        hydraulic_pressure,
        coolant_pressure,
        air_system_pressure,
        coolant_temperature,
        hydraulic_oil_temperature,
        spindle_bearing_temperature,
        spindle_vibration,
        tool_vibration,
        spindle_speed,
        voltage,
        torque,
        cutting
    ]).reshape(1, -1)

    # Apply scaling to the input data using the trained scaler
    scaled_input = SCALER.transform(input_values)

    # Perform prediction using the trained model
    prediction = MODEL.predict(scaled_input)

    # Convert numeric prediction to meaningful string output
    result = "Machine_Failure" if prediction[0] == 1 else "No_Machine_Failure"

    return {"prediction": result}
