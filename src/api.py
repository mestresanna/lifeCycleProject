from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("../models/saved_model.pkl")

# Features your model expects
FEATURES = [
    'open', 'close', 'min', 'max', 'avg', 'quantity', 'volume',
    'ibovespa_close', 'day_of_week', 'daily_return', 'price_range', 'volume_per_quantity',
    'rolling_close_5', 'rolling_std_5', 'rolling_return_5', 'momentum_5', 'rolling_volume_5'
]

# Pydantic model for request validation
class StockFeatures(BaseModel):
    open: float
    close: float
    min: float
    max: float
    avg: float
    quantity: float
    volume: float
    ibovespa_close: float
    day_of_week: int
    daily_return: float
    price_range: float
    volume_per_quantity: float
    rolling_close_5: float
    rolling_std_5: float
    rolling_return_5: float
    momentum_5: float
    rolling_volume_5: float

app = FastAPI(title="Stock Prediction API")

@app.get("/")
def read_root():
    return {"message": "Stock Prediction API is live"}

@app.post("/predict")
def predict(features: StockFeatures):
    # Convert to DataFrame (single row)
    df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(df)[0]
    return {"predicted_target": prediction}
