from pathlib import Path
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODEL=joblib.load(ROOT/"models"/"house_price_model.joblib")
META=joblib.load(ROOT/"models"/"metadata.joblib")

def predict(row: dict):
    X=pd.DataFrame([row])
    price=float(MODEL.predict(X)[0])
    interval=float(META["interval_80"])
    return max(0, price), max(0, price-interval), price+interval
