import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sample_housing_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train(data_path=DATA):
    df = pd.read_csv(data_path)
    target = "price_inr"
    features = [c for c in df.columns if c != target]
    X, y = df[features], df[target]
    cat = X.select_dtypes(include=["object", "str"]).columns.tolist()
    num = [c for c in features if c not in cat]

    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.04, max_depth=3, random_state=42)
    }
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    results=[]
    fitted={}
    for name, model in models.items():
        pipe=Pipeline([("preprocess",pre),("model",model)])
        pipe.fit(Xtr,ytr)
        pred=pipe.predict(Xte)
        results.append({
            "model":name,
            "mae":mean_absolute_error(yte,pred),
            "rmse":mean_squared_error(yte,pred)**0.5,
            "r2":r2_score(yte,pred)
        })
        fitted[name]=pipe
    res=pd.DataFrame(results).sort_values(["mae","rmse"]).reset_index(drop=True)
    best=res.iloc[0]["model"]
    best_pipe=fitted[best]
    val_pred=best_pipe.predict(Xte)
    residual_abs=np.abs(yte.to_numpy()-val_pred)
    interval=float(np.quantile(residual_abs, .80))
    joblib.dump(best_pipe, MODEL_DIR/"house_price_model.joblib")
    joblib.dump({"features":features,"interval_80":interval,"best_model":best}, MODEL_DIR/"metadata.joblib")
    res.to_csv(MODEL_DIR/"metrics.csv", index=False)
    return res, best, interval

if __name__ == "__main__":
    r,b,i=train()
    print(r.to_string(index=False))
    print(f"\nBest model: {b}\n80% empirical error allowance: ₹{i:,.0f}")
