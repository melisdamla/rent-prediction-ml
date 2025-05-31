"""
linear_regression.py

Train a Linear Regression model on rental price data using a full preprocessing pipeline.
Saves the trained model and prints evaluation metrics.

Input:  data/loyers_clean.csv
Output: models/linear_regression_model.pkl
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# === Configuration ===
DATA_FILE = "data/loyers_clean.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "linear_regression_model.pkl")
TARGET = "loyer_m2"

NUMERIC_FEATURES = ["surface", "nombre_observations", "nombre_logements"]
CATEGORICAL_FEATURES = [
    "nombre_pieces",
    "agglomeration",
    "zone_complementaire",
    "type_habitat",
    "epoque_construction_homogene"
]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip().str.lower()
    return df.dropna(subset=[TARGET])

def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", LinearRegression())
    ])
    return pipeline

def evaluate(y_true, y_pred):
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R²:   {r2_score(y_true, y_pred):.3f}")

def main():
    print("[INFO] Loading data...")
    df = load_data(DATA_FILE)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[INFO] Building pipeline...")
    pipeline = build_pipeline()

    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = pipeline.predict(X_test)
    evaluate(y_test, y_pred)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"[✓] Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()