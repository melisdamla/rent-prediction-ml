"""
evaluate_models.py

Train and evaluate regression models on rent prediction data using a full
preprocessing pipeline (numerical + categorical handling).
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso

try:
    from xgboost import XGBRegressor
except ImportError:
    print("XGBoost is not installed. Please install it via 'pip install xgboost'")
    exit()

# Configuration
DATA_PATH = "data/loyers_clean.csv"
MODEL_DIR = "models"
RESULTS_PATH = os.path.join(MODEL_DIR, "evaluation_results.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# Features
NUMERICAL_FEATURES = ["surface", "nombre_observations", "nombre_logements"]
CATEGORICAL_FEATURES = [
    "nombre_pieces", "agglomeration", "zone_complementaire",
    "type_habitat", "epoque_construction_homogene"
]

def load_data(path):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["loyer_m2"])
    df["nombre_pieces"] = df["nombre_pieces"].astype(str)
    return df

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

def build_models(preprocessor):
    return {
        "linear_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]),
        "lasso": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", Lasso(alpha=0.01))
        ]),
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        "xgboost": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0))
        ])
    }

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred)
    }

def save_model(model, name, directory=MODEL_DIR):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"[✓] Saved model to: {path}")

def save_best_model(best_name, best_model):
    joblib.dump({
        "model": best_model.named_steps["regressor"],
        "preprocessor": best_model.named_steps["preprocessor"]
    }, BEST_MODEL_PATH)
    print(f"[🏆] Best model saved to: {BEST_MODEL_PATH}")

def main():
    df = load_data(DATA_PATH)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df["loyer_m2"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()
    models = build_models(preprocessor)

    results = []
    best_r2 = float("-inf")
    best_model = None
    best_name = ""

    for name, model in models.items():
        print(f"\n[Training] {name}")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append({"model": name, **metrics})
        save_model(model, name)

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model = model
            best_name = name

    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\n[✓] Evaluation results saved to: {RESULTS_PATH}")

    if best_model is not None:
        save_best_model(best_name, best_model)

if __name__ == "__main__":
    main()
