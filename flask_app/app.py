import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# === Load model ===
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/best_model.pkl"))
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
preprocessor = model_data["preprocessor"]

# === Load CSV for dropdowns ===
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../script/data/loyers_clean.csv"))
df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")

categorical_options = {
    "agglomeration": sorted(df["agglomeration"].dropna().unique()),
    "zone_complementaire": sorted(df["zone_complementaire"].dropna().unique()),
    "type_habitat": sorted(df["type_habitat"].dropna().unique()),
    "epoque_construction_homogene": sorted(df["epoque_construction_homogene"].dropna().unique())
}

# === Estimate standard deviation of residuals (for confidence interval) ===
features = ["surface", "nombre_pieces", "nombre_observations", "nombre_logements",
            "agglomeration", "zone_complementaire", "type_habitat", "epoque_construction_homogene"]
df = df.dropna(subset=features + ["loyer_m2"])
X_all = df[features]
y_all = df["loyer_m2"]
X_processed = preprocessor.transform(X_all)
y_pred_all = model.predict(X_processed)
residuals = y_all - y_pred_all
std_dev = np.std(residuals)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    conf_interval = None
    form_data = {}

    if request.method == "POST":
        form_data = {
            "surface": request.form.get("surface", ""),
            "nombre_pieces": request.form.get("nombre_pieces", ""),
            "nombre_observations": request.form.get("nombre_observations", ""),
            "nombre_logements": request.form.get("nombre_logements", ""),
            "agglomeration": request.form.get("agglomeration", ""),
            "zone_complementaire": request.form.get("zone_complementaire", ""),
            "type_habitat": request.form.get("type_habitat", ""),
            "epoque_construction_homogene": request.form.get("epoque_construction_homogene", "")
        }

        input_df = pd.DataFrame([form_data])
        for col in ["surface", "nombre_pieces", "nombre_observations", "nombre_logements"]:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

        X_transformed = preprocessor.transform(input_df)
        y_pred = model.predict(X_transformed)[0]

        # Confidence interval (95%)
        margin = 1.96 * std_dev
        prediction = round(y_pred, 2)
        conf_interval = (round(y_pred - margin, 2), round(y_pred + margin, 2))

    return render_template(
        "index.html",
        prediction=prediction,
        conf_interval=conf_interval,
        form_data=form_data,
        options=categorical_options
    )

if __name__ == "__main__":
    app.run(debug=True)
