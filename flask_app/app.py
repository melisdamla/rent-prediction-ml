from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)
model = joblib.load("models/best_model.pkl")
LOG_FILE = "inference_log.csv"

# Valid categorical options (must match training set)
VALID_OPTIONS = {
    "agglomeration": ["Marseille", "Paris", "Lyon", "Bordeaux"],
    "Zone_complementaire": ["1", "2", "3", "4"],
    "Type_habitat": ["Appartement", "Maison"],
    "epoque_construction_homogene": [
        "Avant 1946",
        "1946 à 1970",
        "1971 à 1990",
        "1991 à 2005",
        "Après 2005"
    ]
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Required fields
        required_fields = [
            'surface', 'nombre_pieces', 'nombre_observations', 'nombre_logements',
            'agglomeration', 'Zone_complementaire', 'Type_habitat', 'epoque_construction_homogene'
        ]

        # Check presence
        for field in required_fields:
            if not request.form.get(field):
                return render_template("index.html", error=f"Missing field: {field.replace('_', ' ').capitalize()}")

        # Validate categorical fields
        for field, valid_values in VALID_OPTIONS.items():
            value = request.form.get(field)
            if value not in valid_values:
                return render_template("index.html", error=f"Invalid value for {field}: '{value}'")

        # Prepare input
        input_data = pd.DataFrame({
            "surface": [float(request.form['surface'])],
            "nombre_pieces": [int(request.form['nombre_pieces'])],
            "nombre_observations": [int(request.form['nombre_observations'])],
            "nombre_logements": [int(request.form['nombre_logements'])],
            "agglomeration": [request.form['agglomeration']],
            "Zone_complementaire": [request.form['Zone_complementaire']],
            "Type_habitat": [request.form['Type_habitat']],
            "epoque_construction_homogene": [request.form['epoque_construction_homogene']]
        })

        prediction = model.predict(input_data)[0]

        # Log inference
        log_df = input_data.copy()
        log_df["prediction"] = prediction
        if not os.path.exists(LOG_FILE):
            log_df.to_csv(LOG_FILE, index=False)
        else:
            log_df.to_csv(LOG_FILE, mode="a", index=False, header=False)

        return render_template("index.html", prediction=round(prediction, 2))

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template("index.html", error="Internal error during prediction.")

if __name__ == "__main__":
    app.run(debug=True)
