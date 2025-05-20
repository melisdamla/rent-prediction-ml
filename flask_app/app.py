from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)
model = joblib.load("models/best_model.pkl")
LOG_FILE = "inference_log.csv"

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        required_fields = [
            'surface', 'nombre_pieces', 'nombre_observations', 'nombre_logements',
            'agglomeration', 'Zone_complementaire', 'Type_habitat', 'epoque_construction_homogene'
        ]

        for field in required_fields:
            if not request.form.get(field):
                print(f"[VALIDATION ERROR] Missing field: {field}")
                return render_template("index.html")

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

        log_df = input_data.copy()
        log_df["prediction"] = prediction
        if not os.path.exists(LOG_FILE):
            log_df.to_csv(LOG_FILE, index=False)
        else:
            log_df.to_csv(LOG_FILE, mode="a", index=False, header=False)

        return render_template("index.html", prediction=round(prediction, 2))

    except Exception as e:
        print(f"[EXCEPTION] {e}")
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)