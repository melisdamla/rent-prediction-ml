# Rent Price Prediction — Machine Learning Pipeline & Web App

## Objective

This project aims to predict rent prices per square meter (€/m²) in France based on housing characteristics and urban zones. We implement a full ML pipeline using open public datasets (INSEE, Data.gouv) and deploy a web interface via Flask for real-time predictions.

---

## Project Structure

```
rent-prediction-ml/
├── data/                  # Cleaned datasets (excluded from Git)
├── models/                # Trained models (excluded from Git)
├── script/                # Data pipeline: obtain, scrub, explore
│   ├── loyers_obtain.py
│   ├── loyers_scrub.py
│   └── loyers_explore.py
├── models/
│   ├── linear_regression.py
│   ├── lasso.py
│   ├── random_forest.py
│   └── evaluate_models.py
├── flask_app/
│   ├── app.py
│   ├── templates/index.html
│   └── static/styles.css
├── notebooks/
│   └── eda.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Features

- **Data Collection**: Multi-source integration (INSEE, Data.gouv)
- **Preprocessing**: NaN handling, outlier filtering, feature engineering (`loyer_m2`)
- **Exploratory Analysis**: Histograms, boxplots, correlation heatmaps
- **Modeling**:
  - Linear Regression
  - Lasso Regression
  - Random Forest (GridSearchCV optimization)
- **Evaluation**: RMSE, MAE, R² metrics + residual visualizations
- **Deployment**: Flask web app with form-based input, validation, logging, and dynamic inference

---

## Usage

### 1. Setup
```bash
git clone https://github.com/melisdamla/rent-prediction-ml.git
cd rent-prediction-ml
pip install -r requirements.txt
```

### 2. Run Flask App
```bash
cd flask_app
python app.py
```
Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Model Training

To retrain or test models individually:
```bash
python models/linear_regression.py
python models/lasso.py
python models/random_forest.py
python models/evaluate_models.py
```

---

## Sample Input for Inference

```json
{
  "surface": 48.0,
  "nombre_pieces": 2,
  "nombre_observations": 50,
  "nombre_logements": 25,
  "agglomeration": "Marseille",
  "Zone_complementaire": "2",
  "Type_habitat": "Appartement",
  "epoque_construction_homogene": "Avant 1946"
}
```

---

## Tech Stack

- Python, Pandas, NumPy, Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Flask (Web app)
- HTML/CSS (Jinja2 templating)

---

## Deployment Notes

- Trained model file (`models/best_model.pkl`) is excluded via `.gitignore`.
- Input validation and inference logs are implemented inside `app.py`.

---

## Logs & Security

- Inference logs are stored in `inference_log.csv`.
- Input validation is implemented server-side to prevent crashes.
- Large files (>50MB) are removed from Git history via `git filter-repo`.

---

## Future Improvements

- Model versioning with MLflow
- Docker container for reproducible deployment
- Deployment to cloud (e.g., Heroku, GCP, AWS)
- Real-time map visualizations of rent predictions

---

## License

This project is distributed under the MIT License.
