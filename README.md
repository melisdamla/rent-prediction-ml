# Rent Prediction ML Web App

This project is a full-stack machine learning application designed to predict rental prices per square meter (`loyer_m2`) in France. It integrates data preprocessing, regression model training and evaluation, and a Flask-based web interface to serve real-time predictions.

---

## Features

- End-to-end ML pipeline: from raw data to deployed model
- Multiple model training: Linear Regression, Lasso, Random Forest, XGBoost
- Preprocessing with imputation, scaling, and one-hot encoding
- Model evaluation using MAE, RMSE, and R²
- Web form interface with dropdowns based on training data
- Confidence intervals computed from model residuals
- Docker support for reproducibility and deployment

---

## Project Structure

```
rent-prediction-ml/
├── flask_app/
│   ├── app.py
│   ├── inference_log.csv
│   ├── static/
│   │   └── styles.css
│   └── templates/
│       └── index.html
│
├── models/
│   ├── best_model.pkl
│   ├── evaluation_results.csv
│   ├── lasso_model.pkl
│   ├── lasso.pkl
│   ├── linear_regression.pkl
│   ├── random_forest_model.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   └── xgboost.pkl
│
├── script/
│   ├── data/
│   │   ├── loyers_clean.csv
│   │   └── loyers_raw.csv
│   ├── loyers_explore.py
│   ├── loyers_obtain.py
│   └── loyers_scrub.py
│
├── lasso.py
├── linear_regression.py
├── random_forest.py
├── xgboost_model.py
├── evaluate_models.py
├── eda.ipynb
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and Evaluate Models

```bash
python script/evaluate_models.py
```

This will train and evaluate multiple models, then save the best one as `models/best_model.pkl`.

### 3. Launch the Web App

```bash
cd flask_app
python app.py
```

Visit `http://localhost:5000` to access the form interface and make predictions.

---

## Docker (Optional)

To containerize the application:

```bash
docker build -t rent-predictor .
docker run -p 5000:5000 rent-predictor
```

With Docker Compose:

```bash
docker-compose up --build
```

---

## Model Details and Performance

The application evaluates the following regression models: Linear Regression, Lasso, Random Forest Regressor, and XGBoost Regressor.

Model evaluation was performed using 5-fold cross-validation. The results are summarized in the table below:

| Model             | MAE   | RMSE  | R² Score |
|-------------------|-------|-------|----------|
| Linear Regression | 0.651 | 0.940 | 0.900    |
| Lasso             | 0.842 | 1.185 | 0.842    |
| Random Forest     | 0.328 | 0.497 | 0.972    |
| XGBoost           | 0.577 | 0.757 | 0.935    |

The Random Forest Regressor achieved the best performance across all metrics.

Each model is wrapped in a Scikit-learn `Pipeline` including:

- `SimpleImputer` for missing values
- `StandardScaler` for numeric features
- `OneHotEncoder` for categorical variables
- `ColumnTransformer` to apply preprocessing in parallel

The model with the lowest **RMSE** is selected and saved.

---


## Web Interface

The Flask app provides:

- An HTML form for inputting housing features
- Dynamic dropdowns populated from training data
- Real-time rent prediction with 95% confidence interval

**Example Output**:

```
Prédiction estimée : 12.4 €/m²
Intervalle de confiance : [10.7 – 14.1]
```

---

## Future Improvements

- REST API for programmatic access
- File upload for batch prediction
- Deployment to cloud platforms (e.g., GCP, Heroku, Render)
- Interactive map of predicted rents by region

---

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Attribution
This project was developed by **Melis Damla Sahin** as part of coursework at **Aix-Marseille University (2025)**.

It is shared for educational and academic demonstration purposes.  
All rights reserved unless explicitly stated otherwise.

