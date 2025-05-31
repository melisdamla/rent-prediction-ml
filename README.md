# Rent Prediction ML Web App

This project is a full-stack machine learning application designed to predict rental prices per square meter (`loyer_m2`) in France. It combines data preprocessing, regression model training, evaluation, and a web interface built with Flask to serve real-time predictions.

---

## Features

- Cleans and preprocesses multi-source housing data
- Trains multiple regression models (Linear, Lasso, Random Forest, XGBoost)
- Selects the best model using cross-validated performance metrics (MAE, RMSE, R²)
- Serves predictions via a web interface with dropdowns linked directly to training data
- Displays prediction confidence intervals (± std deviation of residuals)
- Docker support for deployment

---

## Folder Structure

rent-prediction-ml/
├── flask_app/
│ ├── app.py # Flask application logic
│ ├── templates/index.html # Web form UI
│ ├── static/styles.css # Custom styling
│ └── inference_log.csv # Optional logging of predictions
│
├── models/
│ ├── best_model.pkl # Selected model for deployment
│ ├── *_model.pkl # Individual models
│ └── evaluation_results.csv # Evaluation metrics for all models
│
├── script/
│ ├── data/
│ │ ├── loyers_raw.csv # Merged raw data
│ │ └── loyers_clean.csv # Cleaned dataset for modeling
│ ├── loyers_obtain.py # Download and combine datasets
│ ├── loyers_scrub.py # Clean and preprocess data
│ ├── loyers_explore.py # Exploratory data analysis
│ └── evaluate_models.py # Model training and evaluation
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

---

## Usage

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
Step 2: Train Models
python script/evaluate_models.py
This script preprocesses the data, evaluates four regression models, and saves the best model to models/best_model.pkl.

Step 3: Run Flask App
cd flask_app
python app.py
The app runs on http://localhost:5000. Users can input housing characteristics via form fields to receive rent/m² predictions, including a confidence interval.

Docker Usage
To build and run using Docker:


docker build -t rent-predictor .
docker run -p 5000:5000 rent-predictor
Or use Docker Compose:


docker-compose up --build
Model Selection
The following models are evaluated:

LinearRegression

Lasso

RandomForestRegressor

XGBRegressor

All models are integrated with a shared preprocessing pipeline using ColumnTransformer, including:

Imputation (mean for numeric, most frequent for categorical)

Standard scaling

One-hot encoding

The best model is chosen based on RMSE and saved as best_model.pkl.

Web Interface
Uses dropdown menus for categorical inputs (auto-populated from the dataset)

Shows prediction with ± confidence interval (computed from training residuals)

Optional logging of predictions to inference_log.csv

Future Improvements
Add REST API endpoint

Add map visualizations of predicted rents by region

Implement upload and batch prediction

Integrate model monitoring or auto-retraining

License
MIT License. This project is open for learning, improvement, and reuse.

---

Let me know if you'd like a version:
- In French  
- With Markdown badges  
- With screenshots  
- Or converted to PDF for academic/report submission.
