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
rent-prediction-ml/
├── flask_app/
│   ├── app.py                  # Flask web server
│   ├── templates/index.html    # HTML frontend
│   ├── static/styles.css       # Styling
│   └── inference_log.csv       # Optional prediction logs
│
├── models/
│   ├── best_model.pkl          # Deployed model
│   ├── *_model.pkl             # Trained models
│   └── evaluation_results.csv  # Comparison of model performance
│
├── script/
│   ├── data/
│   │   ├── loyers_raw.csv      # Combined raw data
│   │   └── loyers_clean.csv    # Cleaned dataset
│   ├── loyers_obtain.py        # Downloads raw datasets
│   ├── loyers_scrub.py         # Preprocessing script
│   ├── loyers_explore.py       # EDA and correlation analysis
│   └── evaluate_models.py      # Trains and evaluates models
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt

2. Train and Evaluate Models
python script/evaluate_models.py
This will evaluate multiple models and save the best one as models/best_model.pkl.

3. Launch the Web App
cd flask_app
python app.py
Visit http://localhost:5000 to access the form interface and make predictions.

Docker (Optional)

To containerize the application:
docker build -t rent-predictor .
docker run -p 5000:5000 rent-predictor

With Docker Compose:
docker-compose up --build

Model Details

The application evaluates the following regressors:
	•	Linear Regression
	•	Lasso
	•	Random Forest Regressor
	•	XGBoost Regressor

Each model is wrapped in a Scikit-learn Pipeline with:
	•	SimpleImputer for missing values
	•	StandardScaler for numeric features
	•	OneHotEncoder for categorical variables
	•	ColumnTransformer to unify preprocessing

The model with the lowest RMSE is selected and saved.

⸻

Web Interface

The app supports:
	•	Input of housing features via an HTML form
	•	Dynamic dropdowns populated from the training dataset
	•	Rent prediction output with a 95% confidence interval (± 1.96 × residual std)

Example output:
Prédiction estimée : 12.4 €/m²
Intervalle de confiance : [10.7 – 14.1]


Future Improvements
	•	REST API for programmatic access
	•	File upload for batch prediction
	•	Deployment to cloud platforms (e.g., GCP, Heroku, Render)
	•	Interactive map of predictions by region

## License

This project was developed by Melis Damla Sahin as part of a coursework at Aix-Marseille University (2025).  
It is shared for educational and academic demonstration purposes.  
All rights reserved unless explicitly stated otherwise.
