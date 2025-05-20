# Rent Price Prediction in France (€/m²)

This project aims to predict the rent price per square meter using open public datasets, implementing a full data science pipeline from acquisition to modeling and error analysis.

## Objectives

- Integrate and standardize multiple open data sources
- Build a machine learning pipeline for rent price prediction (€/m²)
- Compare model performance using MAE, RMSE, and R²
- Identify and analyze major prediction errors

## Project Structure

```
rent-prediction-ml/
├── data/
│   ├── loyers_raw.csv
│   └── loyers_clean.csv
├── figures/
│   ├── hist_loyer_m2.png
│   ├── boxplot_type_habitat.png
│   ├── heatmap_correlation.png
│   └── importances_rf_lasso.png
├── top_erreurs_random_forest.xlsx
├── loyers_obtain.py
├── loyers_scrub.py
├── loyers_explore.py
├── loyers_modelisation.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Datasets

- OpenDataSoft (via JSON API)
- Data.gouv.fr (CSV files, 2014–2024)

## Technologies

- **Language**: Python 3.10
- **Libraries**: pandas, numpy, scikit-learn, seaborn, matplotlib
- **Tools**: Google Colab (collaborative development), Excel (manual error review)

## Models Used

- Linear Regression (baseline)
- Lasso Regression (feature selection)
- Random Forest Regressor (main model)

## Methodology (OSEMN)

1. **Obtain** – Automated data extraction via scripts (`loyers_obtain.py`)
2. **Scrub** – Statistical and semantic cleaning of raw data (`loyers_scrub.py`)
3. **Explore** – Data visualization and correlation analysis (`loyers_explore.py`)
4. **Model** – Model training with pipelines and cross-validation (`loyers_modelisation.py`)
5. **Interpret** – Error distribution, outlier analysis, and result export

## Results

- **MAE** ≈ 78.6 €
- **RMSE** ≈ 123.4 €
- **R²** ≈ 0.995 (Random Forest)

## Author

**Melis Damla Sahin**  
Data Engineer — responsible for automated data collection, statistical and semantic cleaning, feature preparation, and technical analysis of model errors.