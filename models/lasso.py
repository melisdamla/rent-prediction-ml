import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

DATA_FILE = "data/loyers_clean.csv"
df = pd.read_csv(DATA_FILE, encoding="ISO-8859-1")
df.columns = df.columns.str.strip()

numerical_cols = ["surface", "nombre_pieces", "nombre_observations", "nombre_logements"]
categorical_cols = ["agglomeration", "Zone_complementaire", "Type_habitat", "epoque_construction_homogene"]
target = "loyer_m2"
X = df[numerical_cols + categorical_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor = ColumnTransformer([("num", num_pipeline, numerical_cols), ("cat", cat_pipeline, categorical_cols)])

model_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", Lasso(alpha=0.01))])
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLasso Regression Performance:")
print(f"RMSE: {rmse:.2f}  |  MAE: {mae:.2f}  |  R²: {r2:.4f}")

residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, edgecolor="black", color="salmon")
plt.title("Residual Distribution (Lasso Regression)")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
