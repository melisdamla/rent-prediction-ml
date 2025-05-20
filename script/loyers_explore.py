import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# === Configuration ===
DATA_DIR = "data"
CLEAN_FILE = os.path.join(DATA_DIR, "loyers_clean.csv")

# === Load dataset ===
df = pd.read_csv(CLEAN_FILE, encoding="ISO-8859-1")
df.columns = df.columns.str.strip()
print("Cleaned data loaded:", df.shape)

# === Basic statistics ===
print("\nDescriptive statistics for 'loyer_m2':")
print(df["loyer_m2"].describe())

# === Distribution of target ===
plt.figure(figsize=(10, 5))
sns.histplot(df["loyer_m2"], kde=True, bins=30, color="steelblue", edgecolor="black")
plt.title("Distribution du loyer au m²")
plt.xlabel("Loyer (€/m²)")
plt.ylabel("Nombre de logements")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Boxplot: loyer_m2 by type of housing ===
plt.figure(figsize=(10, 6))
sns.boxplot(x="Type_habitat", y="loyer_m2", data=df)
plt.title("Loyer au m² par type d'habitat")
plt.ylabel("Loyer (€/m²)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# === Boxplot: loyer_m2 by number of rooms ===
plt.figure(figsize=(12, 6))
sns.boxplot(x="nombre_pieces", y="loyer_m2", data=df)
plt.title("Loyer au m² selon le nombre de pièces")
plt.xlabel("Nombre de pièces")
plt.ylabel("Loyer (€/m²)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Correlation matrix ===
numeric_cols = ["loyer", "surface", "nombre_pieces", "nombre_observations", "nombre_logements", "loyer_m2"]
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corrélation des variables numériques")
plt.tight_layout()
plt.show()

# === Feature importance via RandomForest & Lasso ===
numerical_cols = ["surface", "nombre_pieces", "nombre_observations", "nombre_logements"]
categorical_cols = ["agglomeration", "Zone_complementaire", "Type_habitat", "epoque_construction_homogene"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)
])

X = df[numerical_cols + categorical_cols]
y = df["loyer_m2"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_transformed = preprocessor.fit_transform(X_train)
feature_names = preprocessor.get_feature_names_out()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
lasso = Lasso(alpha=0.01)

rf.fit(X_train_transformed, y_train)
lasso.fit(X_train_transformed, y_train)

importance_df = pd.DataFrame({
    "Variable": feature_names,
    "Random Forest": rf.feature_importances_,
    "Lasso": np.abs(lasso.coef_)
}).sort_values(by="Random Forest", ascending=False).head(10)

# === Plot feature importances ===
plt.figure(figsize=(12, 6))
importance_df.set_index("Variable").plot(kind="bar", figsize=(12, 6))
plt.title("Top 10 - Importance des variables (Random Forest & Lasso)")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()
