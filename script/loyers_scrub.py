import os
import pandas as pd

# Define file paths
DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "loyers_raw.csv")
CLEAN_FILE = os.path.join(DATA_DIR, "loyers_clean.csv")

# Check if raw data file exists
if not os.path.exists(RAW_FILE):
    print(f"File not found: {RAW_FILE}")
    exit()

# Load raw data
df = pd.read_csv(RAW_FILE, delimiter=",", encoding="ISO-8859-1", low_memory=False)
print("Raw data loaded successfully.")
print(f"Initial shape: {df.shape}")

# Rename important columns
column_mapping = {
    "loyer_median": "loyer",
    "surface_moyenne": "surface",
    "nombre_pieces_homogene": "nombre_pieces"
}
df = df.rename(columns=column_mapping)

# Extract number of rooms from text
def extract_number_of_rooms(val):
    if isinstance(val, str):
        for part in val.split():
            try:
                return float(part.replace("P", ""))
            except:
                continue
    try:
        return float(val)
    except:
        return None

df["nombre_pieces"] = df["nombre_pieces"].apply(extract_number_of_rooms)

# Check required columns
required_columns = ["loyer", "surface", "nombre_pieces"]
for col in required_columns:
    if col not in df.columns:
        print(f"Missing required column: {col}")
        exit()

# Drop rows with missing values
df = df.dropna(subset=required_columns)
print(f"After dropping rows with missing values: {df.shape}")

# Convert to numeric types
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)
print(f"After numeric conversion: {df.shape}")

# Filter outliers
df = df[(df["loyer"] > 5) & (df["loyer"] < 10000)]
df = df[(df["surface"] > 5) & (df["surface"] < 500)]
print(f"After filtering outliers: {df.shape}")

# Standardize categorical column values
if "epoque_construction_homogene" in df.columns:
    df["epoque_construction_homogene"] = df["epoque_construction_homogene"].str.strip()
    df["epoque_construction_homogene"] = df["epoque_construction_homogene"].replace({
        "2. Entre 1991-2005": "4. Entre 1991-2005"
    })

    valid_values = [
        "1. Avant 1946",
        "2. Entre 1946-1970",
        "3. Entre 1971-1990",
        "4. Entre 1991-2005",
        "5. Après 2005"
    ]
    df = df[df["epoque_construction_homogene"].isin(valid_values)]

# Create target variable
df["loyer_m2"] = df["loyer"] / df["surface"]

# Save cleaned dataset
if len(df) == 0:
    print("No data remaining after cleaning. Please review the filters.")
else:
    df.to_csv(CLEAN_FILE, index=False, encoding="ISO-8859-1")
    print(f"Cleaned dataset saved to '{CLEAN_FILE}' with {len(df)} rows.")
