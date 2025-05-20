import os
import requests
import pandas as pd

# Output configuration
DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "loyers_raw.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# OpenDataSoft API endpoint
ODS_API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "resultats-nationaux-des-observatoires-locaux-des-loyers-france/exports/json"
)

# Load data from OpenDataSoft
try:
    print("Fetching OpenDataSoft data...")
    response = requests.get(ODS_API_URL)
    response.raise_for_status()
    df_ods = pd.DataFrame(response.json())
    print("OpenDataSoft data loaded successfully.")
except requests.exceptions.RequestException as e:
    print(f"Failed to fetch OpenDataSoft data: {e}")
    df_ods = pd.DataFrame()

# Data.gouv.fr sources (CSV files for years 2014–2024)
DATA_GOUV_CSV_URLS = [
    "https://www.data.gouv.fr/fr/datasets/r/13d660de-6108-4df6-8a54-1828a991a186",
    "https://www.data.gouv.fr/fr/datasets/r/3da6ba68-1f1e-4c49-a7e8-f521bd099599",
    "https://www.data.gouv.fr/fr/datasets/r/2ae4fb01-c69d-4a4d-bd09-f02c1b02882e",
    "https://www.data.gouv.fr/fr/datasets/r/422b3274-114d-4abe-850c-dfc0c69b981f",
    "https://www.data.gouv.fr/fr/datasets/r/f0f3abb1-2aed-4301-a44a-b14b43d73e1a",
    "https://www.data.gouv.fr/fr/datasets/r/5dcdae40-91b5-44ba-8ffc-65af94b61c6a",
    "https://www.data.gouv.fr/fr/datasets/r/4f1363af-28d9-4ec2-bdad-88f8b51fd7f2",
    "https://www.data.gouv.fr/fr/datasets/r/e30cb3ba-e1ca-4bec-b6ea-3cb751d6b862",
    "https://www.data.gouv.fr/fr/datasets/r/1fee314d-c278-424f-a029-a74d877eb185",
    "https://www.data.gouv.fr/fr/datasets/r/15d902ed-4dc3-457d-9c5d-bfe1151cb573",
    "https://www.data.gouv.fr/fr/datasets/r/42aaf838-46c9-4434-95a9-00173c6d4627",
]

# Load data from data.gouv.fr CSV files
df_gouv_list = []
for url in DATA_GOUV_CSV_URLS:
    try:
        print(f"Downloading: {url}")
        df = pd.read_csv(url, delimiter=";", encoding="ISO-8859-1", low_memory=False)
        df_gouv_list.append(df)
    except Exception as e:
        print(f"Error loading CSV from {url}: {e}")

# Concatenate all datasets
df_gouv = pd.concat(df_gouv_list, ignore_index=True) if df_gouv_list else pd.DataFrame()
df_loyers = pd.concat([df_ods, df_gouv], ignore_index=True)

# Clean column names
df_loyers.columns = df_loyers.columns.str.strip()

# Save result
if not df_loyers.empty:
    df_loyers.to_csv(RAW_FILE, index=False, encoding="ISO-8859-1")
    print(f"Combined dataset saved to '{RAW_FILE}' ({len(df_loyers)} rows).")
    print("First 5 rows:")
    print(df_loyers.head())
else:
    print("No valid data collected. Check sources or connection.")
