import os
import pandas as pd

DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "loyers_raw.csv")
CLEAN_FILE = os.path.join(DATA_DIR, "loyers_clean.csv")

if not os.path.exists(RAW_FILE):
    print(f"❌ Fichier introuvable : {RAW_FILE}")
    exit()

df = pd.read_csv(RAW_FILE, delimiter=",", encoding="ISO-8859-1", low_memory=False)
print("✅ Données brutes chargées avec succès.")
print(f"🔍 Dimensions initiales : {df.shape}")

column_mapping = {
    "loyer_median": "loyer",
    "surface_moyenne": "surface",
    "nombre_pieces_homogene": "nombre_pieces"
}
df = df.rename(columns=column_mapping)

def extraire_nombre_pieces(val):
    if isinstance(val, str):
        for part in val.split():
            try:
                return float(part.replace('P', ''))
            except:
                continue
    try:
        return float(val)
    except:
        return None

df["nombre_pieces"] = df["nombre_pieces"].apply(extraire_nombre_pieces)

required_columns = ["loyer", "surface", "nombre_pieces"]
for col in required_columns:
    if col not in df.columns:
        print(f"❌ Colonne manquante : {col}")
        exit()

df = df.dropna(subset=required_columns)
print(f"✅ Après suppression des NaN : {df.shape}")

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)
print(f"✅ Après conversion numérique : {df.shape}")

df = df[(df["loyer"] > 5) & (df["loyer"] < 10000)]
df = df[(df["surface"] > 5) & (df["surface"] < 500)]
print(f"✅ Après suppression des outliers : {df.shape}")

df["loyer_m2"] = df["loyer"] / df["surface"]

if len(df) == 0:
    print("❌ Plus aucune donnée après nettoyage. Vérifie les filtres.")
else:
    df.to_csv(CLEAN_FILE, index=False, encoding="ISO-8859-1")
    print(f"✅ Données nettoyées sauvegardées dans '{CLEAN_FILE}' ({len(df)} lignes).")
