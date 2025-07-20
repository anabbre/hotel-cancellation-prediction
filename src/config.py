from pathlib import Path

# Rutas

# Ruta del dataset original
DATA_RAW = Path("data/raw/dataset.csv")
# Ruta del dataset limpio
DATA_PROCESSED = Path("data/processed/dataset_limpio.csv")
# Carpeta para los modelos
MODEL_DIR = Path("models/")
# Ruta del modelo final
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"


# Columnas

# Columna objetivo
TARGET_COLUMN = "is_canceled"
# Columnas numéricas
NUM_FEATURES = []   # <-- PENDIENTES
# Columnas categóricas
CAT_FEATURES = []   # <-- PENDIENTES