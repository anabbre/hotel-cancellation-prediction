from pathlib import Path

# Rutas

# Ruta base del proyecto (dos niveles por encima de este archivo)
BASE_DIR = Path(__file__).resolve().parent.parent
# Ruta del dataset original
# Rutas de datos
DATA_RAW = BASE_DIR / "data" / "raw" / "dataset_practica_final.csv"
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "dataset_limpio.csv"
# Rutas para los modelos
MODEL_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"

# Columnas

# Columna objetivo
TARGET_COLUMN = "is_canceled"

# Variables numéricas
NUM_FEATURES = [
    "lead_time", "stays_in_week_nights", "stays_in_weekend_nights",
    "adults", "children", "babies", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes",
    "days_in_waiting_list", "adr", "required_car_parking_spaces",
    "total_of_special_requests", "arrival_month_num", "length_of_stay"
]

# Variables categóricas
CAT_FEATURES = [
    "hotel", "meal", "country", "market_segment",
    "distribution_channel", "is_repeated_guest",
    "deposit_type", "customer_type", "season", "arrival_quarter"
]