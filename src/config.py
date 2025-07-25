from pathlib import Path

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos
DATA_RAW       = ROOT_DIR / "data" / "raw"       / "dataset_practica_final.csv"
DATA_PROCESSED = ROOT_DIR / "data" / "processed" / "dataset_limpio.csv"

# Directorios de salida
MODEL_DIR   = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Parámetros generales
TARGET_COLUMN = "is_canceled"
RANDOM_STATE  = 42

# Features para el preprocesado
NUM_FEATURES = [
    "lead_time",
    "arrival_date_week_number",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "agent",
    "room_changed",
    "arrival_month_num",
    "arrival_quarter",
    "length_of_stay",
]

CAT_FEATURES = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "season",
]

# Parámetros para la red MLP con TensorFlow/Keras
MLP_PARAMS = {
    "hidden_layers":   [128, 64],
    "activation":      "relu",
    "optimizer":       "adam",
    "metric":          "AUC",
    "batch_size":      32,
    "epochs":          20,
    "validation_split": 0.2,
    "random_seed":     42
}

# Parámetros de tuning
METRIC = "roc_auc"
CV     = 5
N_JOBS = -1

TUNE_PARAMS = {
    "decision_tree": {
        "max_depth":         [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "logistic_regression": {
        "C":        [0.01, 0.1, 1, 10],
        "penalty":  ["l2"],
        "solver":   ["liblinear"],
        "max_iter": [100, 200],
    },
    "gradient_boost": {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 5],
        "learning_rate": [0.01, 0.1],
        "subsample":     [0.8, 1.0],
    },
    "random_forest": {
        "n_estimators":  [100, 200],
        "max_depth":     [None, 10, 20],
        "max_features":  ["sqrt", "log2"],
    },
}

# Lista de modelos a visualizar/evaluar
MODEL_NAMES = [
    "decision_tree",
    "logistic_regression",
    "gradient_boost",
    "random_forest",
    "mlp_tf",
]
