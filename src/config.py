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

# Parámetros por defecto para el MLP y su grid de búsqueda
MLP_PARAMS = {
    "hidden_layer_sizes": (100,),
    "activation":          "relu",
    "solver":              "adam",
    "alpha":               1e-4,
    "learning_rate_init":  1e-3,
    "max_iter":            200,
    "random_state":        42
}

# Grids de hiperparámetros para tuning 
HYPERPARAM_GRIDS = {
    "decision_tree": {
        "criterion":         ["gini", "entropy"],
        "max_depth":         [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "class_weight":      [None, "balanced"]
    },
    "random_forest": {
        "n_estimators":        [100, 200, 500],
        "max_depth":           [None, 10, 20],
        "min_samples_split":   [2, 5, 10],
        "min_samples_leaf":    [1, 2, 4],
        "max_features":        ["sqrt", "log2", 0.6],
        "bootstrap":           [True, False],
        "class_weight":        [None, "balanced"]
    },
    "gradient_boost": {
        "n_estimators":     [100, 200, 500],
        "learning_rate":    [0.05, 0.1, 0.2],
        "max_depth":        [3, 5, 7],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma":            [0, 1, 5]
        # XGBoost no soporta class_weight directamente; se usaría scale_pos_weight si hay imbalance
    },
    "logistic_regression": {
        "penalty":    ["l1", "l2", "elasticnet", "none"],
        "C":          [0.01, 0.1, 1, 10],
        "solver":     ["saga"],    # soporta l1, l2 y elasticnet
        "l1_ratio":   [0, 0.5, 1],  # solo para elasticnet
        "class_weight":[None, "balanced"]
    },
    "mlp": {
        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "activation":         ["relu", "tanh"],
        "alpha":              [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 1e-2],
        # el resto se hereda de MLP_PARAMS
    }
}