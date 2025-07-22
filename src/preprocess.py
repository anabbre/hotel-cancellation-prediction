# src/preprocess.py

import pandas as pd
from pathlib import Path
import joblib
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.data_loader import load_processed
from src.config import TARGET_COLUMN, NUM_FEATURES, CAT_FEATURES, MODEL_DIR

# Si save_transformer=True: fit_transform + dump(preprocessor); si save_transformer=False: load(preprocessor) + transform Devuelve (X_processed, y)
def preprocess(df_clean: pd.DataFrame, save_transformer: bool = True):
    # Separa X e y
    X = df_clean.drop(columns=[TARGET_COLUMN], errors="ignore")
    y = df_clean[TARGET_COLUMN] if TARGET_COLUMN in df_clean else None

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    # OHE con argumento correcto según versión
    if sklearn.__version__ >= "1.2":
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe",      ohe),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUM_FEATURES),
        ("cat", cat_pipeline, CAT_FEATURES),
    ], remainder="drop")

    # Fit/Transform o solo Transform
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"

    if save_transformer:
        X_processed = preprocessor.fit_transform(X)
        joblib.dump(preprocessor, preprocessor_path)
    else:
        preprocessor = joblib.load(preprocessor_path)
        X_processed = preprocessor.transform(X)

    return X_processed, y


if __name__ == "__main__":
    # Prueba de sanity
    df = load_processed()
    Xp, y = preprocess(df, save_transformer=True)
    print(f"✔ Preprocesado OK — Xp shape: {Xp.shape}, y shape: {y.shape}")
