import pandas as pd
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.data_loader import load_processed

from src.config import (
    TARGET_COLUMN,
    NUM_FEATURES,
    CAT_FEATURES,
    MODEL_DIR,
)

def preprocess(df_clean: pd.DataFrame, save_transformer: bool = True):
    # Separa X e y de df_clean
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]

    # Pipelines
    # num: imputa (mean) + estandariza
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    # cat: OHE con ignore unknown
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUM_FEATURES),
        ("cat", cat_pipeline, CAT_FEATURES),
    ], remainder="drop")

    # Ajusta y transforma ColumnTransformer
    X_processed = preprocessor.fit_transform(X)

    # Si save_transformer=True, guarda el transformer 
    if save_transformer:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    return X_processed, y

if __name__ == "__main__":
    # Prueba rápida desde consola:
    df = load_processed()
    Xp, y  = preprocess(df)
    print(f"✔ Preprocesado OK — Xp shape: {Xp.shape}, y shape: {y.shape}")