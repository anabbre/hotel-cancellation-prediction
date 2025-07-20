import pandas as pd
from pathlib import Path
from src.config import DATA_RAW, DATA_PROCESSED, TARGET_COLUMN
from sklearn.model_selection import train_test_split

# Cargar el CSV que resulta de limpiar datos nulos y devuelve un DataFrame
def load_raw() -> pd.DataFrame:

    if not DATA_RAW.exists():
        raise FileNotFoundError(f"No encuentro raw data en {DATA_RAW}")
    return pd.read_csv(DATA_RAW)

# Guardamos el CSV limpio
def save_processed(df: pd.DataFrame) -> None:

    DATA_PROCESSED.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(DATA_PROCESSED, index=False)

# Carga del CSV procesado
def load_processed() -> pd.DataFrame:

    if not DATA_PROCESSED.exists():
        raise FileNotFoundError(f"No encuentro processed data en {DATA_PROCESSED}")
    return pd.read_csv(DATA_PROCESSED)

# Devuelve (X_train, X_val, X_test, y_train, y_val, y_test).
def split_data(df: pd.DataFrame,
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = 42) -> tuple:

    # Separa el dataset en conjuntos de entrenamiento, validación y prueba según la variable objetivo

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Primero separamos test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Luego validación de lo que queda
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test