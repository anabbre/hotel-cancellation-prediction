import pandas as pd
from pathlib import Path
from src.config import DATA_PROCESSED, TARGET_COLUMN
from sklearn.model_selection import train_test_split

# Cargar el CSV que resulta de limpiar datos nulos y devuelve un DataFrame
def load_dataset(path: Path = DATA_PROCESSED) -> pd.DataFrame:

    # Control de errores: si no existe en la ruta correcta
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset limpio en {path}")
    return pd.read_csv(path)

# Devuelve el dataset limpio separando X e Y y haz train_test_split
def load_split_data(test_size: float = 0.2,
                    random_state: int = 42):

    # 1. Carga el DataFrame
    df = load_dataset()

    # 2. Separa en variables predictoras y target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 3. Divide en train y test
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
