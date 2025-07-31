import pandas as pd
from pathlib import Path
from src.config import DATA_RAW, DATA_PROCESSED, TARGET_COLUMN
from sklearn.model_selection import train_test_split
from src.clean_data import clean_dataset


# Carga el CSV original
def load_raw() -> pd.DataFrame:
    """
    Carga el dataset crudo original desde la ruta configurada.

    Raises:
        FileNotFoundError: Si el archivo de datos crudos no se encuentra.

    Returns:
        pd.DataFrame: El DataFrame cargado.
    """
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"No encuentro raw data en {DATA_RAW}")
    return pd.read_csv(DATA_RAW)

#Guarda el DataFrame procesado en DATA_PROCESSED
def save_processed(df: pd.DataFrame) -> None:
    """
    Guarda el DataFrame procesado en la ruta especificada.

    Crea el directorio si no existe.

    Args:
        df (pd.DataFrame): El DataFrame a guardar.
    """   
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False)
    print(f"✔ Dataset limpio guardado en {DATA_PROCESSED}")

#Carga el dataset ya procesado. Si no existe, invoca a clean_dataset para generarlo desde el raw
def load_processed() -> pd.DataFrame:
    """
    Carga el dataset ya procesado desde disco. Si no existe el archivo,
    genera primero el dataset llamando a la función `clean_dataset` 
    previamente importada desde src.clean_data.

    Returns:
        pd.DataFrame: El DataFrame procesado cargado.
    """
if not DATA_PROCESSED.exists():
    print(f"⚠️ Dataset procesado no encontrado en {DATA_PROCESSED}, generando desde raw…")
    df = clean_dataset()

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
):
    """
    Divide un DataFrame en conjuntos de entrenamiento, validación y prueba.

    Realiza una división estratificada de los datos.

    Args:
        df (pd.DataFrame): DataFrame de entrada que contiene características y la variable objetivo.
        test_size (float, optional): Proporción del dataset a usar para el conjunto de prueba.
                                    Por defecto es 0.2.
        val_size (float, optional): Proporción del dataset a usar para el conjunto de validación.
                                    Por defecto es 0.2.
        random_state (int, optional): Semilla para la reproducibilidad de la división. Por defecto es 42.

    Returns:
        tuple: Una tupla que contiene (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Divide el DataFrame en train/val/test 
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Sanity check
    df = load_processed()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"✔ Split OK — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
