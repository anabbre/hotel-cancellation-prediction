import pandas as pd
from pathlib import Path
from src.config import DATA_RAW, DATA_PROCESSED, TARGET_COLUMN
from sklearn.model_selection import train_test_split

# Carga el CSV original
def load_raw() -> pd.DataFrame:
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"No encuentro raw data en {DATA_RAW}")
    return pd.read_csv(DATA_RAW)

#Guarda el DataFrame procesado en DATA_PROCESSED
def save_processed(df: pd.DataFrame) -> None:
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False)
    print(f"✔ Dataset limpio guardado en {DATA_PROCESSED}")

#Carga el dataset ya procesado. Si no existe, invoca a clean_dataset para generarlo desde el raw
def load_processed() -> pd.DataFrame:
    if not DATA_PROCESSED.exists():
        print(f"⚠️ Dataset procesado no encontrado en {DATA_PROCESSED}, generando desde raw…")
        from src.clean_data import clean_dataset
        df = clean_dataset()
    else:
        df = pd.read_csv(DATA_PROCESSED)
    return df

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
):
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
