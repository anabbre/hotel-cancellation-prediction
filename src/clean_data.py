import pandas as pd
from pathlib import Path
from src.config import DATA_RAW, DATA_PROCESSED, TARGET_COLUMN
from src.config import NUM_FEATURES, CAT_FEATURES

# Cargamos el CSV original para su limpieza
def clean_dataset() -> pd.DataFrame:

    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Dataset crudo no encontrado en {DATA_RAW}")
    df = pd.read_csv(DATA_RAW)

    df = df.copy()  # asegura que trabajamos sobre una copia “propia”

    # Quitamos reservas en las que nadie va a alojarse
    df = df[~((df["adults"] == 0) & (df["children"] == 0) & (df["babies"] == 0))]

    # Cambiamos las celdas vacías por 0 y si no conocemos el país lo ponemos a Unknown
    df["children"] = df["children"].fillna(0)
    df["country"]  = df["country"].fillna("Unknown")
    df["agent"]    = df["agent"].fillna(0)

    # Transformamos los valores desconocidos a SC para no tener dos valores para un mismo dato
    df["meal"]     = df["meal"].replace("Undefined", "SC")

    # Añade room_changed = 1 si el tipo de habitación asignado es distinto al reservado, si no 0
    df["room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)

    # Convierte fechas
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_date_month"] + "-" +
        df["arrival_date_day_of_month"].astype(str),
        errors="coerce"
    )
    df["arrival_month_num"] = df["arrival_date"].dt.month
    df["arrival_quarter"] = df["arrival_date"].dt.quarter

    # Season (simplificada hemisferio norte)
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["season"] = df["arrival_month_num"].map(season_map)

    # Duración total de la estancia
    df["length_of_stay"] = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]

    # Clip de outliers en days_in_waiting_list al p99
    p99_wait = df["days_in_waiting_list"].quantile(0.99)
    df["days_in_waiting_list"] = df["days_in_waiting_list"].clip(upper=p99_wait)

    # Recorta valores extremos (el resto)
    df["adr"] = df["adr"].clip(upper=df["adr"].quantile(0.99))
    df["lead_time"] = df["lead_time"].clip(upper=df["lead_time"].quantile(0.99))

    # Borra columnas inútiles
    drop_cols = [
        "company", "reservation_status", "reservation_status_date",
        "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"
    ]
    df.drop(columns=drop_cols, inplace=True)

    # Guarda el fichero limpio
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False)
    print(f"✔ Dataset limpio guardado en {DATA_PROCESSED}")
    return df

if __name__ == "__main__":
    clean_dataset()
