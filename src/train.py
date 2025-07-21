import sys
from src.data_loader import load_processed, split_data

def main():
    # Carga el dataset ya limpio
    try:
        df = load_processed()
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("Ejecuta primero: python -m src.clean_data")
        sys.exit(1)

    # Mostramos dimensiones del DataFrame procesado
    print("✅ Dataset procesado cargado:")
    print(f" → Filas: {len(df):,}")
    print(f" → Columnas: {df.shape[1]}\n")

    # Generamos los splits (train/val/test)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Mostramos las shapes resultantes
    print("🔀 Splits generados:")
    print(f" X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f" X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f" X_test:  {X_test.shape},  y_test:  {y_test.shape}\n")

    print("🎉 Fase 2 completada: preprocesado y splits OK.")

if __name__ == "__main__":
    main()
