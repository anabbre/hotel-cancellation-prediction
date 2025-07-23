import joblib
from src.data_loader import load_processed, split_data
from src.preprocess  import preprocess
from src.evaluate    import evaluate_model
from src.config      import BEST_MODEL_PATH, MODEL_DIR

def main():
    # Carga y split
    df = load_processed()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Ajusta y guarda el transformer sobre train (si no existe lo vuelve a crear)
    _, _ = preprocess(X_train, save_transformer=True)

    # Carga el preprocesador guardado y transforma el test set
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    Xp_test = preprocessor.transform(X_test)

    # Carga el best model y evalúa sobre TEST
    best = joblib.load(BEST_MODEL_PATH)
    print("📊 Métricas del Best Model sobre el TEST set:")
    evaluate_model(best, Xp_test, y_test, prefix="Best Model Test")

if __name__ == "__main__":
    main()
