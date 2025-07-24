import joblib
import pandas as pd
from src.data_loader  import load_processed, split_data
from src.preprocess   import preprocess
from src.evaluate     import evaluate_model
from src.config       import MODEL_DIR, REPORTS_DIR, TARGET_COLUMN

def main():
    # Carga y split
    df = load_processed()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Preprocesado test (transformer ya está guardado)
    df_test = X_test.copy()
    df_test[TARGET_COLUMN] = y_test
    Xp_test, y_test = preprocess(df_test, save_transformer=False)

    # Cargar y evaluar cada *_best.joblib producido en tune.py
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    final_metrics = []

    for path in MODEL_DIR.glob("*_best.joblib"):
        name  = path.stem.replace("_best", "")
        model = joblib.load(path)
        print(f"\n📊 Evaluando modelo {name} en test set…")
        # evaluate_model imprime y devuelve diccionario de métricas
        m = evaluate_model(model, Xp_test, y_test, prefix=name)
        final_metrics.append({"model": name, **m})

    # Guardar CSV final
    pd.DataFrame(final_metrics)\
      .to_csv(REPORTS_DIR / "final_metrics.csv", index=False)

    print(f"\n✅ Evaluación final completa. CSV en {REPORTS_DIR/'final_metrics.csv'}")

if __name__ == "__main__":
    main()
