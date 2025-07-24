import joblib

from src.data_loader   import load_processed, split_data
from src.preprocess    import preprocess
from src.evaluate      import evaluate_model
from src.config        import MODEL_DIR, TARGET_COLUMN

from src.model_zoo.decision_tree       import build_model as dt_builder
from src.model_zoo.logistic_regression import build_model as lr_builder
from src.model_zoo.gradient_boost      import build_model as gb_builder
from src.model_zoo.random_forest       import build_model as rf_builder

def main():
    # Carga datos (crea processed si falta)
    df = load_processed()

    # División estratificada
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"\n✅ Splits OK — train: {len(X_train)} / val: {len(X_val)} / test: {len(X_test)}\n")

    # Preprocesado
    # fit en train (guarda transformer)
    df_train = X_train.copy()
    df_train[TARGET_COLUMN] = y_train
    Xp_train, y_train = preprocess(df_train, save_transformer=True)

    # transform en val y test
    df_val = X_val.copy()
    df_val[TARGET_COLUMN] = y_val
    Xp_val, y_val = preprocess(df_val, save_transformer=False)

    df_test = X_test.copy()
    df_test[TARGET_COLUMN] = y_test
    Xp_test, y_test = preprocess(df_test, save_transformer=False)

    # Entrenamiento y evaluación de modelos
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    builders = {
        "decision_tree":       dt_builder(),
        "logistic_regression": lr_builder(),
        "gradient_boost":      gb_builder(),
        "random_forest":       rf_builder(),
    }

    for name, model in builders.items():
        print(f"🚀 Entrenando {name}…")
        model.fit(Xp_train, y_train)

        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        print(f" ✔ Modelo guardado en {path}")

        print(f"\n📊 Métricas {name} (val):")
        evaluate_model(model, Xp_val, y_val, prefix=name)
        print("-" * 50 + "\n")

    print("✅ ¡Todos los modelos han sido entrenados y evaluados!")

if __name__ == "__main__":
    main()
