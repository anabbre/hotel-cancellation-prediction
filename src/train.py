import sys
import joblib

from src.data_loader import load_processed, split_data
from src.preprocess   import preprocess
from src.evaluate     import evaluate_model
from src.config       import MODEL_DIR

from src.model_zoo.decision_tree       import build_model as dt_builder
from src.model_zoo.logistic_regression import build_model as lr_builder
from src.model_zoo.gradient_boost      import build_model as gb_builder
from src.model_zoo.random_forest       import build_model as rf_builder

def main():
    # Carga los datos
    try:
        df = load_processed()
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   Ejecuta primero: python -m src.clean_data")
        sys.exit(1)

    # Split estratificado
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"\n✅ Splits OK — train: {len(X_train)} / val: {len(X_val)} / test: {len(X_test)}\n")

    # Preprocesado
    Xp_train, _ = preprocess(X_train, save_transformer=True)
    Xp_val,   _ = preprocess(X_val,   save_transformer=False)
    Xp_test,  _ = preprocess(X_test,  save_transformer=False)
    print(f"✅ Preprocesado OK — dims: train {Xp_train.shape}, val {Xp_val.shape}, test {Xp_test.shape}\n")

    # Modelos a entrenar
    models = {
        "decision_tree":       dt_builder(),
        "logistic_regression": lr_builder(),
        "gradient_boost":      gb_builder(),
        "random_forest":       rf_builder(),
    }

    # Entrenamiento, serializado y evaluación val
    for name, model in models.items():
        print(f"🚀 Entrenando {name}…")
        model.fit(Xp_train, y_train)

        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        print(f" Modelo guardado en {path}")

        print(f"\n Métricas {name} (val):")
        evaluate_model(model, Xp_val, y_val, prefix=name)
        print("-" * 50 + "\n")

    print("✅ ¡Todos los modelos han sido entrenados y evaluados!")

if __name__ == "__main__":
    main()
