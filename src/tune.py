import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.data_loader               import load_processed, split_data
from src.preprocess                import preprocess
from src.config                    import (
    TUNE_PARAMS, METRIC, CV, N_JOBS,
    MODEL_DIR, REPORTS_DIR, TARGET_COLUMN
)

from src.model_zoo.decision_tree       import build_model as dt_builder
from src.model_zoo.logistic_regression import build_model as lr_builder
from src.model_zoo.gradient_boost      import build_model as gb_builder
from src.model_zoo.random_forest       import build_model as rf_builder


def main():
    """
    Realiza la búsqueda de hiperparámetros con GridSearchCV para 4 modelos.

    Carga los datos preprocesados, ajusta el transformador en entrenamiento,
    transforma el conjunto de validación y luego aplica GridSearchCV a:
    árbol de decisión, regresión logística, gradient boost y random forest.

    Guarda el mejor modelo de cada uno en 'models/' y los resultados de la búsqueda
    en 'reports/'. También genera un CSV resumen con los mejores scores y parámetros.

    No incluye el modelo MLP, ya que no se tunearon sus hiperparámetros.
    """
    # Carga y particionado
    df = load_processed()
    X_train, X_val, _, y_train, y_val, _ = split_data(df)

    # Preprocesado
    # Fit transformer en train (se guarda internamente)
    df_train = X_train.copy()
    df_train[TARGET_COLUMN] = y_train
    Xp_train, y_train = preprocess(df_train, save_transformer=True)

    # Transform en val
    df_val = X_val.copy()
    df_val[TARGET_COLUMN] = y_val
    Xp_val, y_val = preprocess(df_val, save_transformer=False)

    # GridSearch para cada modelo
    builders = {
        "decision_tree":       dt_builder(),
        "logistic_regression": lr_builder(),
        "gradient_boost":      gb_builder(),
        "random_forest":       rf_builder(),
    }

    summary = []

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in builders.items():
        print(f"\n🔎 Tuning {name} con GridSearchCV…")
        grid = GridSearchCV(
            estimator  = model,
            param_grid = TUNE_PARAMS[name],
            scoring    = METRIC,
            cv         = CV,
            n_jobs     = N_JOBS,
            refit      = True,
            verbose    = 1
        )
        grid.fit(Xp_train, y_train)

        best_est = grid.best_estimator_
        # Guardar modelo óptimo
        joblib.dump(best_est, MODEL_DIR / f"{name}_best.joblib")

        # Guardar resultados de CV
        pd.DataFrame(grid.cv_results_)\
          .to_csv(REPORTS_DIR / f"{name}_cv_results.csv", index=False)

        # Añadir resumen
        summary.append({
            "model":     name,
            METRIC:      grid.best_score_,
            **{f"best_{k}": v for k, v in grid.best_params_.items()}
        })

    # Exportar CSV resumen de tuning
    pd.DataFrame(summary)\
      .to_csv(REPORTS_DIR / "tuning_summary.csv", index=False)

    print("\n✅ Búsqueda de hiperparámetros completada.")
    print(f" Resumen en {REPORTS_DIR/'tuning_summary.csv'}")

if __name__ == "__main__":
    main()
