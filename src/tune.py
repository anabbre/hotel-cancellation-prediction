import joblib
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from src.data_loader import load_processed, split_data
from src.preprocess  import preprocess
from src.config      import (
    HYPERPARAM_GRIDS,
    MODEL_DIR,
    BEST_MODEL_PATH
)
from src.model_zoo.decision_tree       import build_model as dt_builder
from src.model_zoo.logistic_regression import build_model as lr_builder
from src.model_zoo.gradient_boost      import build_model as gb_builder
from src.model_zoo.random_forest       import build_model as rf_builder

def tune_models(Xp_train, y_train, n_iter=20, cv=3):
    builders = {
        "decision_tree":       dt_builder,
        "logistic_regression": lr_builder,
        "gradient_boost":      gb_builder,
        "random_forest":       rf_builder,
    }
    results = {}
    for name, builder in builders.items():
        print(f"\n🔎 Tuning {name}…")
        rs = RandomizedSearchCV(
            builder(),
            param_distributions=HYPERPARAM_GRIDS[name],
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42
        )
        rs.fit(Xp_train, y_train)
        results[name] = {
            "estimator": rs.best_estimator_,
            "score":     rs.best_score_,
            "params":    rs.best_params_
        }
        print(f"→ {name} best AUC-ROC: {rs.best_score_:.4f}")
    return results

def main():
    # Carga y particionado
    df = load_processed()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Preprocesado solo sobre train
    Xp_train, _ = preprocess(X_train, save_transformer=True)

    # Tuning de cada modelo
    bests = tune_models(Xp_train, y_train, n_iter=20, cv=3)

    # Guardar cada modelo optimizado
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, info in bests.items():
        joblib.dump(info["estimator"], MODEL_DIR / f"{name}_best.joblib")

    # Seleccionar y copiar el “best of the best”
    # Buscamos el modelo con mayor AUC-ROC
    winner = max(bests.items(), key=lambda kv: kv[1]["score"])[0]
    src_path = MODEL_DIR / f"{winner}_best.joblib"
    shutil.copy(src_path, BEST_MODEL_PATH)
    print(f"\n 🏆Best model is '{winner}' (AUC-ROC={bests[winner]['score']:.4f})")
    print(f"→ Copiado a {BEST_MODEL_PATH}")

    # Generar resumen en CSV
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    df_summary = pd.DataFrame([
        {"model": name, "auc_roc": info["score"]}
        for name, info in bests.items()
    ])
    csv_path = reports_dir / "auc_comparison.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"→ Resumen de AUC-ROC guardado en '{csv_path}'")
    print("\n✅ Tuning completado.")

if __name__ == "__main__":
    main()
