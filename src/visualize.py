import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# cargadores y rutas
from src.data_loader import load_processed, split_data
from src.preprocess   import preprocess
from src.config       import MODEL_NAMES, MODEL_DIR, REPORTS_DIR, TARGET_COLUMN

# Directorio de figuras y CSV
FIGURES_DIR = REPORTS_DIR / "figures"
ROC_CSV_DIR = REPORTS_DIR / "roc_csv"

# Carga todos los modelos desde /models/*.joblib
def load_models():
    models = {}
    for name in MODEL_NAMES:
        path = MODEL_DIR / f"{name}.joblib"
        models[name] = joblib.load(path)
    return models

# Calcula FPR, TPR y AUC para cada modelo
def compute_roc(models, X, y):
    roc_data = {}
    for name, model in models.items():
        # extraer score
        if hasattr(model, "decision_function"):
            score = model.decision_function(X)
        elif hasattr(model, "predict_proba"):
            score = model.predict_proba(X)[:,1]
        else:
            # Keras Sequential
            score = model.predict(X).ravel()
        fpr, tpr, _ = roc_curve(y, score)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
    return roc_data

# Guarda para cada modelo un CSV con columnas FPR, TPR y una fila final con el AUC
def save_roc_csv(roc_data):
    ROC_CSV_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in roc_data.items():
        base = pd.DataFrame({"fpr": d["fpr"], "tpr": d["tpr"]})
        footer = pd.DataFrame([
        {"fpr": pd.NA, "tpr": pd.NA},
        {"fpr": "AUC" , "tpr": f"{d['auc']:.6f}"}
    ])
    result = pd.concat([base, footer], ignore_index=True, sort=False)
    result.to_csv(ROC_CSV_DIR / f"roc_{name}.csv", index=False)
    print(f"▶️  Guardado ROC CSV en {ROC_CSV_DIR/f'roc_{name}.csv'}")

# Dibuja y guarda la figura comparativa de las curvas ROC
def plot_roc(roc_data):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for name, d in roc_data.items():
        plt.plot(d["fpr"], d["tpr"],
                 label=f"{name} (AUC = {d['auc']:.3f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_comparison.png")
    plt.close()

# Dibuja y salva matrices de confusión para cada modelo
def plot_confusion(models, X, y):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            y_pred = model.predict(X)
        else:
            y_pred = (model.predict(X).ravel() >= 0.5).astype(int)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(FIGURES_DIR / f"cm_{name}.png")
        plt.close()

# Genera el CSV comparativo de AUCs a partir final_metrics.csv
def compare_metrics_csv():
    df = pd.read_csv(REPORTS_DIR / "final_metrics.csv")
    # Ordenar por la columna interna 'roc_auc'
    df_sorted = df.sort_values("roc_auc", ascending=False)
    df_sorted.to_csv(REPORTS_DIR / "auc_comparison.csv", index=False)
    print("▶️  Guardado resumen de AUC en reports/auc_comparison.csv")

def main():
    # Cargar datos limpios y partirlos
    df = load_processed()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Preprocesar solo test (el transformer ya está en models/preprocessor.joblib)
    df_test = X_test.copy()
    df_test[TARGET_COLUMN] = y_test
    Xp_test, y_test = preprocess(df_test, save_transformer=False)

    # Cargar modelos y generar datos y gráficas ROC
    models   = load_models()
    roc_data = compute_roc(models, Xp_test, y_test)

    # Guardar CSVs de curvas ROC
    save_roc_csv(roc_data)

    # Dibujar curvas ROC comparadas
    plot_roc(roc_data)

    # Dibujar matrices de confusión
    plot_confusion(models, Xp_test, y_test)

    # CSV comparativo de AUC
    compare_metrics_csv()

if __name__ == "__main__":
    main()
