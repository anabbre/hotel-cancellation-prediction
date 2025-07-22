from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Ajusta el modelo ya entrenado sobre X, y y 
def evaluate_model(model, X, y, prefix: str = ""):
    # predicciones
    y_pred = model.predict(X)

    # probabilidades (para AUC ROC) 
    try:
        y_proba = model.predict_proba(X)[:, 1]
    except Exception:
        y_proba = None

    # métricas básicas (recall, f1, precision, accuracy)
    print(f"\n Métricas {prefix}:")
    print(f"  • Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"  • Precision: {precision_score(y, y_pred):.4f}")
    print(f"  • Recall   : {recall_score(y, y_pred):.4f}")
    print(f"  • F1-score : {f1_score(y, y_pred):.4f}")

    if y_proba is not None:
        print(f"  • AUC ROC  : {roc_auc_score(y, y_proba):.4f}")

    # reporte más detallado
    print("\n" + classification_report(y, y_pred, digits=4))
