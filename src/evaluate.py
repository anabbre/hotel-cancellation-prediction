import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Imprime métricas y devuelve un dict con accuracy, precision, recall, f1, roc_auc
def evaluate_model(model, X, y, prefix: str = "") -> dict:

    # Predicciones de clase
    y_pred = model.predict(X)

    # Probabilidades o scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)
    
    # Cálculo de métricas
    acc   = accuracy_score(y, y_pred)
    prec  = precision_score(y, y_pred)
    rec   = recall_score(y, y_pred)
    f1    = f1_score(y, y_pred)
    auc   = roc_auc_score(y, y_score)

    # Imprime
    title = f"{prefix} " if prefix else ""
    print(f"\n Métricas {title.strip()}:")
    print(f" • Accuracy : {acc:.4f}")
    print(f" • Precision: {prec:.4f}")
    print(f" • Recall   : {rec:.4f}")
    print(f" • F1-score : {f1:.4f}")
    print(f" • AUC ROC  : {auc:.4f}\n")
    print(classification_report(y, y_pred, digits=4))

    # Devuevle las métricas para poder agruparlas
    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1_score":  f1,
        "roc_auc":   auc,
    }

