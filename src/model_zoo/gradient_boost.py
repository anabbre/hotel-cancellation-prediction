from xgboost import XGBClassifier

# Crea un XGBClassifier parametrizable
def build_model(**kwargs):
    """
    Devuelve un clasificador XGBoost.

    Permite ajustar parámetros como n_estimators, learning_rate o max_depth
    mediante kwargs. Usa 'logloss' como métrica por defecto.
    """
    return XGBClassifier(
        n_estimators=kwargs.get("n_estimators", 100),
        learning_rate=kwargs.get("learning_rate", 0.1),
        max_depth=kwargs.get("max_depth", 6),
        random_state=kwargs.get("random_state", 42),
        use_label_encoder=False,
        eval_metric="logloss",
    )
