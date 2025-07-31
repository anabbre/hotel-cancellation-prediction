from sklearn.ensemble import RandomForestClassifier

# Crea un RandomForestClassifier parametrizable
def build_model(**kwargs):
    """
    Devuelve un clasificador Random Forest.

    Acepta parámetros como n_estimators o max_depth por kwargs.
    Entrena en paralelo con n_jobs=-1.
    """
    return RandomForestClassifier(
        n_estimators=kwargs.get("n_estimators", 100),
        max_depth=kwargs.get("max_depth", None),
        random_state=kwargs.get("random_state", 42),
        n_jobs=kwargs.get("n_jobs", -1),
    )
