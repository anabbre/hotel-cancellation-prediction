from sklearn.ensemble import RandomForestClassifier

# Crea un RandomForestClassifier parametrizable
def build_model(**kwargs):
    return RandomForestClassifier(
        n_estimators=kwargs.get("n_estimators", 100),
        max_depth=kwargs.get("max_depth", None),
        random_state=kwargs.get("random_state", 42),
        n_jobs=kwargs.get("n_jobs", -1),
    )
