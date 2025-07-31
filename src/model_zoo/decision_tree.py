from sklearn.tree import DecisionTreeClassifier

# Construye y devuelve un DecisionTreeClassifier
def build_model(**kwargs):
    """
    Devuelve un clasificador Decision Tree.

    Se pueden pasar parámetros como 'criterion' o 'max_depth' por kwargs.
    """
    return DecisionTreeClassifier(
        criterion=kwargs.get("criterion", "gini"),
        max_depth=kwargs.get("max_depth", None),
        random_state=kwargs.get("random_state", 42)
    )
