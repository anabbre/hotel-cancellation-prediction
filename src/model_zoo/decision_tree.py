from sklearn.tree import DecisionTreeClassifier

# Construye y devuelve un DecisionTreeClassifier
def build_model(**kwargs):
    return DecisionTreeClassifier(
        criterion=kwargs.get("criterion", "gini"),
        max_depth=kwargs.get("max_depth", None),
        random_state=kwargs.get("random_state", 42)
    )
