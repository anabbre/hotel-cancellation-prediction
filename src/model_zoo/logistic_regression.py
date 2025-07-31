from sklearn.linear_model import LogisticRegression

def build_model(**kwargs):
    """
    Devuelve un clasificador de regresión logística.

    Permite ajustar el tipo de penalización, el valor de C, y otros parámetros.
    """
    return LogisticRegression(
        penalty=kwargs.get("penalty", "l2"),
        C=kwargs.get("C", 1.0),
        random_state=kwargs.get("random_state", 42),
        max_iter=kwargs.get("max_iter", 1000),
        n_jobs=kwargs.get("n_jobs", -1),
    )
