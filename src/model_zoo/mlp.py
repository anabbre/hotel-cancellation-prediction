from sklearn.neural_network import MLPClassifier
from src.config import MLP_PARAMS

# Construye un MLPClassifier a partir de los parámetros por defecto en config.MLP_PARAMS (config.py)
def build_model(**override_params):
    params = MLP_PARAMS.copy()
    params.update(override_params)
    return MLPClassifier(**params) # devuelve un MlPClassifier sin entrentar (se hará en train.py)
