import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import MLP_PARAMS

def build_model(input_shape):
    # Fijar semilla para reproducibilidad
    tf.random.set_seed(MLP_PARAMS["random_seed"])

    model = models.Sequential([
        layers.InputLayer(input_shape=(input_shape,)),
        # Capas ocultas según la configuración
        *[
            layers.Dense(units, activation=MLP_PARAMS["activation"])
            for units in MLP_PARAMS["hidden_layers"]
        ],
        # Capa de salida binaria
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=MLP_PARAMS["optimizer"],
        loss="binary_crossentropy",
        metrics=[MLP_PARAMS["metric"]]
    )

    return model
