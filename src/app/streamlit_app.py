import sys
import os
# Añade la raíz del proyecto al path para permitir importar src.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from src.config import MODEL_DIR, TARGET_COLUMN
from src.preprocess import preprocess


def load_model(model_name="random_forest"):
    """Carga el modelo y el preprocesador desde el directorio de modelos."""
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    return model, preprocessor


def build_form():
    """Construye el formulario de entrada de datos en la interfaz de Streamlit y devuelve un DataFrame."""
    st.write("Introduce los datos de la reserva para predecir si será cancelada.")

    lead_time = st.number_input("Lead time (días de antelación)", min_value=0, value=30)
    adr = st.number_input("ADR (precio medio por noche)", min_value=0.0, value=100.0)
    country = st.selectbox("País", ["PRT", "ESP", "FRA", "GBR", "DEU"])
    deposit_type = st.selectbox("Tipo de depósito", ["No Deposit", "Refundable", "Non Refund"])
    total_special_requests = st.slider("Solicitudes especiales", 0, 5, 1)
    room_changed = st.radio("¿Se cambió el tipo de habitación?", ["Sí", "No"])
    room_changed = 1 if room_changed == "Sí" else 0
    previous_cancellations = st.number_input("Cancelaciones previas del cliente", min_value=0, value=0)
    booking_changes = st.number_input("Nº de cambios en la reserva", min_value=0, value=0)
    customer_type = st.selectbox("Tipo de cliente", ["Transient", "Contract", "Transient-Party", "Group"])
    market_segment = st.selectbox("Segmento de mercado", ["Online TA", "Offline TA/TO", "Direct", "Corporate"])
    arrival_month_num = st.slider("Mes de llegada (número)", min_value=1, max_value=12, value=6)

    input_data = {
        "lead_time": lead_time,
        "adr": adr,
        "country": country,
        "deposit_type": deposit_type,
        "total_of_special_requests": total_special_requests,
        "room_changed": room_changed,
        "previous_cancellations": previous_cancellations,
        "booking_changes": booking_changes,
        "customer_type": customer_type,
        "market_segment": market_segment,
        "arrival_month_num": arrival_month_num
    }

    return pd.DataFrame([input_data])


def main():
    """App principal de Streamlit para predecir cancelaciones de hotel."""
    st.set_page_config(page_title="Predicción de Cancelación", layout="centered")
    st.title("🛎️ Predicción de Cancelación de Reserva de Hotel")

    model, preprocessor = load_model()

    input_df = build_form()

    if st.button("Predecir"):
        # Añadir columnas faltantes si las hubiera
        for col in preprocessor.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0

        X_input, _ = preprocess(input_df, save_transformer=False)
        prediction = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Esta reserva probablemente será CANCELADA (probabilidad: {prob:.2f})")
        else:
            st.success(f"✅ Esta reserva probablemente NO será cancelada (probabilidad: {prob:.2f})")


if __name__ == "__main__":
    main()
