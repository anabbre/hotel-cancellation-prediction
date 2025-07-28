import streamlit as st
import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, TARGET_COLUMN
from src.preprocess import preprocess

# Cargar modelo y preprocesador
model = joblib.load(MODEL_DIR / "random_forest.joblib")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")

st.set_page_config(page_title="Predicción de Cancelación", layout="centered")
st.title("🛎️ Predicción de Cancelación de Reserva de Hotel")
st.write("Introduce los datos de la reserva para predecir si será cancelada.")

# Variables más importantes del modelo Random Forest
lead_time = st.number_input("Lead time (días de antelación)", min_value=0, value=30)
adr = st.number_input("ADR (precio medio por noche)", min_value=0.0, value=100.0)
country = st.selectbox("País", ["PRT", "ESP", "FRA", "GBR", "DEU"])
deposit_type = st.selectbox("Tipo de depósito", ["No Deposit", "Refundable", "Non Refund"])
total_special_requests = st.slider("Solicitudes especiales", 0, 5, 1)


# Variables adicionales relevantes
room_changed = st.radio("¿Se cambió el tipo de habitación?", ["Sí", "No"])
room_changed = 1 if room_changed == "Sí" else 0

previous_cancellations = st.number_input("Cancelaciones previas del cliente", min_value=0, value=0)

booking_changes = st.number_input("Nº de cambios en la reserva", min_value=0, value=0)

customer_type = st.selectbox("Tipo de cliente", ["Transient", "Contract", "Transient-Party", "Group"])

market_segment = st.selectbox("Segmento de mercado", ["Online TA", "Offline TA/TO", "Direct", "Corporate"])

arrival_month_num = st.slider("Mes de llegada (número)", min_value=1, max_value=12, value=6)

# Botón de predicción
if st.button("Predecir"):
    # Crear DataFrame con los datos introducidos
    input_df = pd.DataFrame([{
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
    }])

    # Asegurar que todas las columnas esperadas existen
    for col in preprocessor.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    # Preprocesar
    X_input, _ = preprocess(input_df, save_transformer=False)

    # Predecir
    prediction = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    # Mostrar resultado
    if prediction == 1:
        st.error(f"⚠️ Esta reserva probablemente será CANCELADA (probabilidad: {prob:.2f})")
    else:
        st.success(f"✅ Esta reserva probablemente NO será cancelada (probabilidad: {prob:.2f})")