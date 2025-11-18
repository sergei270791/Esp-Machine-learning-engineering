import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de la app
st.set_page_config(page_title="Clasificación de Riesgo de Crédito", layout="centered")
st.title("🔎 Clasificación de Riesgo de Crédito")
st.markdown("Ingresa los datos del cliente para estimar su nivel de **riesgo crediticio**.")

# Inputs del formulario
age = st.slider("🎂 Edad", 18, 70, 35)
income = st.number_input("💰 Ingreso mensual (USD)", min_value=0.0, step=100.0, value=2000.0)
loan = st.number_input("🏦 Monto solicitado del préstamo (USD)", min_value=0.0, step=500.0, value=5000.0)
term = st.selectbox("📆 Plazo del préstamo (meses)", [12, 24, 36, 48, 60])
loans_past = st.slider("📄 Créditos previos (últimos 5 años)", 0, 10, 2)
arrears = st.slider("❌ Cuotas vencidas actualmente", 0, 10, 0)
region = st.selectbox("📍 Región", ["Lima", "Arequipa", "Cusco", "Piura", "Trujillo"])

# Botón de predicción
if st.button("🔍 Evaluar Riesgo"):
    with st.spinner("Enviando solicitud al modelo..."):
        payload = {
            "age": age,
            "income": income,
            "loan_amount": loan,
            "term_months": term,
            "num_loans_last_5y": loans_past,
            "current_arrears": arrears,
            "region": region
        }

        try:
            # ✅ URL ajustada para tu MV
            r = requests.post("http://20.51.121.119:8000/predict_risk", json=payload)
            if r.status_code == 200:
                resultado = r.json()
                riesgo = resultado["riesgo estimado"]
                probs = resultado["probabilidades"]

                st.subheader("🔐 Riesgo estimado:")
                if riesgo == "alto":
                    st.error("⚠️ Riesgo **ALTO**")
                elif riesgo == "medio":
                    st.warning("🟡 Riesgo **MEDIO**")
                else:
                    st.success("✅ Riesgo **BAJO**")

                # Mostrar gráfica de barras
                if probs:
                    st.subheader("📊 Probabilidades por categoría de riesgo:")
                    labels = [k.replace("Score_", "") for k in probs.keys()]
                    values = list(probs.values())

                    fig, ax = plt.subplots()
                    ax.barh(labels, values, color=["red", "orange", "green"])
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Probabilidad")
                    st.pyplot(fig)
                else:
                    st.error("❌ No se encontraron probabilidades para graficar.")
            else:
                st.error("❌ Error en la respuesta del servidor.")
        except Exception as e:
            st.error(f"❌ Error de conexión: {e}")
