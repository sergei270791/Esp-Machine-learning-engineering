import streamlit as st
import requests

# Configuración de página
st.set_page_config(page_title="Predicción de Upselling", layout="centered")
st.title("📈 Predicción de Aceptación de Upselling")
st.markdown("Simula el comportamiento de un cliente ante una oferta adicional de seguro.")

# Inputs del cliente
age = st.slider("🎂 Edad", 25, 70, 40)
coverage = st.number_input("💵 Monto asegurado actual (USD)", min_value=0.0, step=1000.0, value=15000.0)
years = st.slider("📆 Años como cliente", 1, 20, 5)
claims = st.slider("📄 Reclamos pasados (últimos 5 años)", 0, 10, 1)
income = st.selectbox("💼 Nivel de ingresos", ["Bajo", "Medio", "Alto"])
response = st.radio("📬 Respondió campaña anterior", ["No", "Sí"])

# Threshold slider
threshold = st.slider("🎚 Umbral de aceptación (threshold)", 0.0, 1.0, 0.5, step=0.01)

# Botón de predicción
if st.button("🔍 Evaluar Probabilidad"):
    with st.spinner("Consultando modelo..."):
        try:
            payload = {
                "age": age,
                "current_policy_coverage": coverage,
                "years_with_company": years,
                "past_claims_count": claims,
                "income_level": income,
                "response_last_campaign": 1 if response == "Sí" else 0,
                "threshold": threshold
            }

            r = requests.post("http://20.51.121.119:8000/predict_upsell", json=payload)
            if r.status_code == 200:
                resultado = r.json()
                score = resultado["score_probabilidad"]
                aceptara = resultado["aceptará"]

                st.markdown(f"### 🔢 Score de aceptación: **{score:.3f}**")
                st.markdown(f"### 🎯 Umbral usado: **{threshold:.2f}**")

                if aceptara:
                    st.success("✅ El cliente probablemente **aceptará** el upselling.")
                else:
                    st.warning("⚠️ El cliente probablemente **rechazará** la oferta.")
            else:
                st.error("❌ Error en la respuesta del modelo.")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")


#uvicorn api:app --reload --port 8000
#streamlit run app.py
