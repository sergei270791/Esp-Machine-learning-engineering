import streamlit as st
import requests

# ⚙️ Configuración de página
st.set_page_config(page_title="Aprobación de Crédito Digital", layout="centered")
st.title("💳 Predicción de Aprobación de Crédito")
st.markdown(
    "Simula la **aprobación de crédito** considerando perfil financiero, "
    "riesgo político y comportamiento digital."
)

st.subheader("🧍 Datos del solicitante")

# Inputs numéricos
age = st.slider("🎂 Edad", 18, 75, 30)
income = st.number_input("💰 Ingreso mensual estimado (USD)", min_value=0.0, step=100.0, value=800.0)
app_usage = st.slider("📱 Score de uso de la app (0–10)", 0.0, 10.0, 5.0)
digital_profile = st.slider("🌐 Fortaleza del perfil digital (0–100)", 0.0, 100.0, 50.0)
contacts = st.number_input("📇 Contactos sincronizados desde el móvil", min_value=0, step=5, value=50)

# Riesgo político / zona
residence_risk_zone = st.selectbox(
    "📍 Zona de residencia (riesgo político)",
    ["baja", "media", "alta"]
)

political_event = st.radio(
    "⚠️ ¿Hubo disturbios / eventos políticos el último mes en su zona?",
    ["No", "Sí"]
)

# Threshold
st.subheader("🎚 Política de riesgo")
threshold = st.slider(
    "Umbral de aprobación (threshold)",
    0.0, 1.0, 0.5, step=0.01,
    help="Valores altos → política más estricta, valores bajos → política más flexible."
)

# Botón de predicción
if st.button("🔍 Evaluar solicitud"):
    with st.spinner("Consultando modelo de aprobación de crédito..."):
        payload = {
            "age": age,
            "monthly_income_usd": income,
            "app_usage_score": app_usage,
            "digital_profile_strength": digital_profile,
            "num_contacts_uploaded": contacts,
            "residence_risk_zone": residence_risk_zone,
            "political_event_last_month": 1 if political_event == "Sí" else 0,
            "threshold": threshold
        }

        try:
            # 🔗 URL del API en tu MV
            r = requests.post("http://20.51.121.119:8000/predict_approval", json=payload)

            if r.status_code == 200:
                resultado = r.json()
                score = resultado["score_aprobacion"]
                aprobado = resultado["aprobado"]

                st.markdown(f"### 🔢 Score de aprobación: **{score:.3f}**")
                st.markdown(f"### 🎯 Umbral usado: **{threshold:.2f}**")

                # Mapeo simple de score → nivel de riesgo
                if score >= 0.8:
                    riesgo = "bajo"
                    st.success("✅ Alta probabilidad de aprobación. **Riesgo BAJO**.")
                elif score >= 0.5:
                    riesgo = "medio"
                    st.warning("🟡 Probabilidad moderada de aprobación. **Riesgo MEDIO**.")
                else:
                    riesgo = "alto"
                    st.error("⚠️ Baja probabilidad de aprobación. **Riesgo ALTO**.")

                st.markdown(f"**Clasificación de riesgo del solicitante:** `{riesgo.upper()}`")

                # Resultado final según threshold
                st.markdown("---")
                if aprobado:
                    st.success("✅ Según la política de riesgo (threshold), el crédito **sería APROBADO**.")
                else:
                    st.warning("❌ Según la política de riesgo (threshold), el crédito **sería RECHAZADO**.")

            else:
                st.error(f"❌ Error en la respuesta del servidor: {r.status_code}")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")
