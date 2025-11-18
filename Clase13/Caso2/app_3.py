import streamlit as st
import requests
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(page_title="Predicción de Ventas Semanales", layout="centered")
st.title("🛒 Predicción de Ventas Semanales")
st.markdown("Estima la venta de productos clave según condiciones de la semana.")

# Inputs del usuario
temp = st.slider("🌡 Temperatura promedio (°C)", min_value=5.0, max_value=40.0, value=22.0)
holiday = st.radio("🎉 ¿Semana con feriado?", ["No", "Sí"])
promo = st.slider("📣 Nivel de promoción (0–10)", 0.0, 10.0, 5.0)
traffic = st.number_input("🚶‍♂️ Tráfico estimado en tienda", min_value=100, step=50, value=1500)

# Botón de predicción
if st.button("🔍 Predecir ventas"):
    with st.spinner("Consultando modelo..."):
        payload = {
            "temperature_c": temp,
            "holiday_flag": 1 if holiday == "Sí" else 0,
            "promotion_score": promo,
            "foot_traffic": traffic
        }

        try:
            r = requests.post("http://20.51.121.119:8000/predict_sales", json=payload)
            if r.status_code == 200:
                pred = r.json()["predicciones"]
                st.success("✅ Predicción generada exitosamente")

                # Gráfico de barras
                productos = ["🧻 Papel", "🍞 Pan", "🥛 Leche"]
                cantidades = [pred["sales_paper"], pred["sales_bread"], pred["sales_milk"]]

                st.subheader("📊 Ventas estimadas (unidades)")
                fig, ax = plt.subplots()
                ax.bar(productos, cantidades, color=["skyblue", "orange", "lightgreen"])
                ax.set_ylabel("Unidades")
                st.pyplot(fig)

            else:
                st.error("❌ Error en la respuesta del servidor.")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")

#uvicorn api:app --reload --port 8000
#streamlit run app.py
