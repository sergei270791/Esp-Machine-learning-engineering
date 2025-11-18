from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

# 🛰 Inicializar API
app = FastAPI(title="API de Aprobación de Crédito con Riesgo Político")

# 📁 Cargar modelo
MODEL_PATH = "/home/SERGEICALLE/airflow/models/credit_approval_model"
model = load_model(MODEL_PATH)

# 🎯 Esquema de entrada
class SolicitudCredito(BaseModel):
    age: int
    monthly_income_usd: float
    app_usage_score: float
    digital_profile_strength: float
    num_contacts_uploaded: int
    residence_risk_zone: str  # "baja", "media", "alta"
    political_event_last_month: int  # 0/1
    threshold: float  # Umbral de aprobación (0.0 - 1.0)

@app.post("/predict_approval")
def predict(solicitud: SolicitudCredito):
    # Convertir a DataFrame
    data = pd.DataFrame([solicitud.dict()])

    # Extraer threshold y quitarlo de las features
    threshold = data.pop("threshold").iloc[0]

    # Asegurar tipos categóricos
    data["residence_risk_zone"] = data["residence_risk_zone"].astype("category")

    # Predicción con PyCaret
    resultado = predict_model(model, data=data)

    # Probabilidad de aprobación (clase positiva, en general '1')
    score = float(resultado["prediction_score"][0])

    # Decisión binaria según threshold
    aprobado = int(score >= threshold)

    return {
        "input": solicitud.dict(),
        "score_aprobacion": round(score, 4),
        "threshold_usado": threshold,
        "aprobado": bool(aprobado)
    }
