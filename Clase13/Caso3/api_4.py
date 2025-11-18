from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

# Inicializar FastAPI
app = FastAPI(title="API de Clasificación de Upselling")

# Cargar modelo
model = load_model("/home/SERGEICALLE/airflow/models/upsell_model")

# Entrada esperada
class Cliente(BaseModel):
    age: int
    current_policy_coverage: float
    years_with_company: int
    past_claims_count: int
    income_level: str
    response_last_campaign: int
    threshold: float  # nuevo: umbral personalizado (0.0 - 1.0)

@app.post("/predict_upsell")
def predict(cliente: Cliente):
    input_data = pd.DataFrame([cliente.dict()])
    threshold = input_data.pop("threshold").iloc[0]
    input_data["income_level"] = input_data["income_level"].astype("category")
    
    resultado = predict_model(model, data=input_data)
    score = float(resultado["prediction_score"][0])
    decision = int(score >= threshold)

    return {
        "input": cliente.dict(),
        "score_probabilidad": round(score, 4),
        "threshold_usado": threshold,
        "aceptará": bool(decision)
    }


#pip install fastapi uvicorn
#uvicorn api:app --reload --port 8000
