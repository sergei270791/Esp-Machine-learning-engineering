from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd
import os

app = FastAPI(title="API de Riesgo de Crédito Multiclase")

MODEL_PATH = "/home/SERGEICALLE/airflow/models/credit_risk_model"

if not os.path.exists(MODEL_PATH + ".pkl"):
    raise RuntimeError(f"❌ Modelo no encontrado en {MODEL_PATH}.pkl")

model = load_model(MODEL_PATH)

class Cliente(BaseModel):
    age: int
    income: float
    loan_amount: float
    term_months: int
    num_loans_last_5y: int
    current_arrears: int
    region: str


@app.post("/predict_risk")
def predict(cliente: Cliente):
    try:
        data = pd.DataFrame([cliente.dict()])
        data["region"] = data["region"].astype("category")

        resultado = predict_model(model, data=data)

        col_scores = [c for c in resultado.columns if c.startswith("Score_")]
        probabilidades = resultado[col_scores].iloc[0].to_dict() if col_scores else {}

        if "prediction_label" in resultado.columns:
            prediccion = resultado["prediction_label"][0]
        elif "Label" in resultado.columns:
            prediccion = resultado["Label"][0]
        else:
            prediccion = "desconocido"

        return {
            "input": cliente.dict(),
            "riesgo estimado": prediccion,
            "probabilidades": probabilidades
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
