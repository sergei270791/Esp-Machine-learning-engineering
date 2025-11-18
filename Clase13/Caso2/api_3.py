from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.regression import load_model, predict_model
import pandas as pd

# Inicializar API
app = FastAPI(title="API de Forecast de Ventas Semanales")

# Cargar modelos
model_paper = load_model("/home/SERGEICALLE/airflow/models/model_paper")
model_bread = load_model("/home/SERGEICALLE/airflow/models/model_bread")
model_milk = load_model("/home/SERGEICALLE/airflow/models/model_milk")

# Clase de entrada
class Semana(BaseModel):
    temperature_c: float
    holiday_flag: int
    promotion_score: float
    foot_traffic: int

# Endpoint para predecir ventas
@app.post("/predict_sales")
def predict(semana: Semana):
    data = pd.DataFrame([semana.dict()])
    
    pred_paper = predict_model(model_paper, data=data)["prediction_label"][0]
    pred_bread = predict_model(model_bread, data=data)["prediction_label"][0]
    pred_milk = predict_model(model_milk, data=data)["prediction_label"][0]
    
    return {
        "input": semana.dict(),
        "predicciones": {
            "sales_paper": round(pred_paper),
            "sales_bread": round(pred_bread),
            "sales_milk": round(pred_milk)
        }
    }

#pip install fastapi uvicorn
#uvicorn api:app --reload --port 8000
