from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pycaret.classification import load_model,predict_model


#Voy a crear una instancia para llamar al servicio
app = FastAPI()
model = load_model("/home/SERGEICALLE/airflow/models/modelo_bank_mlflow")

#Voy a crear una clase cliente, DONDE  ESTE LA ESTRUCTURA DE ESTE.

class Cliente(BaseModel):
    edad: int
    segmento: str
    saldo_total:float
    numero_productos:int
    visitas_app_mes:int
    usa_web:int
    usa_tarjeta_credito:int
    reclamos_6m:int
    satisfaccion_encuesta:float
    tasa_credito_personal:float
    rango_ingresos:str
    region:str


#Para crear el API--http://1.1.1.1/predict/
@app.post("/predict")
def predict(cliente:Cliente):
    data = pd.DataFrame([cliente.dict()])
    pred = predict_model(model,data=data)
    #Esto es la respuesta del api
    return {
        "score":float(pred['prediction_score'][0]),
        "prediccion":int(pred['prediction_label'][0])
    }
    