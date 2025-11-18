from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pycaret.clustering import load_model, assign_model

# Cargar modelo previamente entrenado
modelo = load_model("/home/SERGEICALLE/airflow/models/cluster_model_clientes")

# Crear instancia de API
app = FastAPI()

# Esquema del cliente que recibiremos
class Cliente(BaseModel):
    edad: int
    ingresos_mensuales: float
    gastos_mensuales: float
    visitas_app: int
    usa_web: int
    numero_tarjetas: int
    score_crediticio: int
    reclamos_anuales: int
    actividad_mixta: float

@app.post("/predecir_cluster")
def predecir_cluster(cliente: Cliente):
    data = pd.DataFrame([cliente.dict()])
    resultado = assign_model(modelo, data=data)
    cluster = int(resultado['Cluster'][0])
    return {"cluster_asignado": cluster}
