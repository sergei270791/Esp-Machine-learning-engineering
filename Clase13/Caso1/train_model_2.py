import pandas as pd
from pycaret.classification import *
import mlflow
import os

# 📁 Ruta absoluta al dataset en tu MV
DATA_PATH = "/home/SERGEICALLE/airflow/data/credit_risk_multiclass.csv"

# ✅ Cargar dataset
df = pd.read_csv(DATA_PATH)

# ✅ Ajustar tipos
df["region"] = df["region"].astype("category")

# ✅ Preparar dataset
df_model = df.drop(columns=["client_id"])

# ✅ Configurar MLflow local
mlflow.set_tracking_uri("http://20.51.121.119:5000/")
mlflow.set_experiment("riesgo_credito_multiclase")

# ✅ Configuración de PyCaret
setup(
    data=df_model,
    target="risk_level",
    session_id=404,
    log_experiment=True,
    experiment_name="riesgo_credito_multiclase",
    verbose=True,
    profile=False,
    use_gpu=False
)

# ✅ Entrenar modelo específico
lightgbm_model = create_model("lightgbm")

# ✅ Evaluar modelo
evaluate_model(lightgbm_model)

# ✅ Guardar modelo localmente
save_model(lightgbm_model, "/home/SERGEICALLE/airflow/models/credit_risk_model")

print("✅ Modelo LightGBM entrenado, evaluado y guardado.")
