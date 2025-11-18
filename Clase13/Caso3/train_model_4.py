import pandas as pd
from pycaret.classification import *
import mlflow
#import mlflow.pycaret

# Cargar dataset
df = pd.read_csv("/home/SERGEICALLE/airflow/data/upsell_insurance.csv")

# Preparar datos
df_model = df.drop(columns=["customer_id"])

# Configurar MLflow local
mlflow.set_tracking_uri("http://20.51.121.119:5000")
mlflow.set_experiment("upsell_seguro")

# Setup de PyCaret
s = setup(
    data=df_model,
    target="accepted_upsell",
    session_id=606,
    log_experiment=True,
    experiment_name="upsell_seguro",
    verbose=True,
    profile=False,
    use_gpu=False
)

# Entrenar modelo y seleccionar el mejor
best_model = compare_models()

# Evaluar con visualizaciones: ROC, PR, matriz confusión, SHAP
evaluate_model(best_model)

# Registrar modelo en MLflow
#mlflow.pycaret.log_model(best_model, "upsell_model")

# Guardar localmente
save_model(best_model, "/home/SERGEICALLE/airflow/models/upsell_model")

print("✅ Modelo de upsell entrenado, evaluado y registrado.")

#mlflow ui --backend-store-uri file:./mlruns --port 5000
#python train_model.py
