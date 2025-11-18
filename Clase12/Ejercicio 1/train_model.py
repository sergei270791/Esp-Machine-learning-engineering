from pycaret.classification import * 
import mlflow
import pandas as pd

#activar el mlflow
#mlflow.set_experiment("banking-autoML")

df = pd.read_csv("/home/SERGEICALLE/airflow/data/churn_bank_automl.csv")


# Configuración de MLflows
mlflow.set_tracking_uri("http://20.51.121.119:5000")
mlflow.set_experiment("banking-autoML-SSCC")
#Iniciarmos el pipeline de pycaret

s = setup(data=df,
          target='cerrara_cuenta',
          session_id=123,
          log_experiment=True,
          experiment_name="banking-autoML-SSCC",
          log_plots=True
          )

#Comparar modelos y el registro en el mlflow
best_model = compare_models()
final_model = tune_model(best_model)
save_model(final_model,"/home/SERGEICALLE/airflow/models/modelo_bank_mlflow")

# Registrar modelo
mlflow.sklearn.log_model(final_model, "mejor_modelo")
mlflow.log_param("modelo_principal", str(final_model))
