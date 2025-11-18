import pandas as pd
from pycaret.classification import *
import mlflow
import shap
import matplotlib.pyplot as plt

# 📁 Ruta al dataset (ajusta si tu archivo tiene otro nombre)
DATA_PATH = "/home/SERGEICALLE/airflow/data/fintech_credit_approval.csv"

# ✅ Cargar dataset
df = pd.read_csv(DATA_PATH)

# ✅ Preparar datos
# Convertir variables categóricas si aplica
df["residence_risk_zone"] = df["residence_risk_zone"].astype("category")
df["political_event_last_month"] = df["political_event_last_month"].astype("category")


def winsorize_iqr(df, column):
    """
    Reemplaza los outliers en una columna de un DataFrame usando el método IQR.
    Los valores menores que el límite inferior se reemplazan por ese límite,
    y lo mismo para los superiores.

    Parámetros:
        df (pd.DataFrame): El DataFrame original.
        column (str): El nombre de la columna a tratar.

    Retorna:
        pd.Series: La columna modificada con winsorización aplicada.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Aplicar winsorización
    winsorized = df[column].copy()
    winsorized[winsorized < lower_bound] = lower_bound
    winsorized[winsorized > upper_bound] = upper_bound

    return winsorized

df['monthly_income_usd'] = winsorize_iqr(df, 'monthly_income_usd') # ya que habian negativos y valores muy bajos


# Eliminar ID del usuario
df_model = df.drop(columns=["user_id"])

# ✅ Configurar MLflow local
mlflow.set_tracking_uri("http://20.51.121.119:5000")
mlflow.set_experiment("credit_approval_political_risk")

# ✅ Setup de PyCaret para clasificación
s = setup(
    data=df_model,                       # dataframe con los datos
    target='approved',               # columna objetivo (clase)
    session_id=123,                 # para reproducibilidad
    normalize=True,                 # normaliza variables numéricas
    categorical_features=['residence_risk_zone', 'political_event_last_month'],  # columnas categóricas
    log_experiment=True,
    experiment_name="credit_approval_political_risk",
    numeric_imputation='mean',      # reemplazo de valores faltantes numéricos
    categorical_imputation='mode',  # reemplazo de valores faltantes categóricos
    fix_imbalance=True,             # habilita técnicas de rebalanceo de clase
    fix_imbalance_method='SMOTE',   # método para generar muestras de la clase minoritaria
    verbose=True,
    profile=False,
    use_gpu=False,
    fold_strategy='stratifiedkfold' # mantiene la proporción de clases en folds de validación
)

# ✅ Entrenar y seleccionar mejor modelo
best_model = compare_models(sort='AUC')

# ✅ Evaluar con visualizaciones (ROC, matriz, PR, etc.)
evaluate_model(best_model)

# ✅ Tune_model(): Ajuste automático de hiperparámetros
tuned_best_model = tune_model(best_model,optimize='F1')

# 🧠 Interpretar con SHAP manualmente
X_train_transformed = get_config('X_train_transformed')
explainer = shap.TreeExplainer(tuned_best_model)
shap_values = explainer.shap_values(X_train_transformed)

# Detectar si shap_values es una lista (por clase) o una matriz única
if isinstance(shap_values, list):
    # Modelo binario: usar la matriz de la clase positiva
    shap_matrix = shap_values[1]
else:
    # Modelo binario con una sola matriz
    shap_matrix = shap_values

# Ajustar si hay columna extra (offset)
if shap_matrix.shape[1] == X_train_transformed.shape[1] + 1:
    shap_values_fixed = shap_matrix[:, :-1]
else:
    shap_values_fixed = shap_matrix

# Alinear filas si hay desajuste
min_rows = min(shap_values_fixed.shape[0], X_train_transformed.shape[0])

# Crear el gráfico y guardarlo
plt.figure()
shap.summary_plot(shap_values_fixed[:min_rows], X_train_transformed.iloc[:min_rows], show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches='tight')  # Guardar imagen
plt.close()

# Registrar en MLflow
mlflow.log_artifact("shap_summary.png")

# ✅ Guardar modelo localmente
MODEL_PATH = "/home/SERGEICALLE/airflow/models/credit_approval_model"
save_model(tuned_best_model, MODEL_PATH)

print("✅ Modelo de aprobación de crédito entrenado, evaluado e interpretado con SHAP.")

