import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos de los candidatos potenciales
potential_candidates_rf = pd.read_csv("./data/potential_candidates_random_forest.csv")
potential_candidates_opt_rf = pd.read_csv(
    "./data/potential_candidates_optimized_rf.csv"
)
potential_candidates_gb = pd.read_csv(
    "./data/potential_candidates_gradient_boosting.csv"
)
potential_candidates_xgb = pd.read_csv("./data/potential_candidates_xgboost.csv")
potential_candidates_xgb_opt = pd.read_csv(
    "./data/potential_candidates_xgboost_optimized.csv"
)
potential_candidates_ensemble = pd.read_csv(
    "./data/potential_candidates_optimized_ensemble.csv"
)

# Cargar las importancias de las características desde los archivos CSV
feature_importances_rf = pd.read_csv("./data/feature_importances_random_forest.csv")
feature_importances_xgb = pd.read_csv("./data/feature_importances_xgboost.csv")

# Título de la aplicación
st.title("Talent Analyzer - Potenciales Candidatos sin Empleo")

# Descripción de la aplicación
st.write(
    """
Esta aplicación muestra los candidatos potenciales sin empleo que, debido a sus características, deberían estar empleados. El dataset es de LinkedIn y contiene información sobre los candidatos, como su experiencia laboral, habilidades, educación, etc.
\nSe han utilizado diversos modelos de machine learning para identificar a estos candidatos.
"""
)

# Mostrar los resultados para cada modelo
st.header("Candidatos Potenciales - Random Forest")
st.dataframe(potential_candidates_rf)

st.header("Candidatos Potenciales - Random Forest Optimizado")
st.dataframe(potential_candidates_opt_rf)

st.header("Candidatos Potenciales - XGBoost")
st.dataframe(potential_candidates_xgb)

st.header("Candidatos Potenciales - XGBoost Optimizado")
st.dataframe(potential_candidates_xgb_opt)

st.header("Candidatos Potenciales - Ensemble Optimizado")
st.dataframe(potential_candidates_ensemble)

st.header("Candidatos Potenciales - Gradient Boosting")
st.dataframe(potential_candidates_gb)

# Visualización de características importantes para Random Forest
st.header("Importancia de Características (Random Forest)")
fig_rf, ax_rf = plt.subplots()
feature_importances_rf.set_index("feature").plot(kind="bar", ax=ax_rf, legend=False)
ax_rf.set_ylabel("Importancia")
plt.title("Importancia de Características - Random Forest")
st.pyplot(fig_rf)

# Visualización de características importantes para XGBoost
st.header("Importancia de Características (XGBoost)")
fig_xgb, ax_xgb = plt.subplots()
feature_importances_xgb.set_index("feature").plot(kind="bar", ax=ax_xgb, legend=False)
ax_xgb.set_ylabel("Importancia")
plt.title("Importancia de Características - XGBoost")
st.pyplot(fig_xgb)

# Comparación de modelos
st.header("Comparación de Modelos")
st.write(
    """
Esta sección permite comparar el rendimiento de diferentes modelos y ajustes de hiperparámetros utilizados
para identificar los candidatos potenciales.
"""
)
