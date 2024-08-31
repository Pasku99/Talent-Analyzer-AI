# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos de los candidatos potenciales
potential_candidates_rf = pd.read_csv('./data/potential_candidates_random_forest.csv')
potential_candidates_opt = pd.read_csv('./data/potential_candidates_optimized_rf.csv')

# Cargar las importancias de las características desde el archivo CSV
feature_importances_df = pd.read_csv('./data/feature_importances.csv')

# Título de la aplicación
st.title("Talent Analyzer - Potenciales Candidatos sin Empleo")

# Descripción de la aplicación
st.write("""
Esta aplicación muestra los candidatos potenciales sin empleo que, debido a sus características, deberían estar empleados.
Utiliza modelos de machine learning para identificar estos perfiles.
""")

# Mostrar los resultados para Random Forest
st.header("Candidatos Potenciales - Random Forest")
st.dataframe(potential_candidates_rf)

# Mostrar los resultados para el modelo optimizado
st.header("Candidatos Potenciales - Random Forest Optimizado")
st.dataframe(potential_candidates_opt)

# Visualización de características importantes
st.header("Importancia de Características (Random Forest)")
fig, ax = plt.subplots()
feature_importances_df.set_index('feature').plot(kind='bar', ax=ax, legend=False)
ax.set_ylabel('Importancia')
plt.title('Importancia de Características')
st.pyplot(fig)

# Comparación de modelos
st.header("Comparación de Modelos")
st.write("""
Esta sección permite comparar el rendimiento de diferentes modelos y ajustes de hiperparámetros utilizados
para identificar los candidatos potenciales.
""")