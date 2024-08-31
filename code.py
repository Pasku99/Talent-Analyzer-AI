import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
file_path = './../data/linkedin_profiles_v9.csv'  # Cambia esto por la ruta correcta de tu archivo
df = pd.read_csv(file_path)

# Filtrar perfiles sin empresa actual
df['employed'] = df['current_company:company_id'].notna().astype(int)

# Total profiles in the dataset
total_profiles = len(df)

# Define unemployment condition based on specified fields
unemployed_condition = (
    (df['current_company:company_id'].isna() | (df['current_company:company_id'] == '')) &
    (df['current_company:name'].isna() | (df['current_company:name'] == ''))
)

# Calculate number of employed and unemployed profiles
unemployed = df[unemployed_condition].shape[0]
employed = total_profiles - unemployed

# Calculate percentages
percentage_employed = (employed / total_profiles) * 100
percentage_unemployed = (unemployed / total_profiles) * 100

# Calculate education and certifications count
df['education_count'] = df['education'].apply(lambda x: len(eval(x)) if pd.notnull(x) and x.startswith('[') else 0)
df['certifications_count'] = df['certifications'].apply(lambda x: len(eval(x)) if pd.notnull(x) and x.startswith('[') else 0)

# Profiles with educational details
profiles_with_education = df['education_count'].gt(0).sum()
percentage_with_education = (profiles_with_education / total_profiles) * 100

# Profiles with certifications
profiles_with_certifications = df['certifications_count'].gt(0).sum()
percentage_with_certifications = (profiles_with_certifications / total_profiles) * 100

# Display the results
print(f"Employed: {employed} ({percentage_employed:.2f}%)")
print(f"Unemployed: {unemployed} ({percentage_unemployed:.2f}%)")
print(f"Profiles with Education: {profiles_with_education} ({percentage_with_education:.2f}%)")
print(f"Profiles with Certifications: {profiles_with_certifications} ({percentage_with_certifications:.2f}%)")

# Función para contar entradas en columnas JSON-like
def count_entries(column):
    try:
        return len(eval(column)) if pd.notnull(column) and column.startswith('[') else 0
    except:
        return 0

# Aplicar funciones de extracción
df['education_count'] = df['education'].apply(count_entries)
df['certifications_count'] = df['certifications'].apply(count_entries)
df['languages_count'] = df['languages'].apply(count_entries)
df['volunteer_experience_count'] = df['volunteer_experience'].apply(count_entries)
df['courses_count'] = df['сourses'].apply(count_entries)

# Función para calcular años de experiencia a partir del campo JSON
def extract_experience_years_from_json(experience):
    total_years = 0
    total_months = 0
    try:
        experience_data = json.loads(experience)
        for job in experience_data:
            positions = job.get('positions', [])
            for position in positions:
                duration_short = position.get('duration_short', '')
                # Extraer años y meses de 'duration_short'
                years_match = re.search(r'(\d+)\s*years?', duration_short)
                months_match = re.search(r'(\d+)\s*months?', duration_short)
                if years_match:
                    total_years += int(years_match.group(1))
                if months_match:
                    total_months += int(months_match.group(1))
        total_years += total_months // 12
        return total_years
    except Exception as e:
        print(f"Error procesando la experiencia: {e}")
        return 0

# Aplicar la función de experiencia
df['experience_years'] = df['experience'].apply(lambda x: extract_experience_years_from_json(x) if isinstance(x, str) else 0)

# Preparar las características y etiquetas
features = ['education_count', 'certifications_count', 'languages_count', 
            'volunteer_experience_count', 'courses_count', 'experience_years']
X = df[features].fillna(0)
y = df['employed']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

# Aplicar SMOTE para balancear las clases en el conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Configurar los modelos y parámetros para GridSearch
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

grid_search_xgb = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid_xgb,
    scoring='f1_weighted',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Ejecutar el Grid Search
grid_search_xgb.fit(X_train_resampled, y_train_resampled)
print(f"Mejores parámetros para XGBoost: {grid_search_xgb.best_params_}")

# Evaluar el modelo optimizado con StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Crear el modelo XGBoost con los mejores hiperparámetros
xgb_model = XGBClassifier(**grid_search_xgb.best_params_, random_state=42)

# Convertir y_train a numpy array para evitar problemas de indexación
y_train_array = np.array(y_train)

# Inicializar listas para almacenar las métricas de cada pliegue
precision_scores = []
recall_scores = []
f1_scores = []
confusion_matrices = []

# Realizar la validación cruzada estratificada
for train_index, test_index in skf.split(X_train_scaled, y_train_array):
    # Dividir los datos en entrenamiento y validación para cada pliegue
    X_fold_train, X_fold_val = X_train_scaled[train_index], X_train_scaled[test_index]
    y_fold_train, y_fold_val = y_train_array[train_index], y_train_array[test_index]

    # Aplicar SMOTE en cada fold para balancear las clases
    X_fold_resampled, y_fold_resampled = smote.fit_resample(X_fold_train, y_fold_train)

    # Entrenar el modelo en el pliegue actual
    xgb_model.fit(X_fold_resampled, y_fold_resampled)
    y_fold_pred = xgb_model.predict(X_fold_val)

    # Calcular métricas de rendimiento
    report = classification_report(y_fold_val, y_fold_pred, output_dict=True)
    precision_scores.append(report['weighted avg']['precision'])
    recall_scores.append(report['weighted avg']['recall'])
    f1_scores.append(report['weighted avg']['f1-score'])
    confusion_matrices.append(confusion_matrix(y_fold_val, y_fold_pred))

# Mostrar los resultados promediados de todos los pliegues
print(f"Precision promedio: {np.mean(precision_scores):.4f}")
print(f"Recall promedio: {np.mean(recall_scores):.4f}")
print(f"F1-Score promedio: {np.mean(f1_scores):.4f}")

# Mostrar la matriz de confusión promediada
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
print("Matriz de Confusión Promediada:")
print(mean_confusion_matrix)

# Usar el modelo optimizado para predecir candidatos potenciales sin empleo
unemployed_profiles = df[df['employed'] == 0].copy()
unemployed_features = unemployed_profiles[features].fillna(0)
unemployed_features_scaled = scaler.transform(unemployed_features)

unemployed_profiles['predicted_employed'] = xgb_model.predict(unemployed_features_scaled)

potential_candidates = unemployed_profiles[unemployed_profiles['predicted_employed'] == 1]
potential_candidates.to_csv('./../data/potential_candidates_xgboost.csv', index=False)
print("Candidatos potenciales guardados en 'potential_candidates_xgboost.csv'")

from sklearn.ensemble import RandomForestClassifier

# Configurar los parámetros para GridSearch de Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Configurar GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    scoring='f1_weighted',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Ejecutar el Grid Search
grid_search_rf.fit(X_train_resampled, y_train_resampled)
print(f"Mejores parámetros para Random Forest: {grid_search_rf.best_params_}")

# Crear el modelo de Random Forest con los mejores hiperparámetros
best_params_rf = grid_search_rf.best_params_
rf_model = RandomForestClassifier(**best_params_rf, random_state=42)

# Evaluar el modelo optimizado con StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inicializar listas para almacenar las métricas de cada pliegue
precision_scores_rf = []
recall_scores_rf = []
f1_scores_rf = []
confusion_matrices_rf = []

# Realizar la validación cruzada estratificada
for train_index, test_index in skf.split(X_train_scaled, y_train_array):
    # Dividir los datos en entrenamiento y validación para cada pliegue
    X_fold_train, X_fold_val = X_train_scaled[train_index], X_train_scaled[test_index]
    y_fold_train, y_fold_val = y_train_array[train_index], y_train_array[test_index]

    # Aplicar SMOTE en cada fold para balancear las clases
    X_fold_resampled, y_fold_resampled = smote.fit_resample(X_fold_train, y_fold_train)

    # Entrenar el modelo en el pliegue actual
    rf_model.fit(X_fold_resampled, y_fold_resampled)
    y_fold_pred = rf_model.predict(X_fold_val)

    # Calcular métricas de rendimiento
    report = classification_report(y_fold_val, y_fold_pred, output_dict=True)
    precision_scores_rf.append(report['weighted avg']['precision'])
    recall_scores_rf.append(report['weighted avg']['recall'])
    f1_scores_rf.append(report['weighted avg']['f1-score'])
    confusion_matrices_rf.append(confusion_matrix(y_fold_val, y_fold_pred))

# Mostrar los resultados promediados de todos los pliegues
print(f"Precision promedio (Random Forest): {np.mean(precision_scores_rf):.4f}")
print(f"Recall promedio (Random Forest): {np.mean(recall_scores_rf):.4f}")
print(f"F1-Score promedio (Random Forest): {np.mean(f1_scores_rf):.4f}")

# Mostrar la matriz de confusión promediada
mean_confusion_matrix_rf = np.mean(confusion_matrices_rf, axis=0)
print("Matriz de Confusión Promediada (Random Forest):")
print(mean_confusion_matrix_rf)

# Filtrar perfiles sin empleo
unemployed_profiles_rf = df[df['employed'] == 0].copy()
unemployed_features_rf = unemployed_profiles_rf[features].fillna(0)
unemployed_features_scaled_rf = scaler.transform(unemployed_features_rf)

# Predecir candidatos potenciales con el modelo optimizado de Random Forest
unemployed_profiles_rf['predicted_employed'] = rf_model.predict(unemployed_features_scaled_rf)

# Filtrar perfiles que son candidatos potenciales
potential_candidates_rf = unemployed_profiles_rf[unemployed_profiles_rf['predicted_employed'] == 1]

# Guardar los candidatos potenciales en un archivo CSV
potential_candidates_rf.to_csv('./../data/potential_candidates_random_forest.csv', index=False)
print("Candidatos potenciales guardados en 'potential_candidates_random_forest.csv'")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay

param_dist_rf = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),  # Limitar la profundidad máxima
    'min_samples_split': randint(5, 20),  # Aumentar para regularización
    'min_samples_leaf': randint(2, 10),  # Aumentar para regularización
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Configurar RandomizedSearchCV para Random Forest
random_search_rf = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist_rf,
    n_iter=100,                   # Número de combinaciones de parámetros a probar
    scoring='f1_weighted',        # Métrica de evaluación
    cv=5,                         # Validación cruzada de 5 pliegues
    verbose=2,                    # Nivel de detalle en la salida
    random_state=42,
    n_jobs=-1                     # Usar todos los núcleos disponibles
)

# Ejecutar Randomized Search
random_search_rf.fit(X_train_resampled, y_train_resampled)
print(f"Mejores parámetros después de Randomized Search: {random_search_rf.best_params_}")

# Crear el modelo de Random Forest con los mejores hiperparámetros encontrados
# Reajustar el modelo usando class_weight='balanced'
# Ajustar el modelo Random Forest con class_weight
optimized_rf_model = RandomForestClassifier(
    **random_search_rf.best_params_,
    class_weight={0: 1.5, 1: 1},  # Dar más peso a la clase 0
    random_state=42
)
optimized_rf_model.fit(X_train_scaled, y_train)

# Predecir probabilidades en el conjunto de prueba
y_probs = optimized_rf_model.predict_proba(X_test_scaled)[:, 1]

# Calcular la curva de precisión-recall
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Calcular F1-Score de forma más robusta
f1_scores = np.where(precision + recall > 0, 2 * (precision * recall) / (precision + recall), 0)

# Encontrar el umbral óptimo para maximizar el F1-Score
optimal_threshold = thresholds[f1_scores.argmax()]
print(f"Umbral óptimo basado en el F1-Score: {optimal_threshold}")

# Establecer el umbral final en 0.45 o 0.50
final_threshold = 0.4125
y_pred_final = (y_probs >= final_threshold).astype(int)
print(f"Classification Report con umbral final {final_threshold}:\n", classification_report(y_test, y_pred_final))

# Entrenar el modelo de Random Forest optimizado para calcular la importancia de las características
optimized_rf_model.fit(X_train_resampled, y_train_resampled)

# Extraer la importancia de las características
feature_importances = pd.Series(optimized_rf_model.feature_importances_, index=features)

# Ordenar las importancias en orden descendente
feature_importances_sorted = feature_importances.sort_values(ascending=False)

# Visualizar las importancias
feature_importances_sorted.plot(kind='bar', title='Importancia de Características')
plt.show()

# Guardar las importancias en un archivo CSV para usar en la aplicación Streamlit
feature_importances_df = pd.DataFrame({
    'feature': feature_importances_sorted.index,
    'importance': feature_importances_sorted.values
})

# Guarda el DataFrame de importancias en un CSV
feature_importances_df.to_csv('./../data/feature_importances.csv', index=False)

print("Importancias de características guardadas en './data/feature_importances.csv'")

# Seleccionar las características más importantes (por encima de un umbral)
important_features = feature_importances[feature_importances > 0.05].index.tolist()
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# Definir un ensemble stacking con tres modelos optimizados
estimators = [
    ('rf', RandomForestClassifier(**random_search_rf.best_params_, random_state=42)),
    ('xgb', XGBClassifier(**grid_search_xgb.best_params_, random_state=42)),
    ('lgbm', LGBMClassifier(random_state=42))  # Agregar LightGBM al ensemble
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
)

# Entrenar el ensemble
stacking_model.fit(X_train_scaled, y_train)

# Evaluar el ensemble en el conjunto de prueba
y_pred_stacking = stacking_model.predict(X_test_scaled)
print("Classification Report para Ensemble (Stacking) con LightGBM:\n", classification_report(y_test, y_pred_stacking))
