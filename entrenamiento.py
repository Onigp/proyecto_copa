import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

print("Iniciando el entrenamiento del modelo de demanda...")

# Define la ruta al archivo CSV
file_path = 'US Airline Flight Routes and Fares 1993-2024.csv'

# Verificar si el archivo existe
if not os.path.exists(file_path):
    print(f"Error: Asegúrate de que el archivo CSV '{file_path}' esté en la misma carpeta que el script.")
else:
    try:
        # Cargar el dataframe
        data = pd.read_csv(file_path)

        # Preprocesamiento y limpieza de datos
        # Eliminar las filas con valores faltantes en las columnas clave
        data.dropna(subset=['nsmiles', 'fare', 'Year', 'quarter', 'airport_1', 'airport_2', 'passengers'], inplace=True)
        
        # Crear la columna 'route' para la codificación
        data['route'] = data['airport_1'].astype(str) + '-' + data['airport_2'].astype(str)
        
        # Eliminar rutas que no aparecen al menos 50 veces para simplificar el modelo
        route_counts = data['route'].value_counts()
        rare_routes = route_counts[route_counts < 50].index
        data = data[~data['route'].isin(rare_routes)]

        # Codificación de la ruta
        route_encodings = {route: i for i, route in enumerate(data['route'].unique())}
        data['route_encoded'] = data['route'].map(route_encodings)

        # Crear la columna objetivo 'high_demand'
        # Usamos un umbral para determinar si la demanda es alta o no (100 pasajeros)
        data['high_demand'] = (data['passengers'] > 100).astype(int)

        # Seleccionar las características y el objetivo
        features = ['nsmiles', 'fare', 'Year', 'quarter', 'route_encoded']
        target = 'high_demand'

        X = data[features]
        y = data[target]

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluar el modelo (opcional)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo de demanda: {accuracy:.2f}")

        # Guardar el modelo en un archivo .pkl
        with open('flight_demand_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        
        # Guardar la codificación de las rutas
        with open('route_encodings.pkl', 'wb') as file:
            pickle.dump(route_encodings, file)

        print("Modelo de demanda guardado en 'flight_demand_model.pkl'.")
        print("Diccionario de rutas actualizado en 'route_encodings.pkl'.")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante el entrenamiento: {e}")
