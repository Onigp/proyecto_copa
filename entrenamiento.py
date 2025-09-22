import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

def generate_models(fares_file):
    """
    Entrena modelos de regresión y genera el diccionario de rutas utilizando un solo archivo.

    Args:
        fares_file (str): Ruta al archivo CSV con los detalles de las tarifas de vuelo.
    """
    try:
        print("Paso 1: Cargando datos...")
        fares_df = pd.read_csv(fares_file, low_memory=False)
        
        print("Paso 2: Creando el diccionario de rutas y validando columnas...")
        required_cols = ['airport_1', 'airport_2', 'nsmiles', 'fare', 'Year', 'quarter', 'passengers', 'lf_ms']
        
        if not all(col in fares_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in fares_df.columns]
            raise KeyError(f"Columnas faltantes en los datos: {', '.join(missing_cols)}")
            
        fares_df['route_name'] = fares_df['airport_1'].astype(str) + '-' + fares_df['airport_2'].astype(str)
        fares_df['route_encoded'] = fares_df.groupby(['route_name']).ngroup()
        
        # Crear el diccionario de rutas y guardar
        route_map = fares_df[['route_name', 'route_encoded']].drop_duplicates().set_index('route_name')['route_encoded'].to_dict()
        joblib.dump(route_map, 'route_encodings.pkl')
        print("Diccionario de rutas guardado.")

        print("Paso 3: Preparando los datos para el entrenamiento...")
        
        # --- LÍNEA CLAVE AÑADIDA ---
        # Limpiar filas con valores nulos en las columnas clave para el entrenamiento.
        # Esto soluciona el error 'Input y contains NaN'.
        fares_df.dropna(subset=['passengers', 'lf_ms'], inplace=True)

        features = fares_df[['route_encoded', 'nsmiles', 'fare', 'Year', 'quarter']].copy()
        
        # Asegurarse de que 'capacity' se encuentre si no hay valor
        if 'capacity' not in fares_df.columns:
            fares_df['capacity'] = fares_df['passengers'] / fares_df['lf_ms']
        
        # Corregir la advertencia de copia
        features.loc[:, 'capacity'] = fares_df['capacity']

        # Modelo de demanda (clasificación, pero se usa un regresor)
        demand_target = (fares_df['passengers'] > fares_df['capacity'] * 0.8).astype(int)
        
        # Modelo de pasajeros
        passengers_target = fares_df['passengers']
        
        # Modelo de factor de ocupación
        load_factor_target = fares_df['lf_ms']

        # Entrenar y guardar los modelos
        for model_name, target in [('demand', demand_target), ('passengers', passengers_target), ('load_factor', load_factor_target)]:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(features, target)
            joblib.dump(model, f'flight_{model_name}_model.pkl')
            print(f"Modelo de {model_name} guardado.")
        
        print("\n¡Todos los modelos y el diccionario se generaron correctamente!")

    except KeyError as e:
        print(f"Error: {e}. Por favor, asegúrate de que tu archivo CSV contenga las columnas correctas.")
    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado. Asegúrate de que '{e.filename}' esté en la misma carpeta.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    fares_filename = 'US Airline Flight Routes and Fares 1993-2024.csv'
    generate_models(fares_filename)
