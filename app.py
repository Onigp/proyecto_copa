import streamlit as st
import pandas as pd
import joblib
import requests
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit.components.v1 import html

# Diccionario con las URLs de los archivos en Google Drive
# Estos son los enlaces de descarga directa
DRIVE_URLS = {
    "route_encodings.pkl": "https://drive.google.com/uc?id=1o54H4gk7OibdsvxjV7h3PO9OgKLCSSii",
    "flight_demand_model.pkl": "https://drive.google.com/uc?id=1KGifUzizc5CgxMW4dwPmUFpSdtFElTA0",
    "load_factor_model.pkl": "https://drive.google.com/uc?id=1N0tRbmvBjQse-k8DAgNrO2gXTSgBnaAc",
    "passengers_model.pkl": "https://drive.google.com/uc?id=1LWQdAb8W6dQdbDXlHxxIDW5vR4ZFI6Nt",
    "historical_data.csv": "https://drive.google.com/uc?id=12SfLLk-gOdZ4PhggEkMN1o8xjei2kiEz",
}

@st.cache_resource
def load_data(url):
    """Carga datos desde una URL de Google Drive y los cachea."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return pd.read_csv(io.BytesIO(response.content), low_memory=False)
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el archivo CSV desde Google Drive: {e}")
        return None

@st.cache_resource
def load_model(url):
    """Carga un modelo o diccionario desde una URL de Google Drive y lo cachea."""
    try:
        # Descarga el contenido completo de la URL
        response = requests.get(url)
        response.raise_for_status()
        # Carga el modelo desde la memoria
        return joblib.load(io.BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el modelo desde Google Drive: {e}")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None
    
@st.cache_data
def get_route_full_names(historical_df):
    """Crea un diccionario de nombres de ruta completos para la visualización."""
    route_full_names_df = historical_df[['route_name', 'city1', 'city2']].drop_duplicates()
    route_full_names_df['full_name'] = route_full_names_df['city1'] + ' - ' + route_full_names_df['city2']
    return pd.Series(route_full_names_df.full_name.values, index=route_full_names_df.route_name).to_dict()

# Título y descripción
st.set_page_config(page_title="Predicción de Vuelos", layout="wide")
st.title("✈️ Predictor de Vuelos de Aerolíneas")
st.markdown("¡Bienvenido! Ingresa los detalles de un vuelo para predecir su demanda, el número de pasajeros y el factor de ocupación.")

# Cargar los modelos y el diccionario de rutas
route_encodings = load_model(DRIVE_URLS["route_encodings.pkl"])
demand_model = load_model(DRIVE_URLS["flight_demand_model.pkl"])
passengers_model = load_model(DRIVE_URLS["passengers_model.pkl"])
load_factor_model = load_model(DRIVE_URLS["load_factor_model.pkl"])

# Cargar los datos históricos para las gráficas y la información de la ruta
historical_data = load_data(DRIVE_URLS["historical_data.csv"])

# Verificar que todos los recursos se hayan cargado antes de continuar
if (historical_data is not None and 
    route_encodings is not None and 
    demand_model is not None and 
    passengers_model is not None and 
    load_factor_model is not None):
    
    # Preprocesamiento de datos y obtención de nombres de rutas
    historical_data['route_name'] = historical_data['airport_1'].astype(str) + '-' + historical_data['airport_2'].astype(str)
    historical_data.dropna(subset=['passengers', 'lf_ms', 'route_name', 'city1', 'city2'], inplace=True)
    historical_data['route_encoded'] = historical_data['route_name'].map(route_encodings)
    historical_data.dropna(subset=['route_encoded'], inplace=True)

    # Crear un diccionario para los nombres completos de las rutas
    route_full_names = get_route_full_names(historical_data)
    
    # Sidebar para la entrada de datos del usuario
    st.sidebar.header("Predicciones de Vuelos")

    route_options = sorted(route_encodings.keys())
    selected_route = st.sidebar.selectbox("Selecciona una ruta de vuelo:", route_options)
    
    # Muestra el nombre completo de la ruta seleccionada
    if selected_route in route_full_names:
        st.sidebar.markdown(f"**{route_full_names[selected_route]}**")
    
    col1, col2 = st.sidebar.columns(2)
    nsmiles = col1.number_input("Número de Millas:", min_value=1, value=1000)
    fare = col2.number_input("Tarifa del Vuelo:", min_value=1.0, value=250.0)

    col3, col4 = st.sidebar.columns(2)
    year = col3.number_input("Año:", min_value=historical_data['Year'].min(), max_value=2025, value=2024)
    quarter = col4.number_input("Trimestre:", min_value=1, max_value=4, value=1)
    
    # Esta es la capacidad de asientos que se usará para la predicción
    seat_capacity = st.sidebar.number_input("Capacidad de Asientos:", min_value=1, value=180)


    # Botón para hacer la predicción
    if st.sidebar.button("Predecir Demanda"):
        st.balloons()
        
        # Preprocesar los datos de entrada
        route_encoded_value = route_encodings.get(selected_route, -1)
        
        # Crear DataFrame de entrada para los modelos
        input_df = pd.DataFrame([[route_encoded_value, nsmiles, fare, year, quarter, seat_capacity]], 
                                columns=['route_encoded', 'nsmiles', 'fare', 'Year', 'quarter', 'capacity'])
        
        # Predicciones
        demand_pred = demand_model.predict(input_df)[0]
        passengers_pred = passengers_model.predict(input_df)[0]
        load_factor_pred = load_factor_model.predict(input_df)[0]

        # Mostrar resultados
        st.markdown("---")
        st.header(f"Resultados de la Predicción para la Ruta {selected_route}")
        
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        
        with col_pred1:
            predicted_demand = "Alta" if demand_pred > 0.5 else "Baja"
            st.metric(label="Demanda Predicha", value=predicted_demand)
        
        with col_pred2:
            st.metric(label="Pasajeros Predichos", value=f"{int(passengers_pred):,.0f}")
            st.caption("Número estimado de pasajeros")

        with col_pred3:
            st.metric(label="Factor de Ocupación", value=f"{load_factor_pred:.2%}")
            st.caption("Porcentaje de asientos ocupados")
        
        if predicted_demand == "Alta":
            st.success("Esta ruta probablemente será popular. ¡Una excelente oportunidad!")
        else:
            st.warning("La demanda podría ser baja en esta ruta. Considera ajustar la tarifa.")
        
        st.markdown("---")

    # Visualización de datos históricos
    st.header("Análisis Histórico de la Ruta Seleccionada")

    if selected_route:
        # Filtrar datos históricos por la ruta seleccionada
        filtered_df = historical_data[historical_data['route_name'] == selected_route].copy()
        
        if not filtered_df.empty:
            # Eliminar duplicados para evitar errores de graficación
            filtered_df.drop_duplicates(subset=['Year', 'quarter'], inplace=True)
            filtered_df.sort_values(by=['Year', 'quarter'], inplace=True)
            
            # Crear un solo gráfico con dos ejes Y
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Graficar pasajeros
            fig.add_trace(go.Scatter(x=filtered_df['Year'].astype(str) + ' Q' + filtered_df['quarter'].astype(str),
                                    y=filtered_df['passengers'],
                                    mode='lines+markers',
                                    name='Pasajeros',
                                    marker=dict(color='blue')),
                          secondary_y=False)
            
            # Graficar factor de ocupación
            fig.add_trace(go.Scatter(x=filtered_df['Year'].astype(str) + ' Q' + filtered_df['quarter'].astype(str),
                                    y=filtered_df['lf_ms'],
                                    mode='lines+markers',
                                    name='Factor de Ocupación (lf_ms)',
                                    marker=dict(color='red')),
                          secondary_y=True)

            fig.update_layout(title_text=f"Pasajeros vs. Factor de Ocupación para la Ruta {selected_route}")
            fig.update_xaxes(title_text="Año y Trimestre")
            fig.update_yaxes(title_text="Pasajeros", secondary_y=False)
            fig.update_yaxes(title_text="Factor de Ocupación", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No hay datos históricos disponibles para esta ruta.")
else:
    st.warning("Cargando archivos... Por favor, espera.")

st.markdown("---")
st.header("Dashboard de Databricks")
st.markdown("Ten en cuenta que si el dashboard no es público, se te pedirá que inicies sesión.")
# El iframe usa la URL que proporcionaste
html(f'<iframe src="https://dbc-89a702ac-a7de.cloud.databricks.com/embed/dashboardsv3/01f0925efc9f14bd93a0a54faab352f5?o=3046397561422742" width="100%" height="600" frameborder="0"></iframe>', height=600)
