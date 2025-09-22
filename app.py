import streamlit as st
import pandas as pd
import joblib
import altair as alt
import requests
import io
import os
from streamlit.components.v1 import html

st.set_page_config(layout="wide", page_title="Predicción de Vuelos", page_icon="✈️")

# URLs de descarga de Google Drive (asegúrate de que los enlaces sean públicos y directos)
DRIVE_URLS = {
    "route_encodings.pkl": "https://drive.google.com/uc?id=1uJ74bpggf_dy9HimnLnky1ym3aTGaIZU",
    "flight_demand_model.pkl": "https://drive.google.com/uc?id=1Nz4G1zqscbcPTaqtagK21CVCz1JWTR4Z",
    "load_factor_model.pkl": "https://drive.google.com/uc?id=1rrfe2DVK9yWH2ULaXxEBE4XDbl_i7yVa",
    "passengers_model.pkl": "https://drive.google.com/uc?id=1Zby-f9i8WynyYD-yf1qB-nfu1TXNnX9W",
    "historical_data.csv": "https://drive.google.com/uc?id=12SfLLk-gOdZ4PhggEkMN1o8xjei2kiEz",
}

# Caching para descargar archivos y evitar descargas repetidas
@st.cache_data
def load_data(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Lanza un error para códigos de estado HTTP incorrectos
        
        # Lee el contenido del CSV directamente desde la respuesta
        data = pd.read_csv(io.StringIO(response.content.decode('utf-8')), low_memory=False)
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el archivo CSV: {e}")
        return None

@st.cache_resource
def load_model(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(io.BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el modelo: {e}")
        return None

# Cargar el dataframe de forma global para usar en el análisis histórico
df_historical = load_data(DRIVE_URLS["historical_data.csv"])

# --- IMPORTANTE: CREAR LA COLUMNA 'route' AQUÍ ---
if df_historical is not None:
    df_historical['route'] = df_historical['airport_1'].astype(str) + ' - ' + df_historical['airport_2'].astype(str)


st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    .stSidebar {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .stButton>button {
        color: #121212;
        background-color: #4CAF50;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Encabezado
st.title('✈️ Predictor de Vuelos de Aerolíneas')
st.markdown('¡Bienvenido! Ingresa los detalles de un vuelo para predecir su demanda, el número de pasajeros y el factor de ocupación.')

# Sidebar para la entrada de parámetros del vuelo
st.sidebar.header('Ingresar Parámetros del Vuelo')

def user_input_features():
    # Cargar los datos y modelos
    route_encodings = load_model(DRIVE_URLS["route_encodings.pkl"])
    if route_encodings is None:
        return None, None, None, None, None, None, None

    # Usa las claves del diccionario (los nombres de las rutas) para el selectbox
    sorted_route_names = sorted(route_encodings.keys())
    
    selected_route_name = st.sidebar.selectbox('Selecciona la Ruta de Vuelo:', options=sorted_route_names)
    
    # Obtener el route_encoded
    route_encoded_value = route_encodings.get(selected_route_name, -1)
    
    miles = st.sidebar.number_input('Número de Millas:', min_value=100, max_value=5000, value=1000)
    year = st.sidebar.slider('Año:', min_value=1993, max_value=2025, value=2025)
    quarter = st.sidebar.slider('Trimestre:', min_value=1, max_value=4, value=1)
    fare = st.sidebar.number_input('Tarifa del Vuelo:', min_value=50.0, max_value=2000.0, value=250.00, step=0.01)
    capacity = st.sidebar.number_input('Capacidad de Asientos:', min_value=50, max_value=500, value=180, step=10)
    
    data = {'route_encoded': route_encoded_value,
            'miles': miles,
            'year': year,
            'quarter': quarter,
            'fare': fare,
            'capacity': capacity}
    
    features = pd.DataFrame(data, index=[0])
    
    return features, selected_route_name, miles, year, quarter, fare, capacity

features, selected_route_name, miles, year, quarter, fare, capacity = user_input_features()

# --- Sección de Análisis Histórico (AHORA SIEMPRE VISIBLE) ---
st.header('Análisis Histórico de la Ruta')
if df_historical is not None and selected_route_name is not None:
    # Filtrar datos históricos
    historical_route_data = df_historical[df_historical['route'] == selected_route_name]
    
    if not historical_route_data.empty:
        historical_route_data = historical_route_data.groupby('year_pred')[['passengers', 'demands', 'load_factor']].sum().reset_index()
        
        st.subheader(f'Visualización de los datos de la ruta {selected_route_name} a lo largo de los años.')
        
        # Gráfico de Pasajeros por Año
        st.subheader('Pasajeros por Año')
        
        chart_passengers = alt.Chart(historical_route_data).mark_bar(color='#4c8bf5').encode(
            x=alt.X('year_pred:O', title='Año'),
            y=alt.Y('passengers', title='Número de Pasajeros')
        ).interactive()
        st.altair_chart(chart_passengers, use_container_width=True)
        
    else:
        st.warning(f'No hay datos históricos disponibles para esta ruta desde el año {df_historical["year_pred"].min()}.')
else:
    if df_historical is None:
        st.warning('No se pudo cargar el archivo de datos históricos.')

# --- Condicional para mostrar la predicción (SÓLO CUANDO SE HACE CLIC EN EL BOTÓN) ---
if features is not None and st.sidebar.button('Hacer Predicción'):
    
    st.header('Resultados de la Predicción')
    
    # Cargar modelos
    flight_demand_model = load_model(DRIVE_URLS["flight_demand_model.pkl"])
    load_factor_model = load_model(DRIVE_URLS["load_factor_model.pkl"])
    passengers_model = load_model(DRIVE_URLS["passengers_model.pkl"])

    if flight_demand_model and load_factor_model and passengers_model:
        prediction_demand = flight_demand_model.predict(features)[0]
        prediction_passengers = passengers_model.predict(features)[0]
        prediction_load_factor = load_factor_model.predict(features)[0]
        
        # Mostrar resultados en columnas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Demanda de Vuelo Estimada", f"{prediction_demand:,.0f} personas", "Demanda")
        with col2:
            st.metric("Pasajeros Estimados", f"{prediction_passengers:,.0f} pasajeros", "Pasajeros")
        with col3:
            st.metric("Factor de Ocupación Estimado", f"{prediction_load_factor:.2f}%", "Ocupación")
            
        st.success('¡La predicción se ha completado con éxito!')
        
    else:
        st.error('No se pudieron cargar los modelos de predicción.')

# Sección para incrustar el dashboard de Databricks
st.markdown("---")
st.header("Dashboard de Databricks")
st.markdown("Ten en cuenta que si el dashboard no es público, se te pedirá que inicies sesión.")
# El iframe usa la URL que proporcionaste
st.components.v1.html(f'<iframe src="https://dbc-89a702ac-a7de.cloud.databricks.com/embed/dashboardsv3/01f0925efc9f14bd93a0a54faab352f5?o=3046397561422742" width="100%" height="600" frameborder="0"></iframe>', height=600)
