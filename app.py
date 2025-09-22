import streamlit as st
import pandas as pd
import pickle
import numpy as np
import altair as alt
from streamlit.components.v1 import html

# Título y descripción
st.set_page_config(page_title="Predicción de Vuelos", layout="wide")
st.title("✈️ Predictor de Vuelos de Aerolíneas")
st.markdown("¡Bienvenido! Ingresa los detalles de un vuelo para predecir su demanda, el número de pasajeros y el factor de ocupación.")

# Cargar los modelos y el diccionario de rutas
try:
    with open('passengers_model.pkl', 'rb') as f:
        passengers_model = pickle.load(f)
    with open('load_factor_model.pkl', 'rb') as f:
        load_factor_model = pickle.load(f)
    with open('route_encodings.pkl', 'rb') as f:
        route_encodings = pickle.load(f)
    with open('flight_demand_model.pkl', 'rb') as f:
        demand_model = pickle.load(f)

except FileNotFoundError:
    st.error("Error: Archivos de modelo o diccionario de rutas no encontrados.")
    st.markdown("Asegúrate de que `passengers_model.pkl`, `load_factor_model.pkl`, `route_encodings.pkl` y `flight_demand_model.pkl` estén en la misma carpeta.")
    st.stop()
    
# Cargar los datos históricos para las gráficas y la información de la ruta
try:
    file_path = 'US Airline Flight Routes and Fares 1993-2024.csv'
    historical_data = pd.read_csv(file_path, low_memory=False)
    historical_data.dropna(subset=['passengers', 'lf_ms', 'airport_1', 'airport_2', 'city1', 'city2'], inplace=True)
    historical_data['route_name'] = historical_data['airport_1'].astype(str) + '-' + historical_data['airport_2'].astype(str)
    
    # Crea un diccionario para los nombres completos de las rutas
    route_full_names_df = historical_data[['route_name', 'city1', 'city2']].drop_duplicates()
    route_full_names_df['full_name'] = route_full_names_df['city1'] + ' - ' + route_full_names_df['city2']
    route_full_names = pd.Series(route_full_names_df.full_name.values, index=route_full_names_df.route_name).to_dict()

except FileNotFoundError:
    st.warning("Advertencia: No se encontraron los datos históricos para las gráficas. Asegúrate de que `US Airline Flight Routes and Fares 1993-2024.csv` esté en la misma carpeta.")
    historical_data = pd.DataFrame()
    route_full_names = {}

# Crear un mapeo inverso para el menú desplegable
inverse_route_encodings = {v: k for k, v in route_encodings.items()}
sorted_route_names = sorted(route_encodings.keys())

# Interfaz de usuario para la entrada de datos
st.sidebar.header("Ingresar Parámetros del Vuelo")

selected_route_name = st.sidebar.selectbox("Selecciona la Ruta de Vuelo:", sorted_route_names)

# Muestra el nombre completo de la ruta seleccionada
if selected_route_name in route_full_names:
    st.sidebar.markdown(f"**{route_full_names[selected_route_name]}**")

# Asignar valores por defecto para los campos
current_year = 2025
current_quarter = 1

col1, col2 = st.sidebar.columns(2)
nsmiles = col1.number_input("Número de Millas:", min_value=1, value=1000)
fare = col2.number_input("Tarifa del Vuelo:", min_value=1.0, value=250.0)

col3, col4 = st.sidebar.columns(2)
year = col3.number_input("Año:", min_value=1993, max_value=2025, value=current_year)
quarter = col4.number_input("Trimestre:", min_value=1, max_value=4, value=current_quarter)

seat_capacity = st.sidebar.number_input("Capacidad de Asientos:", min_value=1, value=180)

# Botón para hacer la predicción
if st.sidebar.button("Hacer Predicción"):
    st.balloons()
    
    # Preprocesar los datos de entrada
    route_encoded_value = route_encodings.get(selected_route_name, -1)
    
    # Crear un DataFrame con las características para el modelo
    input_data = pd.DataFrame([[nsmiles, fare, year, quarter, route_encoded_value]],
                              columns=['nsmiles', 'fare', 'Year', 'quarter', 'route_encoded'])

    # Hacer predicciones
    predicted_demand_raw = demand_model.predict(input_data)[0]
    predicted_demand = "Alta" if predicted_demand_raw == 1 else "Baja"
    
    predicted_passengers = passengers_model.predict(input_data)[0]
    predicted_load_factor = load_factor_model.predict(input_data)[0]

    # Ajustar las predicciones para que sean más realistas
    predicted_passengers = int(max(0, predicted_passengers))
    predicted_load_factor = min(1.0, max(0.0, predicted_load_factor)) * 100

    # Mostrar resultados
    st.subheader("📊 Resultados de la Predicción:")
    
    col_pred1, col_pred2, col_pred3 = st.columns(3)
    
    with col_pred1:
        st.metric(label="Demanda Predicha", value=predicted_demand)
    
    with col_pred2:
        st.metric(label="Pasajeros Predichos", value=f"{predicted_passengers:,.0f}")
        st.caption("Número estimado de pasajeros")

    with col_pred3:
        st.metric(label="Factor de Ocupación", value=f"{predicted_load_factor:.2f}%")
        st.caption("Porcentaje de asientos ocupados")
    
    st.markdown("---")

# Sección de visualización de datos históricos
st.header("Análisis Histórico de la Ruta")
st.markdown(f"Visualización de los datos de la ruta **{selected_route_name}** a lo largo de los años.")
if selected_route_name in route_full_names:
    st.markdown(f"**{route_full_names[selected_route_name]}**")

if not historical_data.empty:
    historical_data['Year'] = pd.to_numeric(historical_data['Year'], errors='coerce')
    historical_data.dropna(subset=['Year'], inplace=True)
    
    start_year = st.slider("Selecciona el año de inicio del gráfico:", min_value=int(historical_data['Year'].min()), max_value=int(historical_data['Year'].max()), value=2000)

    filtered_data = historical_data[(historical_data['route_name'] == selected_route_name) & (historical_data['Year'] >= start_year)]
    
    if not filtered_data.empty:
        quarterly_data = filtered_data.groupby(['Year', 'quarter']).agg(
            passengers=('passengers', 'sum'),
            lf_ms=('lf_ms', 'mean')
        ).reset_index()

        quarterly_data['Time'] = quarterly_data['Year'].astype(str) + ' Q' + quarterly_data['quarter'].astype(str)

        st.subheader("Pasajeros por Trimestre")
        chart1 = alt.Chart(quarterly_data).mark_line(point=True).encode(
            x=alt.X('Time:O', axis=alt.Axis(title='Año y Trimestre')),
            y=alt.Y('passengers:Q', title='Número de Pasajeros')
        ).properties(
            width=600,
            height=300
        )
        st.altair_chart(chart1, use_container_width=True)
        
        st.subheader("Factor de Ocupación por Trimestre")
        chart2 = alt.Chart(quarterly_data).mark_line(point=True).encode(
            x=alt.X('Time:O', axis=alt.Axis(title='Año y Trimestre')),
            y=alt.Y('lf_ms:Q', title='Factor de Ocupación', axis=alt.Axis(format='%'))
        ).properties(
            width=600,
            height=300
        )
        st.altair_chart(chart2, use_container_width=True)

    else:
        st.warning(f"No hay datos históricos disponibles para esta ruta desde el año {start_year}.")
else:
    st.warning("No se pudieron cargar los datos históricos. Asegúrate de que el archivo CSV esté presente.")
    
# Sección para incrustar el dashboard de Databricks
st.markdown("---")
st.header("Dashboard de Databricks")
st.markdown("Ten en cuenta que si el dashboard no es público, se te pedirá que inicies sesión.")
# El iframe usa la URL que proporcionaste
html(f'<iframe src="https://dbc-89a702ac-a7de.cloud.databricks.com/embed/dashboardsv3/01f0925efc9f14bd93a0a54faab352f5?o=3046397561422742" width="100%" height="600" frameborder="0"></iframe>', height=600)
