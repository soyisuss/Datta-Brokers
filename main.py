import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk

st.set_page_config(layout="wide")
st.title("Análisis Interactivo de Bicicletas Públicas: CDMX y Lyon")

# Cargar datos


@st.cache_data
def cargar_datos():
    try:
        cdmx = pd.read_csv("./data/cdmx_data_series.csv", nrows=10000)
        lyon = pd.read_csv("./data/lyon_data_series.csv", nrows=10000)
        cdmx["ciudad"] = "CDMX"
        lyon["ciudad"] = "Lyon"
        return pd.concat([cdmx, lyon], ignore_index=True)
    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos: {e}")
        return pd.DataFrame()


df = cargar_datos()

# Verificación de datos
if df.empty:
    st.stop()

st.sidebar.write("Columnas disponibles:", df.columns.tolist())

occupation_cols = [col for col in df.columns if col.startswith('occupation_')]
if not occupation_cols:
    st.error("No se encontraron columnas de ocupación en los datos")
    st.stop()

# Transformar a formato largo
df_long = df.melt(
    id_vars=[col for col in ['zone_id', 'lat', 'lon', 'tmin', 'tmax',
             'prcp', 'wspd', 'weekday', 'holiday', 'ciudad'] if col in df.columns],
    value_vars=occupation_cols,
    var_name='hour',
    value_name='occupation'
)
df_long['hour'] = df_long['hour'].str.extract('(\d+)').astype(int)
df_long = df_long.dropna(subset=['occupation'])

# Filtros
ciudades_disponibles = df_long['ciudad'].unique()
ciudad = st.sidebar.selectbox("Ciudad", ciudades_disponibles)
df_filtrado = df_long[df_long['ciudad'] == ciudad]

zonas = sorted(df_filtrado['zone_id'].unique())
zonas_seleccionadas = st.sidebar.multiselect(
    "Filtrar por zona(s)", zonas, default=zonas)
df_filtrado = df_filtrado[df_filtrado['zone_id'].isin(zonas_seleccionadas)]

hora_min, hora_max = st.sidebar.slider("Rango de horas", 0, 23, (6, 20))

if 'holiday' in df_long.columns:
    festivo = st.sidebar.selectbox("¿Es día festivo?", ["Ambos", "Sí", "No"])
else:
    festivo = "Ambos"

if 'tmin' in df_long.columns:
    temp_min_val = float(df_long['tmin'].min())
    temp_max_val = float(df_long['tmin'].max())
    temp_min = st.sidebar.slider(
        "Temperatura mínima", temp_min_val, temp_max_val, (temp_min_val, temp_max_val))
else:
    temp_min = None

# Aplicar filtros
df_filtrado = df_filtrado[(df_filtrado['hour'] >= hora_min) & (
    df_filtrado['hour'] <= hora_max)]

if temp_min is not None:
    df_filtrado = df_filtrado[(df_filtrado['tmin'] >= temp_min[0]) & (
        df_filtrado['tmin'] <= temp_min[1])]

if 'holiday' in df_filtrado.columns:
    if festivo == "Sí":
        df_filtrado = df_filtrado[df_filtrado['holiday'] == 1]
    elif festivo == "No":
        df_filtrado = df_filtrado[df_filtrado['holiday'] == 0]

if df_filtrado.empty:
    st.warning("No hay datos que coincidan con los filtros seleccionados")
    st.stop()

# Gráfico de ocupación promedio por hora
st.subheader(f"Ocupación Promedio por Hora - {ciudad}")
ocup_por_hora = df_filtrado.groupby('hour')['occupation'].mean()

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=ocup_por_hora.index, y=ocup_por_hora.values, ax=ax, marker='o')
ax.set_xlabel("Hora del día")
ax.set_ylabel("Ocupación Promedio (%)")
ax.set_title(f"Patrón de ocupación - {ciudad}")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Nuevo: Mapa interactivo de zonas
st.subheader("Mapa de zonas según ocupación promedio")
zona_avg = df_filtrado.groupby(['lat', 'lon']).agg(
    {'occupation': 'mean'}).reset_index()
st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=zona_avg['lat'].mean(),
        longitude=zona_avg['lon'].mean(),
        zoom=11,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=zona_avg,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius='occupation * 5000',
        )
    ],
))

# Métricas
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Registros totales", len(df_filtrado))
with col2:
    st.metric("Ocupación promedio", f"{df_filtrado['occupation'].mean():.1f}%")
with col3:
    st.metric("Horas analizadas", len(ocup_por_hora))

# Vista tabular
st.subheader("Datos filtrados (muestra)")
st.dataframe(df_filtrado.head(100))

# Debug
with st.expander("Información de debug"):
    st.write("Columnas en el dataset:", df.columns.tolist())
    st.write("Forma del dataset original:", df.shape)
    st.write("Forma del dataset filtrado:", df_filtrado.shape)
    st.write("Columnas de ocupación encontradas:", occupation_cols)
