import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Análisis Interactivo de Bicicletas Públicas: CDMX y Lyon")

# Cargar datos


@st.cache_data
def cargar_datos():
    cdmx = pd.read_csv("cdmx_data_series.csv", nrows=10000)
    lyon = pd.read_csv("lyon_data_series.csv", nrows=10000)
    cdmx["ciudad"] = "CDMX"
    lyon["ciudad"] = "Lyon"
    return pd.concat([cdmx, lyon], ignore_index=True)


df = cargar_datos()

# Unir y transformar
occupation_cols = [f'occupation_{i}' for i in range(24)]
df_long = df.melt(
    id_vars=['zone_id', 'lat', 'lon', 'tmin', 'tmax',
             'prcp', 'wspd', 'weekday', 'holiday', 'ciudad'],
    value_vars=occupation_cols,
    var_name='hour',
    value_name='occupation'
)
df_long['hour'] = df_long['hour'].str.extract('(\d+)').astype(int)

# Filtros
ciudad = st.sidebar.selectbox("Ciudad", df_long['ciudad'].unique())
hora_min, hora_max = st.sidebar.slider("Rango de horas", 0, 23, (6, 20))
festivo = st.sidebar.selectbox("¿Es día festivo?", ["Ambos", "Sí", "No"])
temp_min = st.sidebar.slider("Temperatura mínima", float(
    df_long['tmin'].min()), float(df_long['tmin'].max()), (10.0, 30.0))

# Aplicar filtros
df_filtrado = df_long[df_long['ciudad'] == ciudad]
df_filtrado = df_filtrado[(df_filtrado['hour'] >= hora_min) & (
    df_filtrado['hour'] <= hora_max)]
df_filtrado = df_filtrado[(df_filtrado['tmin'] >= temp_min[0]) & (
    df_filtrado['tmin'] <= temp_min[1])]
if festivo == "Sí":
    df_filtrado = df_filtrado[df_filtrado['holiday'] == 1]
elif festivo == "No":
    df_filtrado = df_filtrado[df_filtrado['holiday'] == 0]

# Mostrar resultado
st.subheader(f"Ocupación Promedio por Hora - {ciudad}")
ocup_por_hora = df_filtrado.groupby('hour')['occupation'].mean()
fig, ax = plt.subplots()
sns.lineplot(x=ocup_por_hora.index, y=ocup_por_hora.values, ax=ax)
ax.set_xlabel("Hora")
ax.set_ylabel("Ocupación Promedio")
st.pyplot(fig)

st.subheader("Datos filtrados")
st.dataframe(df_filtrado.head(100))
