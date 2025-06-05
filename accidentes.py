import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

#st.set_page_config(layout="wide")
st.title("🚲 Análisis de Accidentes vs Ocupación de Bicicletas Públicas")

# --- Cargar Datos ---
@st.cache_data
def cargar_datos():
    acc = pd.read_csv("./data/Accidentes.csv", nrows=20000)
    bikes = pd.read_csv("./data/cdmx_data_series.csv", nrows=10000)

    acc.columns = acc.columns.str.strip().str.lower()
    acc = acc.rename(columns={"lat": "latitud", "lon": "longitud", "hora": "hora_creacion"})
    acc = acc.dropna(subset=["latitud", "longitud", "hora_creacion"])
    acc["hora"] = pd.to_datetime(acc["hora_creacion"], errors="coerce").dt.hour
    acc["fecha"] = pd.to_datetime(acc["fecha_creacion"], errors="coerce")
    acc = acc.dropna(subset=["hora", "fecha"])

    # --- Buscar estación más cercana ---
    stations = bikes[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    tree = BallTree(np.radians(stations[['lat', 'lon']]), metric='haversine')
    dist, ind = tree.query(np.radians(acc[['latitud', 'longitud']]), k=1)
    acc["estacion_lat"] = stations.iloc[ind.flatten()]["lat"].values
    acc["estacion_lon"] = stations.iloc[ind.flatten()]["lon"].values

    # Preparar ocupación
    occ_cols = [col for col in bikes.columns if col.startswith("occupation_")]
    bikes_long = bikes.melt(
        id_vars=['lat', 'lon', 'tmin', 'tmax', 'prcp'],
        value_vars=occ_cols,
        var_name="hour", value_name="occupation"
    )
    bikes_long["hour"] = bikes_long["hour"].str.extract("(\d+)").astype(int)

    # Buscar ocupación por fila
    def buscar_ocupacion(row):
        filtro = (
            (bikes_long["lat"] == row["estacion_lat"]) &
            (bikes_long["lon"] == row["estacion_lon"]) &
            (bikes_long["hour"] == row["hora"])
        )
        resultado = bikes_long[filtro]
        if not resultado.empty:
            return pd.Series([
                resultado["occupation"].values[0],
                resultado["prcp"].values[0],
                resultado["tmin"].values[0],
                resultado["tmax"].values[0],
            ])
        else:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])

    acc[["occupation", "prcp", "tmin", "tmax"]] = acc.apply(buscar_ocupacion, axis=1)
    acc["lluvia"] = acc["prcp"].fillna(0) > 0
    return acc

df = cargar_datos()

# --- Sidebar ---
st.sidebar.title("🎛️ Filtros")
hora_min, hora_max = st.sidebar.slider("Rango de hora", 0, 23, (6, 22))
ocup_min, ocup_max = st.sidebar.slider("Rango ocupación (%)", 0.0, 1.0, (0.0, 1.0))

# Filtro de precipitación (valor continuo de lluvia)
prcp_min, prcp_max = st.sidebar.slider("Rango de Precipitación (mm)", 0.0, 10.0, (0.0, 10.0))
df_f = df[(df["hora"] >= hora_min) & (df["hora"] <= hora_max)]
df_f = df_f[(df_f["occupation"] >= ocup_min) & (df_f["occupation"] <= ocup_max)]
df_f = df_f[(df_f["prcp"] >= prcp_min) & (df_f["prcp"] <= prcp_max)]

# --- Modelo XGBoost para predecir riesgo ---
from xgboost import XGBClassifier

# Filtrar datos para el modelo
df_riesgo = df[["hora", "occupation", "prcp", "tmin", "tmax"]].copy()
df_riesgo = df_riesgo.sample(n=5000, random_state=42)  # Muestra aleatoria para mejorar rendimiento
X = df_riesgo[["hora", "occupation", "prcp", "tmin", "tmax"]]
y = (df_riesgo["occupation"] > 0.5).astype(int)  # Suponiendo que >50% ocupación es riesgo

# Entrenamiento del modelo
model = XGBClassifier()
model.fit(X, y)

# Predicción de riesgo para df_f
df_f["riesgo"] = model.predict_proba(df_f[["hora", "occupation", "prcp", "tmin", "tmax"]])[:, 1]

# --- 📊 Gráfico de Accidentes vs Ocupación por Hora ---
st.subheader("📊 Accidentes vs Ocupación por Hora")
df_hora = df.groupby("hora").agg({"folio": "count", "occupation": "mean"}).reset_index()
df_hora.columns = ["hora", "Accidentes", "Ocupación"]

fig, ax1 = plt.subplots()
sns.lineplot(data=df_hora, x="hora", y="Accidentes", ax=ax1, marker="o", label="Accidentes")
ax2 = ax1.twinx()
sns.lineplot(data=df_hora, x="hora", y="Ocupación", ax=ax2, color="orange", marker="s", label="Ocupación")
ax1.set_ylabel("Accidentes")
ax2.set_ylabel("Ocupación")
ax1.set_xlabel("Hora del día")
st.pyplot(fig)



# --- Mapa de Accidentes Reales ---
st.subheader("🗺️ Mapa de Accidentes Geolocalizados")
mapa_real = df_f[["latitud", "longitud", "prcp"]].dropna().rename(columns={"latitud": "lat", "longitud": "lon"})
mapa_real["color"] = mapa_real["prcp"].apply(lambda x: [255, 0, 0] if x > 0 else [0, 120, 255])

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(latitude=19.43, longitude=-99.13, zoom=10.5),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=mapa_real,
            get_position='[lon, lat]',
            get_color='color',
            get_radius=60,
        )
    ]
))

# --- Mapa de Calor de Riesgo Predicho ---
st.subheader("🗺️ Mapa de Calor de Riesgo Predicho")
mapa_riesgo = df_f[["latitud", "longitud", "riesgo"]].dropna().rename(columns={"latitud": "lat", "longitud": "lon"})
mapa_riesgo["color"] = mapa_riesgo["riesgo"].apply(lambda x: [255, int(255*(1-x)), 0] if x > 0 else [0, 120, 255])

# Barra deslizadora para cambiar el riesgo
riesgo_min, riesgo_max = st.sidebar.slider("Rango de Riesgo Predictivo", 0.0, 1.0, (0.0, 1.0))

mapa_riesgo_filtrado = mapa_riesgo[(mapa_riesgo["riesgo"] >= riesgo_min) & (mapa_riesgo["riesgo"] <= riesgo_max)]

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(latitude=19.43, longitude=-99.13, zoom=10.5),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=mapa_riesgo_filtrado,
            get_position='[lon, lat]',
            get_color='color',
            get_radius=60,
        )
    ]
))

# --- Mapa de Calor de Riesgo por Zona ---
st.subheader("🗺️ Mapa de Calor de Riesgo por Zona")
mapa_zona_riesgo = df_f.groupby(["latitud", "longitud"]).agg({"riesgo": "mean"}).reset_index()
mapa_zona_riesgo["color"] = mapa_zona_riesgo["riesgo"].apply(lambda x: [255, int(255*(1-x)), 0] if x > 0 else [0, 120, 255])

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(latitude=19.43, longitude=-99.13, zoom=10.5),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=mapa_zona_riesgo,
            get_position='[longitud, latitud]',
            get_color='color',
            get_radius=100,
        )
    ]
))
# --- 📌 Métricas Generales ---
st.subheader("📌 Métricas Generales")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total accidentes", len(df))  # Total de accidentes (sin filtros)
with col2:
    st.metric("Accidentes filtrados", len(df_f))  # Accidentes después de los filtros
with col3:
    st.metric("Ocupación promedio", f"{df_f['occupation'].mean():.2f}")  # Ocupación promedio
# --- 📄 Ver muestra de datos ---
with st.expander("📄 Ver muestra de datos"):
    st.dataframe(df_f[["fecha", "hora", "occupation", "prcp", "latitud", "longitud"]].head(100))
