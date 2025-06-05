import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np
import ast

# Configuración de la página
st.set_page_config(
    page_title="EcoBici CDMX - Reabastecimiento",
    page_icon="🚛",
    layout="wide"
)

@st.cache_data
def cargar_datos():
    """Carga datos de estaciones"""
    try:
        return pd.read_csv("data/estaciones_con_zonas_detalle.csv")
    except:
        return pd.DataFrame()

@st.cache_resource
def cargar_modelos():
    """Carga modelos entrenados"""
    try:
        return {
            'zonas': joblib.load("modelo_zonas_xgboost.pkl"),
            'encoder': joblib.load("label_encoder_zonas.pkl")
        }
    except:
        return None

def limpiar_zona(zona_str):
    """Limpia formato de zona"""
    if pd.isna(zona_str):
        return 'Desconocido'
    try:
        if zona_str.startswith('['):
            return ast.literal_eval(zona_str)[0]
        return zona_str
    except:
        return 'Desconocido'

def predecir_demanda_simple(modelo, encoder, mes, hora):
    """Genera predicciones simples y realistas"""
    zonas = [z for z in encoder.classes_ if z != 'Desconocido'][:20]  # Solo top 20 zonas
    
    predicciones = []
    for zona in zonas:
        try:
            zona_encoded = encoder.transform([zona])[0]
            X = pd.DataFrame({
                'Zona_encoded': [zona_encoded],
                'Mes': [mes],
                'Hora': [hora]
            })
            pred = modelo.predict(X)[0]
            predicciones.append({
                'Zona': zona,
                'Demanda': max(0, int(pred * 0.1))  # Factor realista 0.1
            })
        except:
            pass
    
    return pd.DataFrame(predicciones)

def calcular_balance_simple(demanda_df):
    """Calcula balance simple sin flujos complejos"""
    # Simular entradas como 30-70% de la demanda
    demanda_df['Entradas'] = (demanda_df['Demanda'] * np.random.uniform(0.3, 0.7, len(demanda_df))).astype(int)
    demanda_df['Balance'] = demanda_df['Entradas'] - demanda_df['Demanda']
    demanda_df['Necesita'] = demanda_df['Balance'] < -10
    demanda_df['Deficit'] = np.where(demanda_df['Balance'] < -10, abs(demanda_df['Balance']), 0)
    demanda_df['Exceso'] = np.where(demanda_df['Balance'] > 10, demanda_df['Balance'], 0)
    
    return demanda_df

def crear_mapa_simple(deficit_df, exceso_df, estaciones_data):
    """Crea mapa básico"""
    mapa = folium.Map(location=[19.4326, -99.1332], zoom_start=11)
    
    # Limpiar datos de estaciones SIN USAR APPLY
    estaciones_clean = estaciones_data.copy()
    
    # Limpiar zona_limpia manualmente sin apply
    zona_limpia = []
    for zone_id in estaciones_clean['zone_id']:
        if pd.notna(zone_id):
            zona_limpia.append(limpiar_zona(zone_id))
        else:
            zona_limpia.append('Desconocido')
    
    estaciones_clean['zona_limpia'] = zona_limpia
    
    # Agregar zonas déficit
    for _, zona in deficit_df.iterrows():
        if zona['Deficit'] > 0:
            estaciones_zona = estaciones_clean[estaciones_clean['zona_limpia'] == zona['Zona']]
            if not estaciones_zona.empty:
                lat = float(estaciones_zona['lat'].mean())
                lon = float(estaciones_zona['lon'].mean())
                
                folium.Marker(
                    [lat, lon],
                    popup=f"DEFICIT: {zona['Zona']} - {int(zona['Deficit'])} bicis",
                    icon=folium.Icon(color='red', icon='exclamation-sign')
                ).add_to(mapa)
    
    # Agregar zonas exceso
    for _, zona in exceso_df.iterrows():
        if zona['Exceso'] > 0:
            estaciones_zona = estaciones_clean[estaciones_clean['zona_limpia'] == zona['Zona']]
            if not estaciones_zona.empty:
                lat = float(estaciones_zona['lat'].mean())
                lon = float(estaciones_zona['lon'].mean())
                
                folium.Marker(
                    [lat, lon],
                    popup=f"EXCESO: {zona['Zona']} - {int(zona['Exceso'])} bicis",
                    icon=folium.Icon(color='green', icon='ok-sign')
                ).add_to(mapa)
    
    return mapa

def main():
    st.title("🚛 EcoBici CDMX - Reabastecimiento Simple")
    
    # Cargar datos
    modelos = cargar_modelos()
    estaciones_data = cargar_datos()
    
    if not modelos:
        st.error("❌ No se encontraron los modelos")
        return
    
    # Sidebar
    st.sidebar.header("Configuración")
    mes = st.sidebar.selectbox("Mes:", list(range(1, 13)), index=5)
    hora = st.sidebar.selectbox("Hora:", list(range(24)), index=8)
    
    # CAMBIO CRÍTICO: Usar session_state para persistir datos
    if st.sidebar.button("🚀 Generar Plan", type="primary"):
        
        # Generar predicciones y GUARDAR EN SESSION STATE
        with st.spinner("Generando plan..."):
            demanda_df = predecir_demanda_simple(modelos['zonas'], modelos['encoder'], mes, hora)
            balance_df = calcular_balance_simple(demanda_df)
            
            deficit_df = balance_df[balance_df['Deficit'] > 0].sort_values('Deficit', ascending=False)
            exceso_df = balance_df[balance_df['Exceso'] > 0].sort_values('Exceso', ascending=False)
            
            # GUARDAR EN SESSION STATE
            st.session_state.plan_generado = True
            st.session_state.deficit_df = deficit_df
            st.session_state.exceso_df = exceso_df
            st.session_state.balance_df = balance_df
            st.session_state.mes_usado = mes
            st.session_state.hora_usado = hora
    
    # MOSTRAR RESULTADOS PERSISTENTES
    if st.session_state.get('plan_generado', False):
        
        # Recuperar datos guardados
        deficit_df = st.session_state.deficit_df
        exceso_df = st.session_state.exceso_df
        balance_df = st.session_state.balance_df
        mes_usado = st.session_state.mes_usado
        hora_usado = st.session_state.hora_usado
        
        # Indicador de configuración usada
        st.success(f"📊 Plan generado para: **Mes {mes_usado}, Hora {hora_usado}**")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚨 Bicis Necesarias", f"{deficit_df['Deficit'].sum():,}")
        with col2:
            st.metric("✅ Bicis Disponibles", f"{exceso_df['Exceso'].sum():,}")
        with col3:
            st.metric("📍 Zonas a Visitar", len(deficit_df) + len(exceso_df))
        
        # Plan de acción
        col_plan1, col_plan2 = st.columns(2)
        
        with col_plan1:
            st.subheader("🚚 RECOGER")
            if not exceso_df.empty:
                for i, (_, zona) in enumerate(exceso_df.head(5).iterrows(), 1):
                    st.write(f"**{i}.** {zona['Zona']}: **{zona['Exceso']}** bicis")
            else:
                st.info("No hay zonas con exceso")
        
        with col_plan2:
            st.subheader("🎯 ENTREGAR")
            if not deficit_df.empty:
                for i, (_, zona) in enumerate(deficit_df.head(5).iterrows(), 1):
                    st.write(f"**{i}.** {zona['Zona']}: **{zona['Deficit']}** bicis")
            else:
                st.success("No hay zonas con déficit")
        
        # Mapa
        st.subheader("🗺️ Mapa de Ruta")
        if not estaciones_data.empty:
            try:
                mapa = crear_mapa_simple(deficit_df, exceso_df, estaciones_data)
                
                # Usar configuración más simple para st_folium
                map_data = st_folium(
                    mapa, 
                    width=700, 
                    height=400
                )
            except Exception as e:
                st.error(f"Error al cargar el mapa: {str(e)}")
                st.info("Mostrando datos en tabla:")
                
                # Tabla alternativa si falla el mapa
                mapa_data = []
                for _, zona in deficit_df.iterrows():
                    mapa_data.append({'Zona': zona['Zona'], 'Tipo': '🚨 DEFICIT', 'Cantidad': zona['Deficit']})
                for _, zona in exceso_df.iterrows():
                    mapa_data.append({'Zona': zona['Zona'], 'Tipo': '✅ EXCESO', 'Cantidad': zona['Exceso']})
                
                if mapa_data:
                    st.dataframe(pd.DataFrame(mapa_data))
        else:
            st.warning("No se pudieron cargar los datos de estaciones")
        
        # Gráfico simple
        if not balance_df.empty:
            fig = px.bar(
                balance_df.head(10), 
                x='Zona', 
                y='Balance',
                title="Balance por Zona (Top 10)",
                color='Balance',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Botones de acción
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            # Generar datos para descarga
            plan_data = []
            for _, zona in exceso_df.iterrows():
                plan_data.append({
                    'Accion': 'RECOGER',
                    'Zona': zona['Zona'],
                    'Cantidad': zona['Exceso']
                })
            for _, zona in deficit_df.iterrows():
                plan_data.append({
                    'Accion': 'ENTREGAR',
                    'Zona': zona['Zona'],
                    'Cantidad': zona['Deficit']
                })
            
            if plan_data:
                plan_df = pd.DataFrame(plan_data)
                csv = plan_df.to_csv(index=False)
                st.download_button(
                    "📥 Descargar Plan CSV",
                    csv,
                    f"plan_reabastecimiento_{mes_usado:02d}_{hora_usado:02d}.csv",
                    "text/csv"
                )
        
        with col_btn2:
            # Botón para limpiar y generar nuevo
            if st.button("🔄 Nuevo Plan"):
                # Limpiar session state
                for key in ['plan_generado', 'deficit_df', 'exceso_df', 'balance_df', 'mes_usado', 'hora_usado']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    else:
        # Pantalla inicial
        st.info("👆 Configura mes y hora en el panel lateral, luego presiona 'Generar Plan'")
        
        # Estadísticas básicas
        if not estaciones_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🚲 Estaciones", len(estaciones_data))
            with col2:
                st.metric("🏢 Zonas", len(estaciones_data['zone_id'].unique()))

if __name__ == "__main__":
    main()