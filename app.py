import streamlit as st
import importlib.util
import sys
import os

# Configuración global de la página
st.set_page_config(
    page_title="EcoBici CDMX - Dashboard",
    page_icon="🚲",
    layout="wide"
)

def load_page(file_path, page_name):
    """Carga una página desde un archivo Python"""
    try:
        spec = importlib.util.spec_from_file_location(page_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'main'):
            module.main()
    except Exception as e:
        st.error(f"Error cargando {page_name}: {str(e)}")

# Sidebar para navegación
st.sidebar.title("🚲 EcoBici CDMX")
st.sidebar.markdown("---")

# Menú de navegación
pagina = st.sidebar.selectbox(
    "Selecciona una página:",
    ["🏠 Inicio", "🚛 Reabastecimiento", "🚨 Accidentes", "📊 Análisis de Bicicletas"]
)

# Página de inicio
if pagina == "🏠 Inicio":
    st.title("🚲 EcoBici CDMX - Dashboard Completo")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🚛 Reabastecimiento")
        st.write("Planifica rutas de reabastecimiento de bicicletas usando modelos predictivos.")
        st.info("• Predicción de demanda\n• Balance de estaciones\n• Mapas de rutas")
    
    with col3:
        st.subheader("🚨 Accidentes")
        st.write("Analiza la relación entre accidentes y ocupación de bicicletas.")
        st.info("• Mapas de riesgo\n• Análisis temporal\n• Predicciones ML")

elif pagina == "📊 Análisis de Bicicletas":
    load_page(".\main.py", "análisis_bicicletas")

# Página de Reabastecimiento
elif pagina == "🚛 Reabastecimiento":
    load_page("app_ecobici.py", "reabastecimiento")

# Página de Accidentes
elif pagina == "🚨 Accidentes":
    load_page("accidentes.py", "accidentes")
