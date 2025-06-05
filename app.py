import streamlit as st

# Configuración de la página, colocada al principio
st.set_page_config(layout="wide")

# Título del panel
st.title("Selector de Análisis")

# Menú de selección
opcion = st.selectbox("Selecciona el análisis que deseas visualizar:", ("Análisis de Accidentes", "Análisis de Bicicletas"))

# Cargar y mostrar el contenido según la opción seleccionada
if opcion == "Análisis de Accidentes":
    # Aquí, puedes incluir el código de 'accidentes.py'
    import accidentes  # Asegúrate de tenerlo en el mismo directorio o en el PYTHONPATH
    st.write("Este es el análisis de accidentes con datos de bicicletas públicas en CDMX.")
    # Ejecuta las funciones y visualizaciones de 'accidentes.py' en el flujo de trabajo de Streamlit

elif opcion == "Análisis de Bicicletas":
    # Aquí, puedes incluir el código de 'main.py'
    import main  # Asegúrate de tenerlo en el mismo directorio o en el PYTHONPATH
    st.write("Este es el análisis de bicicletas públicas en CDMX y Lyon.")
    # Ejecuta las funciones y visualizaciones de 'main.py' en el flujo de trabajo de Streamlit
