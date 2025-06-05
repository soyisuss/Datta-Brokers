import pandas as pd
import geopandas as gpd
import numpy as np
import os
import json
from tqdm import tqdm

def asignar_zonas_ecobici():
    """
    Asigna zonas geográficas a las estaciones de EcoBici basándose en coordenadas reales
    """
    
    # Verificar archivos necesarios
    ecobici_file = "data/ecobici_procesado.csv"
    zonas_file = "cdmx_zones.geojson"
    estaciones_file = "cdmx_stations.json"
    
    if not os.path.exists(ecobici_file):
        print(f"❌ No se encuentra {ecobici_file}")
        return
    
    if not os.path.exists(zonas_file):
        print(f"❌ No se encuentra {zonas_file}")
        return
        
    if not os.path.exists(estaciones_file):
        print(f"❌ No se encuentra {estaciones_file}")
        return
    
    print("📊 Cargando datos de EcoBici...")
    
    # Leer una muestra para obtener estaciones únicas
    sample_data = pd.read_csv(ecobici_file, nrows=100000)
    estaciones_unicas = pd.concat([
        sample_data['Ciclo_Estacion_Retiro'].dropna(),
        sample_data['Ciclo_Estacion_Arribo'].dropna()
    ]).unique()
    
    print(f"📍 Encontradas {len(estaciones_unicas)} estaciones únicas en EcoBici")
    
    print("🗺️ Cargando zonas geográficas reales...")
    
    # Cargar zonas geográficas reales
    zonas_geo = gpd.read_file(zonas_file)
    zonas_geo = zonas_geo.to_crs(epsg=3857)  # Proyección métrica
    
    # Si existe cdmx_zones.csv, cargar nombres
    if os.path.exists('cdmx_zones.csv'):
        zonas_data = pd.read_csv('cdmx_zones.csv')
        if 'name' in zonas_data.columns:
            zonas_geo['name'] = zonas_data['name'].astype('category')
    
    print(f"🏢 Cargadas {len(zonas_geo)} zonas reales")
    
    print("🚲 Cargando estaciones reales de EcoBici...")
    
    # Cargar estaciones reales
    with open(estaciones_file, 'r') as f:
        estaciones_json = json.load(f)
    
    # Extraer datos de estaciones
    if 'data' in estaciones_json and 'stations' in estaciones_json['data']:
        estaciones_reales = pd.DataFrame(estaciones_json['data']['stations'])
    else:
        estaciones_reales = pd.DataFrame(estaciones_json)
    
    print(f"🚲 Encontradas {len(estaciones_reales)} estaciones reales")
    
    # Limpiar datos de estaciones (como en cdmx_generate_series.py)
    columnas_a_eliminar = ['name', 'rental_methods', 'short_name', 
                          'eightd_has_key_dispenser', 'is_charging', 
                          'has_kiosk', 'electric_bike_surcharge_waiver', 
                          'external_id']
    
    # Eliminar columnas que existan
    columnas_existentes = [col for col in columnas_a_eliminar if col in estaciones_reales.columns]
    if columnas_existentes:
        estaciones_reales = estaciones_reales.drop(columns=columnas_existentes)
    
    # Asegurar que station_id sea numérico
    estaciones_reales['station_id'] = pd.to_numeric(estaciones_reales['station_id'], errors='coerce')
    
    # Filtrar estaciones que están en nuestros datos de EcoBici
    estaciones_reales = estaciones_reales[estaciones_reales['station_id'].isin(estaciones_unicas)]
    
    print(f"🔗 {len(estaciones_reales)} estaciones coinciden con datos de EcoBici")
    
    # Crear GeoDataFrame de estaciones reales
    geo_estaciones = gpd.GeoDataFrame(
        estaciones_reales,
        crs="EPSG:4326",
        geometry=gpd.points_from_xy(estaciones_reales["lon"], estaciones_reales["lat"])
    )
    geo_estaciones = geo_estaciones.to_crs(epsg=3857)
    
    print("🔗 Asignando zonas a estaciones...")
    
    # Asignar zona a cada estación (similar a cdmx_generate_series.py)
    def asignar_zona(geometry):
        """Encuentra la zona que contiene la estación"""
        zonas_contenedoras = zonas_geo[zonas_geo.contains(geometry)]
        if not zonas_contenedoras.empty:
            # Usar 'name' si existe, sino usar índice o zone_id
            if 'name' in zonas_contenedoras.columns:
                return zonas_contenedoras.iloc[0]['name']
            elif 'zone_id' in zonas_contenedoras.columns:
                return zonas_contenedoras.iloc[0]['zone_id']
            else:
                return f'Zona_{zonas_contenedoras.index[0]}'
        else:
            return 'Desconocido'
    
    geo_estaciones['zone_id'] = geo_estaciones.geometry.apply(asignar_zona)
    
    print(f"✅ Zonas asignadas. {len(geo_estaciones[geo_estaciones['zone_id'] != 'Desconocido'])} estaciones con zona conocida")
    
    # Crear diccionario de mapeo estación -> zona
    estacion_zona_map = dict(zip(geo_estaciones['station_id'], geo_estaciones['zone_id']))
    
    # Para estaciones que no están en el JSON, asignar zona desconocida
    for estacion in estaciones_unicas:
        if estacion not in estacion_zona_map:
            estacion_zona_map[estacion] = 'Desconocido'
    
    print("💾 Procesando archivo de EcoBici por chunks...")
    
    # Procesar el archivo original por chunks y agregar zonas
    chunk_size = 50000
    output_file = "data/ecobici_con_zonas.csv"
    first_chunk = True
    
    chunk_reader = pd.read_csv(ecobici_file, chunksize=chunk_size)
    
    for i, chunk in enumerate(tqdm(chunk_reader, desc="Procesando chunks")):
        # Asignar zonas
        chunk['Zona_Retiro'] = chunk['Ciclo_Estacion_Retiro'].map(estacion_zona_map)
        chunk['Zona_Arribo'] = chunk['Ciclo_Estacion_Arribo'].map(estacion_zona_map)
        
        # Llenar valores faltantes con 'Desconocido'
        chunk['Zona_Retiro'] = chunk['Zona_Retiro'].fillna('Desconocido')
        chunk['Zona_Arribo'] = chunk['Zona_Arribo'].fillna('Desconocido')
        
        # Guardar chunk
        chunk.to_csv(
            output_file,
            mode='a' if not first_chunk else 'w',
            header=first_chunk,
            index=False
        )
        first_chunk = False
    
    print(f"✅ Archivo con zonas guardado en {output_file}")
    
    # Guardar mapeo de estaciones a zonas
    mapping_df = pd.DataFrame(list(estacion_zona_map.items()), 
                             columns=['Estacion', 'Zona'])
    mapping_df.to_csv("data/mapeo_estaciones_zonas.csv", index=False)
    
    # Guardar información detallada de estaciones con coordenadas
    geo_estaciones_info = geo_estaciones[['station_id', 'lat', 'lon', 'zone_id', 'capacity']].copy()
    geo_estaciones_info.to_csv("data/estaciones_con_zonas_detalle.csv", index=False)
    
    return estacion_zona_map

def analizar_zonas_ecobici():
    """
    Analiza la distribución de viajes por zona usando datos reales
    """
    print("📊 Analizando distribución por zonas...")
    
    try:
        # Leer datos con zonas
        df = pd.read_csv("data/ecobici_con_zonas.csv", nrows=100000)
        
        print(f"📊 Registros analizados: {len(df):,}")
        
        print("\n🔍 Análisis de zonas de retiro:")
        zona_retiro_stats = df['Zona_Retiro'].value_counts().head(15)
        print(zona_retiro_stats)
        
        print("\n🔍 Análisis de zonas de arribo:")
        zona_arribo_stats = df['Zona_Arribo'].value_counts().head(15)
        print(zona_arribo_stats)
        
        print("\n🔍 Flujos entre zonas más comunes:")
        flujos_zonas = df.groupby(['Zona_Retiro', 'Zona_Arribo']).size().sort_values(ascending=False).head(15)
        print(flujos_zonas)
        
        # Análisis de zonas desconocidas
        retiros_desconocidos = (df['Zona_Retiro'] == 'Desconocido').sum()
        arribos_desconocidos = (df['Zona_Arribo'] == 'Desconocido').sum()
        
        print(f"\n⚠️ Estadísticas de zonas desconocidas:")
        print(f"Retiros en zona desconocida: {retiros_desconocidos:,} ({retiros_desconocidos/len(df)*100:.1f}%)")
        print(f"Arribos en zona desconocida: {arribos_desconocidos:,} ({arribos_desconocidos/len(df)*100:.1f}%)")
        
        # Convertir flujos_zonas a formato serializable
        flujos_dict = {}
        for (zona_retiro, zona_arribo), count in flujos_zonas.items():
            key = f"{zona_retiro} -> {zona_arribo}"
            flujos_dict[key] = int(count)
        
        # Guardar resumen
        resumen = {
            'zonas_retiro': {str(k): int(v) for k, v in zona_retiro_stats.to_dict().items()},
            'zonas_arribo': {str(k): int(v) for k, v in zona_arribo_stats.to_dict().items()},
            'flujos_principales': flujos_dict,
            'estadisticas': {
                'total_registros': int(len(df)),
                'retiros_desconocidos': int(retiros_desconocidos),
                'arribos_desconocidos': int(arribos_desconocidos),
                'porcentaje_retiros_desconocidos': round(retiros_desconocidos/len(df)*100, 2),
                'porcentaje_arribos_desconocidos': round(arribos_desconocidos/len(df)*100, 2)
            }
        }
        
        with open('data/resumen_zonas_ecobici.json', 'w', encoding='utf-8') as f:
            json.dump(resumen, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resumen guardado en data/resumen_zonas_ecobici.json")
        
    except FileNotFoundError:
        print("❌ Primero ejecuta la asignación de zonas")
    except Exception as e:
        print(f"❌ Error al analizar zonas: {e}")

def verificar_cobertura_estaciones():
    """
    Verifica qué porcentaje de estaciones de EcoBici tienen coordenadas conocidas
    """
    print("🔍 Verificando cobertura de estaciones...")
    
    try:
        # Leer mapeo
        mapeo = pd.read_csv("data/mapeo_estaciones_zonas.csv")
        
        total_estaciones = len(mapeo)
        estaciones_conocidas = len(mapeo[mapeo['Zona'] != 'Desconocido'])
        
        print(f"📊 Total de estaciones: {total_estaciones}")
        print(f"✅ Estaciones con zona conocida: {estaciones_conocidas}")
        print(f"❓ Estaciones desconocidas: {total_estaciones - estaciones_conocidas}")
        print(f"📈 Porcentaje de cobertura: {estaciones_conocidas/total_estaciones*100:.1f}%")
        
        # Mostrar algunas estaciones desconocidas
        desconocidas = mapeo[mapeo['Zona'] == 'Desconocido']['Estacion'].head(10)
        if len(desconocidas) > 0:
            print(f"\n❓ Algunas estaciones sin zona: {list(desconocidas)}")
            
    except FileNotFoundError:
        print("❌ No se encuentra el archivo de mapeo")

if __name__ == "__main__":
    # Ejecutar asignación de zonas
    mapeo = asignar_zonas_ecobici()
    
    # Verificar cobertura
    verificar_cobertura_estaciones()
    
    # Analizar resultados
    analizar_zonas_ecobici()
    
    print("\n✅ Proceso completado!")
    print("📁 Archivos generados:")
    print("   - data/ecobici_con_zonas.csv")
    print("   - data/mapeo_estaciones_zonas.csv") 
    print("   - data/estaciones_con_zonas_detalle.csv")
    print("   - data/resumen_zonas_ecobici.json")