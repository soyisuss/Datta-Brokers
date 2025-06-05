import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import gc
import ast

# Configuración para procesamiento por chunks
CHUNK_SIZE = 5000000  # Reducir para mejor manejo con más columnas

def limpiar_zona(zona_str):
    """Convierte strings de zona ['Nombre'] a solo Nombre, maneja 'Desconocido'"""
    if pd.isna(zona_str) or zona_str == 'Desconocido':
        return 'Desconocido'
    
    try:
        # Si es una lista como "['Zona']"
        if zona_str.startswith('[') and zona_str.endswith(']'):
            zona_list = ast.literal_eval(zona_str)
            return zona_list[0] if zona_list else 'Desconocido'
        else:
            return zona_str
    except:
        return 'Desconocido'

def procesar_chunk_agrupacion_basico(chunk):
    """Agrupación básica: estación + mes + hora"""
    return chunk.groupby(["Ciclo_Estacion_Retiro", "Mes", "Hora_Retiro"]).size().reset_index(name="Num_Retiros")

def procesar_chunk_agrupacion_zonas(chunk):
    """Agrupación por zonas: zona + mes + hora"""
    return chunk.groupby(["Zona_Retiro_Clean", "Mes", "Hora_Retiro"]).size().reset_index(name="Num_Retiros")

def procesar_chunk_agrupacion_flujos(chunk):
    """Agrupación de flujos: zona_origen -> zona_destino + mes + hora"""
    chunk['Flujo_Zonas'] = chunk['Zona_Retiro_Clean'] + ' -> ' + chunk['Zona_Arribo_Clean']
    return chunk.groupby(["Flujo_Zonas", "Mes", "Hora_Retiro"]).size().reset_index(name="Num_Retiros")

def combinar_agrupaciones(agrupaciones):
    """Combina múltiples agrupaciones y suma los valores"""
    if not agrupaciones:
        return pd.DataFrame()
    
    combined = pd.concat(agrupaciones, ignore_index=True)
    
    # Determinar columnas de agrupación basándose en las columnas disponibles
    if 'Flujo_Zonas' in combined.columns:
        group_cols = ["Flujo_Zonas", "Mes", "Hora_Retiro"]
    elif 'Zona_Retiro_Clean' in combined.columns:
        group_cols = ["Zona_Retiro_Clean", "Mes", "Hora_Retiro"]
    else:
        group_cols = ["Ciclo_Estacion_Retiro", "Mes", "Hora_Retiro"]
    
    return combined.groupby(group_cols)["Num_Retiros"].sum().reset_index()

def entrenar_modelo_basico():
    """Entrena modelo básico por estaciones (como el original)"""
    print("🚲 Entrenando modelo básico por estaciones...")
    
    chunk_results = []
    chunk_count = 0
    
    dtype_dict = {
        'Genero_Usuario': 'str',
        'Edad_Usuario': 'float32',
        'Bici': 'float32',
        'Ciclo_Estacion_Retiro': 'str',
        'Hora_Retiro': 'float32',
        'Ciclo_Estacion_Arribo': 'str',
        'Hora_Arribo': 'float32',
        'Anio': 'float32',
        'Mes': 'float32',
        'Zona_Retiro': 'str',
        'Zona_Arribo': 'str'
    }
    
    chunk_reader = pd.read_csv(
        "./data/ecobici_con_zonas.csv", 
        chunksize=CHUNK_SIZE,
        dtype=dtype_dict,
        low_memory=False
    )
    
    for chunk in chunk_reader:
        chunk_count += 1
        print(f"📄 Procesando chunk {chunk_count} ({len(chunk)} registros)")
        
        # Limpiar datos
        chunk = chunk.dropna(subset=['Ciclo_Estacion_Retiro', 'Mes', 'Hora_Retiro'])
        chunk['Mes'] = chunk['Mes'].astype(int)
        chunk['Hora_Retiro'] = chunk['Hora_Retiro'].astype(int)
        
        if not chunk.empty:
            chunk_grouped = procesar_chunk_agrupacion_basico(chunk)
            chunk_results.append(chunk_grouped)
        
        del chunk
        
        if len(chunk_results) >= 10:
            print("🔄 Combinando chunks intermedios...")
            combined_temp = combinar_agrupaciones(chunk_results)
            chunk_results = [combined_temp]
            gc.collect()
    
    # Combinar resultados finales
    retiros = combinar_agrupaciones(chunk_results)
    del chunk_results
    gc.collect()
    
    # Entrenar modelo
    retiros.rename(columns={"Ciclo_Estacion_Retiro": "Estacion", "Hora_Retiro": "Hora"}, inplace=True)
    
    label_encoder = LabelEncoder()
    retiros['Estacion_encoded'] = label_encoder.fit_transform(retiros['Estacion'].astype(str))
    
    X = retiros[["Estacion_encoded", "Mes", "Hora"]].astype({
        'Estacion_encoded': 'int32', 'Mes': 'int8', 'Hora': 'int8'
    })
    y = retiros["Num_Retiros"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"🚲 Modelo Básico - MAE: {mae:.2f}, R2: {r2:.2f}")
    
    # Guardar modelo
    joblib.dump(model, "modelo_estaciones_xgboost.pkl")
    joblib.dump(label_encoder, "label_encoder_estaciones.pkl")
    
    return model, mae, r2

def entrenar_modelo_zonas():
    """Entrena modelo agregado por zonas"""
    print("🏢 Entrenando modelo por zonas...")
    
    chunk_results = []
    chunk_count = 0
    
    dtype_dict = {
        'Genero_Usuario': 'str',
        'Edad_Usuario': 'float32',
        'Bici': 'float32',
        'Ciclo_Estacion_Retiro': 'str',
        'Hora_Retiro': 'float32',
        'Ciclo_Estacion_Arribo': 'str',
        'Hora_Arribo': 'float32',
        'Anio': 'float32',
        'Mes': 'float32',
        'Zona_Retiro': 'str',
        'Zona_Arribo': 'str'
    }
    
    chunk_reader = pd.read_csv(
        "./data/ecobici_con_zonas.csv", 
        chunksize=CHUNK_SIZE,
        dtype=dtype_dict,
        low_memory=False
    )
    
    for chunk in chunk_reader:
        chunk_count += 1
        print(f"📄 Procesando chunk {chunk_count} ({len(chunk)} registros)")
        
        # Limpiar zonas
        chunk['Zona_Retiro_Clean'] = chunk['Zona_Retiro'].apply(limpiar_zona)
        chunk['Zona_Arribo_Clean'] = chunk['Zona_Arribo'].apply(limpiar_zona)
        
        # Limpiar datos
        chunk = chunk.dropna(subset=['Zona_Retiro_Clean', 'Mes', 'Hora_Retiro'])
        chunk['Mes'] = chunk['Mes'].astype(int)
        chunk['Hora_Retiro'] = chunk['Hora_Retiro'].astype(int)
        
        if not chunk.empty:
            chunk_grouped = procesar_chunk_agrupacion_zonas(chunk)
            chunk_results.append(chunk_grouped)
        
        del chunk
        
        if len(chunk_results) >= 10:
            print("🔄 Combinando chunks intermedios...")
            combined_temp = combinar_agrupaciones(chunk_results)
            chunk_results = [combined_temp]
            gc.collect()
    
    # Combinar resultados finales
    retiros_zonas = combinar_agrupaciones(chunk_results)
    del chunk_results
    gc.collect()
    
    # Entrenar modelo
    retiros_zonas.rename(columns={"Zona_Retiro_Clean": "Zona", "Hora_Retiro": "Hora"}, inplace=True)
    
    label_encoder_zonas = LabelEncoder()
    retiros_zonas['Zona_encoded'] = label_encoder_zonas.fit_transform(retiros_zonas['Zona'].astype(str))
    
    X = retiros_zonas[["Zona_encoded", "Mes", "Hora"]].astype({
        'Zona_encoded': 'int32', 'Mes': 'int8', 'Hora': 'int8'
    })
    y = retiros_zonas["Num_Retiros"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"🏢 Modelo Zonas - MAE: {mae:.2f}, R2: {r2:.2f}")
    
    # Guardar modelo
    joblib.dump(model, "modelo_zonas_xgboost.pkl")
    joblib.dump(label_encoder_zonas, "label_encoder_zonas.pkl")
    
    return model, mae, r2

def entrenar_modelo_flujos():
    """Entrena modelo de flujos entre zonas"""
    print("🔄 Entrenando modelo de flujos entre zonas...")
    
    chunk_results = []
    chunk_count = 0
    
    dtype_dict = {
        'Genero_Usuario': 'str',
        'Edad_Usuario': 'float32',
        'Bici': 'float32',
        'Ciclo_Estacion_Retiro': 'str',
        'Hora_Retiro': 'float32',
        'Ciclo_Estacion_Arribo': 'str',
        'Hora_Arribo': 'float32',
        'Anio': 'float32',
        'Mes': 'float32',
        'Zona_Retiro': 'str',
        'Zona_Arribo': 'str'
    }
    
    chunk_reader = pd.read_csv(
        "./data/ecobici_con_zonas.csv", 
        chunksize=CHUNK_SIZE,
        dtype=dtype_dict,
        low_memory=False
    )
    
    for chunk in chunk_reader:
        chunk_count += 1
        print(f"📄 Procesando chunk {chunk_count} ({len(chunk)} registros)")
        
        # Limpiar zonas
        chunk['Zona_Retiro_Clean'] = chunk['Zona_Retiro'].apply(limpiar_zona)
        chunk['Zona_Arribo_Clean'] = chunk['Zona_Arribo'].apply(limpiar_zona)
        
        # Limpiar datos
        chunk = chunk.dropna(subset=['Zona_Retiro_Clean', 'Zona_Arribo_Clean', 'Mes', 'Hora_Retiro'])
        chunk['Mes'] = chunk['Mes'].astype(int)
        chunk['Hora_Retiro'] = chunk['Hora_Retiro'].astype(int)
        
        if not chunk.empty:
            chunk_grouped = procesar_chunk_agrupacion_flujos(chunk)
            chunk_results.append(chunk_grouped)
        
        del chunk
        
        if len(chunk_results) >= 10:
            print("🔄 Combinando chunks intermedios...")
            combined_temp = combinar_agrupaciones(chunk_results)
            chunk_results = [combined_temp]
            gc.collect()
    
    # Combinar resultados finales
    flujos = combinar_agrupaciones(chunk_results)
    del chunk_results
    gc.collect()
    
    # Entrenar modelo
    flujos.rename(columns={"Hora_Retiro": "Hora"}, inplace=True)
    
    label_encoder_flujos = LabelEncoder()
    flujos['Flujo_encoded'] = label_encoder_flujos.fit_transform(flujos['Flujo_Zonas'].astype(str))
    
    X = flujos[["Flujo_encoded", "Mes", "Hora"]].astype({
        'Flujo_encoded': 'int32', 'Mes': 'int8', 'Hora': 'int8'
    })
    y = flujos["Num_Retiros"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"🔄 Modelo Flujos - MAE: {mae:.2f}, R2: {r2:.2f}")
    
    # Guardar modelo
    joblib.dump(model, "modelo_flujos_xgboost.pkl")
    joblib.dump(label_encoder_flujos, "label_encoder_flujos.pkl")
    
    return model, mae, r2

if __name__ == "__main__":
    print("🚀 Entrenando múltiples modelos con información de zonas...")
    
    # Entrenar los tres modelos
    modelo_basico, mae_basico, r2_basico = entrenar_modelo_basico()
    modelo_zonas, mae_zonas, r2_zonas = entrenar_modelo_zonas()
    modelo_flujos, mae_flujos, r2_flujos = entrenar_modelo_flujos()
    
    # Comparar resultados
    print("\n📊 COMPARACIÓN DE MODELOS:")
    print("=" * 50)
    print(f"🚲 Modelo Básico (Estaciones):  MAE={mae_basico:.2f}, R2={r2_basico:.3f}")
    print(f"🏢 Modelo Zonas:               MAE={mae_zonas:.2f}, R2={r2_zonas:.3f}")
    print(f"🔄 Modelo Flujos:              MAE={mae_flujos:.2f}, R2={r2_flujos:.3f}")
    
    # Determinar mejor modelo
    mejor_modelo = min([
        ('Básico', mae_basico, r2_basico),
        ('Zonas', mae_zonas, r2_zonas),
        ('Flujos', mae_flujos, r2_flujos)
    ], key=lambda x: x[1])  # Menor MAE
    
    print(f"\n🏆 Mejor modelo: {mejor_modelo[0]} (MAE: {mejor_modelo[1]:.2f})")
    
    print("\n✅ Entrenamiento completado!")
    print("📁 Modelos guardados:")
    print("   - modelo_estaciones_xgboost.pkl (básico)")
    print("   - modelo_zonas_xgboost.pkl (agregado por zonas)")
    print("   - modelo_flujos_xgboost.pkl (flujos entre zonas)")
