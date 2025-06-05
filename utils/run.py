# run_model.py
import pandas as pd
import joblib
import os
import ast

# Crear carpeta de salida si no existe
os.makedirs("Processed", exist_ok=True)

def limpiar_zona(zona_str):
    """Convierte strings de zona ['Nombre'] a solo Nombre, maneja 'Desconocido'"""
    if pd.isna(zona_str) or zona_str == 'Desconocido':
        return 'Desconocido'
    
    try:
        if zona_str.startswith('[') and zona_str.endswith(']'):
            zona_list = ast.literal_eval(zona_str)
            return zona_list[0] if zona_list else 'Desconocido'
        else:
            return zona_str
    except:
        return 'Desconocido'

def generar_predicciones_estaciones():
    """Genera predicciones usando el modelo básico de estaciones"""
    print("🚲 Generando predicciones por estaciones...")
    
    try:
        # Cargar modelo básico
        model = joblib.load("modelo_estaciones_xgboost.pkl")
        encoder = joblib.load("label_encoder_estaciones.pkl")
        
        print(f"✅ Modelo de estaciones cargado: {len(encoder.classes_)} estaciones")
        
        # Usar algunas estaciones representativas
        estaciones_disponibles = encoder.classes_[:100]  # Top 100 estaciones
        mes = 6  # Junio
        horas = list(range(24))
        
        # Crear combinaciones
        future_data = pd.DataFrame([
            {"Estacion": est, "Mes": mes, "Hora": h}
            for est in estaciones_disponibles
            for h in horas
        ])
        
        # Codificar y predecir
        future_data["Estacion_encoded"] = encoder.transform(future_data["Estacion"].astype(str))
        future_data = future_data.astype({
            'Estacion_encoded': 'int32', 'Mes': 'int8', 'Hora': 'int8'
        })
        
        X_pred = future_data[["Estacion_encoded", "Mes", "Hora"]]
        future_data["Prediccion_Retiros"] = model.predict(X_pred).round().astype(int)
        
        # Guardar resultados
        output_file = "Processed/predicciones_estaciones_junio2025.csv"
        future_data[["Estacion", "Mes", "Hora", "Prediccion_Retiros"]].to_csv(
            output_file, index=False
        )
        
        print(f"✅ Predicciones por estaciones guardadas en {output_file}")
        
        # Estadísticas
        print(f"📊 Estaciones: {len(estaciones_disponibles)}, Total predicciones: {len(future_data)}")
        print(f"📈 Promedio retiros por hora: {future_data['Prediccion_Retiros'].mean():.1f}")
        
        return future_data
        
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontró el modelo de estaciones: {e}")
        return None

def generar_predicciones_zonas():
    """Genera predicciones usando el modelo agregado por zonas"""
    print("🏢 Generando predicciones por zonas...")
    
    try:
        # Cargar modelo de zonas
        model = joblib.load("modelo_zonas_xgboost.pkl")
        encoder = joblib.load("label_encoder_zonas.pkl")
        
        print(f"✅ Modelo de zonas cargado: {len(encoder.classes_)} zonas")
        print(f"🔍 Zonas disponibles: {list(encoder.classes_[:10])}...")
        
        # Usar todas las zonas disponibles (excepto 'Desconocido' si existe)
        zonas_disponibles = [zona for zona in encoder.classes_ if zona != 'Desconocido']
        mes = 6  # Junio
        horas = list(range(24))
        
        # Crear combinaciones
        future_data = pd.DataFrame([
            {"Zona": zona, "Mes": mes, "Hora": h}
            for zona in zonas_disponibles
            for h in horas
        ])
        
        # Codificar y predecir
        future_data["Zona_encoded"] = encoder.transform(future_data["Zona"].astype(str))
        future_data = future_data.astype({
            'Zona_encoded': 'int32', 'Mes': 'int8', 'Hora': 'int8'
        })
        
        X_pred = future_data[["Zona_encoded", "Mes", "Hora"]]
        future_data["Prediccion_Retiros"] = model.predict(X_pred).round().astype(int)
        
        # Guardar resultados
        output_file = "Processed/predicciones_zonas_junio2025.csv"
        future_data[["Zona", "Mes", "Hora", "Prediccion_Retiros"]].to_csv(
            output_file, index=False
        )
        
        print(f"✅ Predicciones por zonas guardadas en {output_file}")
        
        # Estadísticas y ranking de zonas
        resumen_zonas = future_data.groupby('Zona')['Prediccion_Retiros'].agg([
            'sum', 'mean', 'max'
        ]).round(1).sort_values('sum', ascending=False)
        
        print(f"📊 Zonas: {len(zonas_disponibles)}, Total predicciones: {len(future_data)}")
        print(f"📈 Top 5 zonas con más retiros predichos:")
        print(resumen_zonas.head())
        
        # Guardar resumen de zonas
        resumen_file = "Processed/resumen_zonas_junio2025.csv"
        resumen_zonas.to_csv(resumen_file)
        print(f"📊 Resumen de zonas guardado en {resumen_file}")
        
        return future_data
        
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontró el modelo de zonas: {e}")
        return None

def generar_predicciones_flujos():
    """Genera predicciones usando el modelo de flujos entre zonas"""
    print("🔄 Generando predicciones de flujos entre zonas...")
    
    try:
        # Cargar modelo de flujos
        model = joblib.load("modelo_flujos_xgboost.pkl")
        encoder = joblib.load("label_encoder_flujos.pkl")
        
        print(f"✅ Modelo de flujos cargado: {len(encoder.classes_)} flujos")
        
        # Usar los flujos más comunes (limitar para evitar explosión combinatorial)
        flujos_disponibles = encoder.classes_[:200]  # Top 200 flujos más comunes
        mes = 6  # Junio
        horas = [7, 8, 9, 17, 18, 19]  # Horas pico
        
        print(f"🔍 Ejemplos de flujos: {list(flujos_disponibles[:5])}")
        
        # Crear combinaciones
        future_data = pd.DataFrame([
            {"Flujo_Zonas": flujo, "Mes": mes, "Hora": h}
            for flujo in flujos_disponibles
            for h in horas
        ])
        
        # Codificar y predecir
        future_data["Flujo_encoded"] = encoder.transform(future_data["Flujo_Zonas"].astype(str))
        future_data = future_data.astype({
            'Flujo_encoded': 'int32', 'Mes': 'int8', 'Hora': 'int8'
        })
        
        X_pred = future_data[["Flujo_encoded", "Mes", "Hora"]]
        future_data["Prediccion_Retiros"] = model.predict(X_pred).round().astype(int)
        
        # Separar origen y destino para análisis
        future_data[['Zona_Origen', 'Zona_Destino']] = future_data['Flujo_Zonas'].str.split(' -> ', expand=True)
        
        # Guardar resultados
        output_file = "Processed/predicciones_flujos_junio2025.csv"
        future_data[["Flujo_Zonas", "Zona_Origen", "Zona_Destino", "Mes", "Hora", "Prediccion_Retiros"]].to_csv(
            output_file, index=False
        )
        
        print(f"✅ Predicciones de flujos guardadas en {output_file}")
        
        # Análisis de flujos más intensos
        top_flujos = future_data.groupby('Flujo_Zonas')['Prediccion_Retiros'].sum().sort_values(ascending=False).head(10)
        print(f"📈 Top 10 flujos más intensos:")
        print(top_flujos)
        
        # Análisis por hora pico
        flujos_por_hora = future_data.groupby('Hora')['Prediccion_Retiros'].sum()
        print(f"📊 Flujos por hora pico:")
        print(flujos_por_hora)
        
        return future_data
        
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontró el modelo de flujos: {e}")
        return None

def comparar_modelos():
    """Compara los resultados de los diferentes modelos"""
    print("📊 Comparando resultados de los modelos...")
    
    try:
        # Leer resultados si existen
        archivos = {
            'estaciones': 'Processed/predicciones_estaciones_junio2025.csv',
            'zonas': 'Processed/predicciones_zonas_junio2025.csv',
            'flujos': 'Processed/predicciones_flujos_junio2025.csv'
        }
        
        resultados = {}
        
        for modelo, archivo in archivos.items():
            if os.path.exists(archivo):
                df = pd.read_csv(archivo)
                total_retiros = df['Prediccion_Retiros'].sum()
                promedio_retiros = df['Prediccion_Retiros'].mean()
                resultados[modelo] = {
                    'total': total_retiros,
                    'promedio': promedio_retiros,
                    'registros': len(df)
                }
        
        if resultados:
            print("\n📈 COMPARACIÓN DE MODELOS:")
            print("=" * 60)
            for modelo, stats in resultados.items():
                print(f"{modelo.upper():12} | Total: {stats['total']:,} | Promedio: {stats['promedio']:.1f} | Registros: {stats['registros']:,}")
            
            # Guardar comparación
            comparacion_df = pd.DataFrame(resultados).T
            comparacion_df.to_csv("Processed/comparacion_modelos.csv")
            print(f"\n💾 Comparación guardada en Processed/comparacion_modelos.csv")
        
    except Exception as e:
        print(f"❌ Error al comparar modelos: {e}")

def main():
    """Función principal que ejecuta todos los modelos"""
    print("🚀 Generando predicciones con múltiples modelos...")
    print("=" * 60)
    
    # Generar predicciones con cada modelo
    pred_estaciones = generar_predicciones_estaciones()
    print()
    
    pred_zonas = generar_predicciones_zonas()
    print()
    
    pred_flujos = generar_predicciones_flujos()
    print()
    
    # Comparar resultados
    comparar_modelos()
    
    print("\n✅ Proceso completado!")
    print("📁 Archivos generados en la carpeta 'Processed/':")
    print("   - predicciones_estaciones_junio2025.csv")
    print("   - predicciones_zonas_junio2025.csv") 
    print("   - predicciones_flujos_junio2025.csv")
    print("   - resumen_zonas_junio2025.csv")
    print("   - comparacion_modelos.csv")
    
    print("\n💡 Casos de uso:")
    print("   🚲 Estaciones: Redistribución específica de bicicletas")
    print("   🏢 Zonas: Planificación urbana y análisis regional")
    print("   🔄 Flujos: Optimización de rutas y logística")

if __name__ == "__main__":
    main()
