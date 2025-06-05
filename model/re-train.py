import pandas as pd
from xgboost_model import XGBoostPredictor
import sys
import os


def main():
    print("🔄 RE-ENTRENANDO MODELO CON DATOS ACTUALIZADOS")
    print("=" * 50)

    try:
        # Cargar datos desde la carpeta data
        print("Cargando datos...")
        cdmx = pd.read_csv("../data/cdmx_data_series.csv", nrows=10000)
        lyon = pd.read_csv("../data/lyon_data_series.csv", nrows=10000)

        print(f"CDMX: {cdmx.shape[0]} registros")
        print(f"Lyon: {lyon.shape[0]} registros")

        # Agregar columna de ciudad
        cdmx["ciudad"] = "CDMX"
        lyon["ciudad"] = "Lyon"
        df = pd.concat([cdmx, lyon], ignore_index=True)

        print(f"Datos combinados: {df.shape[0]} registros")
        print(f"Columnas disponibles: {list(df.columns)}")

        # Transformar a formato largo
        occupation_cols = [
            col for col in df.columns if col.startswith('occupation_')]
        print(f"Columnas de ocupación encontradas: {len(occupation_cols)}")

        if not occupation_cols:
            print("❌ Error: No se encontraron columnas de ocupación")
            return

        df_long = df.melt(
            id_vars=[col for col in ['zone_id', 'lat', 'lon', 'tmin', 'tmax',
                                     'prcp', 'wspd', 'weekday', 'holiday', 'ciudad'] if col in df.columns],
            value_vars=occupation_cols,
            var_name='hour',
            value_name='occupation'
        )

        df_long['hour'] = df_long['hour'].str.extract(r'(\d+)').astype(int)
        df_long = df_long.dropna(subset=['occupation'])

        print(f"Datos en formato largo: {df_long.shape[0]} registros")

        # Verificar datos de zonas
        zonas_por_ciudad = df_long.groupby('ciudad')['zone_id'].nunique()
        print(f"Zonas por ciudad:")
        for ciudad, num_zonas in zonas_por_ciudad.items():
            print(f"  {ciudad}: {num_zonas} zonas")

        # Entrenar modelo
        print("\n🤖 Entrenando modelo...")
        predictor = XGBoostPredictor()
        results = predictor.train_with_cv(df_long, target_column='occupation')

        # Guardar modelo actualizado
        model_path = "modelo_ocupacion.pkl"
        predictor.save_model(model_path)

        print(f"\n✅ Modelo re-entrenado y guardado en: {model_path}")

        # Verificar que las zonas se guardaron correctamente
        print(f"\n📍 Información de zonas guardadas:")
        print(f"  Total de zonas: {len(predictor.unique_zones)}")

        ciudades_en_zonas = {}
        for zona in predictor.unique_zones:
            ciudad = zona.get('ciudad', 'Sin ciudad')
            if ciudad not in ciudades_en_zonas:
                ciudades_en_zonas[ciudad] = 0
            ciudades_en_zonas[ciudad] += 1

        for ciudad, count in ciudades_en_zonas.items():
            print(f"  {ciudad}: {count} zonas")

        # Mostrar ejemplo de zona
        if predictor.unique_zones:
            print(f"\n📋 Ejemplo de zona guardada:")
            ejemplo_zona = predictor.unique_zones[0]
            for key, value in ejemplo_zona.items():
                print(f"  {key}: {value}")

        return True

    except FileNotFoundError as e:
        print(f"❌ Error: Archivos no encontrados: {e}")
        return False
    except Exception as e:
        print(f"❌ Error durante re-entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎯 Listo para usar con 'python run.py'")
    else:
        print(f"\n❌ Falló el re-entrenamiento. Revisa los errores arriba.")
