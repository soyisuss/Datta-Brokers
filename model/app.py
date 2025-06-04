import pandas as pd
from xgboost_model import XGBoostPredictor
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("Iniciando entrenamiento del modelo XGBoost...")

    try:
        # Cargar datos
        print("Cargando datos...")
        cdmx = pd.read_csv("../data/cdmx_data_series.csv", nrows=5000)
        lyon = pd.read_csv("../data/lyon_data_series.csv", nrows=5000)

        # Preparar datos
        cdmx["ciudad"] = "CDMX"
        lyon["ciudad"] = "Lyon"
        df = pd.concat([cdmx, lyon], ignore_index=True)

        # Transformar a formato largo
        occupation_cols = [
            col for col in df.columns if col.startswith('occupation_')]

        if not occupation_cols:
            print("Error: No se encontraron columnas de ocupación")
            return

        df_long = df.melt(
            id_vars=[col for col in ['zone_id', 'lat', 'lon', 'tmin', 'tmax',
                                     'prcp', 'wspd', 'weekday', 'holiday', 'ciudad'] if col in df.columns],
            value_vars=occupation_cols,
            var_name='hour',
            value_name='occupation'
        )

        # Corregir la línea problemática usando raw string
        df_long['hour'] = df_long['hour'].str.extract(r'(\d+)').astype(int)
        df_long = df_long.dropna(subset=['occupation'])

        print(f"Datos preparados: {df_long.shape[0]} registros")

        # Crear y entrenar modelo
        print("Creando modelo...")
        predictor = XGBoostPredictor()

        print("Entrenando con cross validation...")
        results = predictor.train_with_cv(df_long, target_column='occupation')

        # Mostrar gráficos
        print("Generando visualizaciones...")
        predictor.plot_results(results)

        # Guardar modelo
        model_path = "modelo_ocupacion.pkl"
        predictor.save_model(model_path)

        # Ejemplo de predicción
        print("\nEjemplo de predicción personalizada:")
        ejemplo = {
            'zone_id': 100,
            'lat': 19.4326,
            'lon': -99.1332,
            'hour': 14,
            'weekday': 1,
            'tmin': 20.0,
            'tmax': 28.0,
            'ciudad': 'CDMX'
        }

        # Agregar campos faltantes con valores por defecto
        for col in predictor.feature_names:
            if col not in ejemplo:
                ejemplo[col] = 0

        prediccion = predictor.predict_custom(ejemplo)
        print(f"Predicción de ocupación: {prediccion[0]:.2f}%")

        print(f"\n✅ Modelo entrenado y guardado exitosamente en: {model_path}")

    except FileNotFoundError as e:
        print(f"❌ Error: No se encontraron los archivos de datos: {e}")
        print("Asegúrate de tener los archivos CSV en la carpeta 'data'")
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")


if __name__ == "__main__":
    main()
