from xgboost_model import XGBoostPredictor
import pandas as pd

# Ejemplo de uso


def main():
    # Cargar datos (usando tu estructura existente)
    try:
        # Corregir la ruta de los archivos
        cdmx = pd.read_csv("../data/cdmx_data_series.csv", nrows=5000)
        lyon = pd.read_csv("../data/lyon_data_series.csv", nrows=5000)
        cdmx["ciudad"] = "CDMX"
        lyon["ciudad"] = "Lyon"
        df = pd.concat([cdmx, lyon], ignore_index=True)

        # Transformar a formato largo para tener una columna target
        occupation_cols = [
            col for col in df.columns if col.startswith('occupation_')]
        if occupation_cols:
            df_long = df.melt(
                id_vars=[col for col in ['zone_id', 'lat', 'lon', 'tmin', 'tmax',
                                         'prcp', 'wspd', 'weekday', 'holiday', 'ciudad'] if col in df.columns],
                value_vars=occupation_cols,
                var_name='hour',
                value_name='occupation'
            )
            # Corregir el raw string para evitar el warning
            df_long['hour'] = df_long['hour'].str.extract(r'(\d+)').astype(int)
            df_long = df_long.dropna(subset=['occupation'])

            # Crear y entrenar modelo
            predictor = XGBoostPredictor()
            results = predictor.train_with_cv(
                df_long, target_column='occupation')

            # Visualizar resultados
            predictor.plot_results(results)

            # Mostrar las características que necesita el modelo
            print(f"\nCaracterísticas requeridas por el modelo:")
            for i, feature in enumerate(predictor.feature_names):
                print(f"  {i+1}. {feature}")

            # Ejemplo de predicción personalizada - CORREGIDO
            print("\nCreando ejemplo completo con todas las características...")

            # Crear ejemplo con TODAS las características necesarias
            ejemplo_input = {}

            # Llenar con valores de ejemplo para todas las características
            for feature in predictor.feature_names:
                if feature == 'zone_id':
                    ejemplo_input[feature] = 100
                elif feature == 'lat':
                    ejemplo_input[feature] = 19.4326
                elif feature == 'lon':
                    ejemplo_input[feature] = -99.1332
                elif feature == 'hour':
                    ejemplo_input[feature] = 14
                elif feature == 'weekday':
                    ejemplo_input[feature] = 1
                elif feature == 'tmin':
                    ejemplo_input[feature] = 15.0
                elif feature == 'tmax':
                    ejemplo_input[feature] = 25.0
                elif feature == 'prcp':  # precipitación
                    ejemplo_input[feature] = 0.0
                elif feature == 'wspd':  # velocidad del viento
                    ejemplo_input[feature] = 5.0
                elif feature == 'holiday':
                    ejemplo_input[feature] = 0
                elif feature == 'ciudad':
                    ejemplo_input[feature] = 'CDMX'
                elif feature == 'hour_sin':
                    # Se calculará automáticamente en predict_custom
                    continue
                elif feature == 'hour_cos':
                    # Se calculará automáticamente en predict_custom
                    continue
                else:
                    # Para cualquier otra característica, usar valor por defecto
                    ejemplo_input[feature] = 0.0

            print(f"\nEjemplo de entrada:")
            for key, value in ejemplo_input.items():
                print(f"  {key}: {value}")

            prediccion = predictor.predict_custom(ejemplo_input)
            print(f"\n✅ Predicción de ocupación: {prediccion[0]:.2f}%")

            # Guardar modelo
            predictor.save_model("modelo_ocupacion.pkl")

            print(f"\n🎯 Resumen del modelo:")
            print(f"  - R² Score: {results['test_r2']:.4f}")
            print(f"  - RMSE: {results['test_rmse']:.4f}")
            print(f"  - MAE: {results['test_mae']:.4f}")
            print(
                f"  - Características usadas: {len(predictor.feature_names)}")

    except FileNotFoundError as e:
        print(f"❌ Error: Archivos de datos no encontrados: {e}")
        print("Verificando rutas posibles...")

        # Intentar diferentes rutas
        rutas_posibles = [
            "../data/cdmx_data_series.csv",
            "./data/cdmx_data_series.csv",
            "data/cdmx_data_series.csv",
            "../cdmx_data_series.csv",
            "cdmx_data_series.csv"
        ]

        import os
        print(f"Directorio actual: {os.getcwd()}")
        print("Buscando archivos en:")

        for ruta in rutas_posibles:
            if os.path.exists(ruta):
                print(f"✅ Encontrado: {ruta}")
            else:
                print(f"❌ No existe: {ruta}")

        # Listar archivos en directorio actual y padre
        print("\nArchivos en directorio actual:")
        for file in os.listdir("."):
            print(f"  {file}")

        if os.path.exists(".."):
            print("\nArchivos en directorio padre:")
            for file in os.listdir(".."):
                print(f"  {file}")

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
