from restock_predictor import RestockPredictor
from datetime import datetime


def main():
    print("🚴 SISTEMA DE PREDICCIÓN DE REABASTECIMIENTO DE BICICLETAS")
    print("=" * 60)

    # Inicializar predictor
    restock = RestockPredictor()

    # Parámetros configurables
    HOUR = 14  # Hora objetivo (14:00)
    CITY = "CDMX"  # Ciudad
    THRESHOLD = 75  # Percentil para zonas de alta demanda

    # Coordenadas del depósito
    DEPOT_COORDS = {
        'lat': 19.4326,  # Centro de CDMX
        'lon': -99.1332
    }

    # Condiciones meteorológicas
    WEATHER = {
        'tmin': 20.0,
        'tmax': 28.0,
        'prcp': 0.0,
        'wspd': 5.0,
        'weekday': datetime.now().weekday(),
        'holiday': 0
    }

    print(f"Configuración:")
    print(f"  - Hora objetivo: {HOUR}:00")
    print(f"  - Ciudad: {CITY}")
    print(f"  - Umbral de demanda: {THRESHOLD}%")
    print(f"  - Temperatura: {WEATHER['tmin']}°C - {WEATHER['tmax']}°C")

    # Generar plan
    plan = restock.generate_restock_plan(
        target_hour=HOUR,
        city=CITY,
        weather_conditions=WEATHER,
        threshold_percentile=THRESHOLD,
        depot_coords=DEPOT_COORDS
    )

    if 'error' not in plan:
        # Guardar resultados
        restock.save_plan_to_files(plan, HOUR, CITY)

        print("\n" + "=" * 60)
        print("✅ PLAN GENERADO EXITOSAMENTE")
        print("   Revisa los archivos generados para ver:")
        print("   - Mapa interactivo de la ruta")
        print("   - Predicciones detalladas por zona")
        print("   - Reporte completo de reabastecimiento")
    else:
        print(f"\n❌ Error: {plan['error']}")


if __name__ == "__main__":
    main()
