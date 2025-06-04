from xgboost_model import XGBoostPredictor
from route_optimizer import RouteOptimizer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class RestockPredictor:
    def __init__(self, model_path="modelo_ocupacion.pkl"):
        self.predictor = XGBoostPredictor()
        self.route_optimizer = RouteOptimizer()
        self.model_path = model_path

    def load_model(self):
        """Cargar modelo entrenado"""
        try:
            self.predictor.load_model(self.model_path)
            print("✅ Modelo cargado exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False

    def predict_hourly_demand(self, target_hour, weather_conditions=None, city="CDMX"):
        """Predecir demanda por hora para todas las zonas"""
        if self.predictor.model is None:
            if not self.load_model():
                return None

        # Condiciones meteorológicas por defecto
        if weather_conditions is None:
            weather_conditions = {
                'tmin': 20.0,
                'tmax': 28.0,
                'prcp': 0.0,
                'wspd': 5.0,
                'weekday': datetime.now().weekday(),
                'holiday': 0
            }

        print(f"Prediciendo ocupación para hora {target_hour}:00 en {city}")

        # Filtrar zonas por ciudad
        city_zones = [
            zone for zone in self.predictor.unique_zones if zone['ciudad'] == city]

        if not city_zones:
            print(f"❌ No se encontraron zonas para la ciudad: {city}")
            return None

        predictions = []

        for zone in city_zones:
            # Crear input para predicción
            zone_input = {
                'zone_id': zone['zone_id'],
                'lat': zone['lat'],
                'lon': zone['lon'],
                'hour': target_hour,
                'ciudad': city,
                **weather_conditions
            }

            try:
                pred = self.predictor.predict_custom([zone_input])
                predictions.append({
                    'zone_id': zone['zone_id'],
                    'lat': zone['lat'],
                    'lon': zone['lon'],
                    'ciudad': city,
                    'predicted_occupation': pred[0],
                    'hour': target_hour
                })
            except Exception as e:
                print(f"Error prediciendo zona {zone['zone_id']}: {e}")
                continue

        return sorted(predictions, key=lambda x: x['predicted_occupation'], reverse=True)

    def generate_restock_plan(self, target_hour, city="CDMX", weather_conditions=None,
                              threshold_percentile=75, depot_coords=None):
        """Generar plan completo de reabastecimiento"""

        print(f"\n🚴 GENERANDO PLAN DE REABASTECIMIENTO")
        print(f"=" * 50)

        # Hacer predicciones
        predictions = self.predict_hourly_demand(
            target_hour, weather_conditions, city)

        if not predictions:
            return {"error": "No se pudieron generar predicciones"}

        print(f"✅ Predicciones generadas para {len(predictions)} zonas")

        # Mostrar top 10 zonas con mayor ocupación
        print(f"\n📊 TOP 10 ZONAS CON MAYOR OCUPACIÓN PREDICHA:")
        for i, pred in enumerate(predictions[:10]):
            print(
                f"  {i+1}. Zona {pred['zone_id']}: {pred['predicted_occupation']:.1f}%")

        # Optimizar ruta
        print(f"\n🗺️  OPTIMIZANDO RUTA DE REABASTECIMIENTO...")
        route_result = self.route_optimizer.optimize_restock_route(
            predictions, threshold_percentile, depot_coords
        )

        if 'error' in route_result:
            return route_result

        print(f"✅ Ruta optimizada")
        print(f"   - Zonas a visitar: {route_result['total_zones']}")
        print(f"   - Distancia total: {route_result['total_distance_km']} km")
        print(
            f"   - Tiempo estimado: {route_result['estimated_time_hours']} horas")

        # Crear mapa
        route_map = self.route_optimizer.create_route_map(route_result)

        return {
            'predictions': predictions,
            'route': route_result,
            'map': route_map,
            'report': self.route_optimizer.generate_route_report(route_result)
        }

    def save_plan_to_files(self, plan, hour, city):
        """Guardar plan en archivos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Guardar predicciones en CSV
        predictions_df = pd.DataFrame(plan['predictions'])
        pred_filename = f"predicciones_{city}_{hour}h_{timestamp}.csv"
        predictions_df.to_csv(pred_filename, index=False)

        # Guardar ruta en CSV
        if 'error' not in plan['route']:
            route_df = pd.DataFrame(plan['route']['route'])
            route_filename = f"ruta_{city}_{hour}h_{timestamp}.csv"
            route_df.to_csv(route_filename, index=False)

        # Guardar mapa HTML
        if plan['map']:
            map_filename = f"mapa_ruta_{city}_{hour}h_{timestamp}.html"
            plan['map'].save(map_filename)

        # Guardar reporte
        report_filename = f"reporte_{city}_{hour}h_{timestamp}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(plan['report'])

        print(f"\n💾 Archivos guardados:")
        print(f"   - Predicciones: {pred_filename}")
        print(f"   - Ruta: {route_filename}")
        print(f"   - Mapa: {map_filename}")
        print(f"   - Reporte: {report_filename}")


def main():
    # Ejemplo de uso
    restock = RestockPredictor()

    # Coordenadas del depósito (ejemplo: centro de CDMX)
    depot = {'lat': 19.4326, 'lon': -99.1332}

    # Condiciones meteorológicas ejemplo
    weather = {
        'tmin': 18.0,
        'tmax': 26.0,
        'prcp': 0.0,
        'wspd': 3.0,
        'weekday': 1,  # Lunes
        'holiday': 0
    }

    # Generar plan para las 14:00 horas
    plan = restock.generate_restock_plan(
        target_hour=14,
        city="CDMX",
        weather_conditions=weather,
        threshold_percentile=70,  # Top 30% de zonas
        depot_coords=depot
    )

    if 'error' not in plan:
        # Mostrar reporte
        print(f"\n📋 REPORTE DETALLADO:")
        print(plan['report'])

        # Guardar archivos
        restock.save_plan_to_files(plan, 14, "CDMX")

        print(f"\n✅ Plan de reabastecimiento generado exitosamente")
    else:
        print(f"❌ Error: {plan['error']}")


if __name__ == "__main__":
    main()
