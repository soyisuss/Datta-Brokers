import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import folium

class RouteOptimizer:
    def __init__(self):
        self.high_demand_zones = []
        self.distance_matrix = None
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calcular distancia entre dos puntos usando fórmula de Haversine"""
        # Convertir a radianes
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Fórmula de Haversine
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radio de la Tierra en km
        r = 6371
        return c * r
    
    def identify_high_demand_zones(self, predictions, threshold_percentile=75):
        """Identificar zonas de alta demanda"""
        occupations = [p['predicted_occupation'] for p in predictions]
        threshold = np.percentile(occupations, threshold_percentile)
        
        self.high_demand_zones = [
            p for p in predictions 
            if p['predicted_occupation'] >= threshold
        ]
        
        print(f"Zonas de alta demanda identificadas: {len(self.high_demand_zones)}")
        return self.high_demand_zones
    
    def create_distance_matrix(self, zones):
        """Crear matriz de distancias entre zonas"""
        n = len(zones)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = self.haversine_distance(
                        zones[i]['lat'], zones[i]['lon'],
                        zones[j]['lat'], zones[j]['lon']
                    )
                    distances[i][j] = dist
        
        self.distance_matrix = distances
        return distances
    
    def solve_tsp_greedy(self, start_index=0):
        """Resolver TSP usando algoritmo greedy (para demostración)"""
        if self.distance_matrix is None:
            raise ValueError("Crear matriz de distancias primero")
        
        n = len(self.distance_matrix)
        unvisited = set(range(n))
        current = start_index
        route = [current]
        unvisited.remove(current)
        
        total_distance = 0
        
        while unvisited:
            # Encontrar la zona más cercana no visitada
            nearest = min(unvisited, key=lambda x: self.distance_matrix[current][x])
            total_distance += self.distance_matrix[current][nearest]
            route.append(nearest)
            current = nearest
            unvisited.remove(current)
        
        # Regresar al inicio
        total_distance += self.distance_matrix[current][start_index]
        route.append(start_index)
        
        return route, total_distance
    
    def optimize_restock_route(self, predictions, threshold_percentile=75, depot_coords=None):
        """Optimizar ruta de reabastecimiento completa"""
        # Identificar zonas de alta demanda
        high_demand = self.identify_high_demand_zones(predictions, threshold_percentile)
        
        if not high_demand:
            return {"error": "No hay zonas de alta demanda"}
        
        # Si hay depósito, agregarlo al inicio
        if depot_coords:
            depot = {
                'zone_id': 'DEPOT',
                'lat': depot_coords['lat'],
                'lon': depot_coords['lon'],
                'predicted_occupation': 0,
                'ciudad': high_demand[0]['ciudad']
            }
            zones_to_visit = [depot] + high_demand
            start_index = 0
        else:
            zones_to_visit = high_demand
            start_index = 0
        
        # Crear matriz de distancias
        self.create_distance_matrix(zones_to_visit)
        
        # Resolver TSP
        route_indices, total_distance = self.solve_tsp_greedy(start_index)
        
        # Crear ruta con información completa
        route = []
        for i, idx in enumerate(route_indices):
            zone = zones_to_visit[idx].copy()
            zone['order'] = i
            zone['is_depot'] = (idx == 0 and depot_coords is not None)
            route.append(zone)
        
        return {
            'route': route,
            'total_distance_km': round(total_distance, 2),
            'total_zones': len(high_demand),
            'estimated_time_hours': round(total_distance / 30, 1)  # Asumiendo 30 km/h
        }
    
    def create_route_map(self, route_result, map_center=None):
        """Crear mapa interactivo de la ruta"""
        if 'error' in route_result:
            return None
        
        route = route_result['route']
        
        # Determinar centro del mapa
        if map_center is None:
            lats = [zone['lat'] for zone in route]
            lons = [zone['lon'] for zone in route]
            map_center = [np.mean(lats), np.mean(lons)]
        
        # Crear mapa
        m = folium.Map(location=map_center, zoom_start=11)
        
        # Colores para diferentes niveles de ocupación
        def get_color(occupation):
            if occupation >= 80:
                return 'red'
            elif occupation >= 60:
                return 'orange'
            elif occupation >= 40:
                return 'yellow'
            else:
                return 'green'
        
        # Agregar marcadores para cada zona
        for zone in route:
            if zone['is_depot'] if 'is_depot' in zone else False:
                # Marcador especial para depósito
                folium.Marker(
                    [zone['lat'], zone['lon']],
                    popup=f"DEPOT - Punto de inicio",
                    icon=folium.Icon(color='blue', icon='home')
                ).add_to(m)
            else:
                # Marcadores para zonas de reabastecimiento
                folium.Marker(
                    [zone['lat'], zone['lon']],
                    popup=f"Zona {zone['zone_id']}<br>"
                          f"Ocupación: {zone['predicted_occupation']:.1f}%<br>"
                          f"Orden: {zone['order']}",
                    icon=folium.Icon(
                        color=get_color(zone['predicted_occupation']),
                        icon='bicycle'
                    )
                ).add_to(m)
        
        # Agregar líneas de ruta
        route_coords = [[zone['lat'], zone['lon']] for zone in route]
        folium.PolyLine(
            route_coords,
            color='blue',
            weight=3,
            opacity=0.7
        ).add_to(m)
        
        # Agregar información de la ruta
        legend_html = f"""
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Información de Ruta</b></p>
        <p>Distancia total: {route_result['total_distance_km']} km</p>
        <p>Tiempo estimado: {route_result['estimated_time_hours']} horas</p>
        <p>Zonas a reabastecer: {route_result['total_zones']}</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def generate_route_report(self, route_result):
        """Generar reporte detallado de la ruta"""
        if 'error' in route_result:
            return route_result['error']
        
        route = route_result['route']
        
        report = f"""
        REPORTE DE RUTA DE REABASTECIMIENTO
        ===================================
        
        Resumen:
        - Distancia total: {route_result['total_distance_km']} km
        - Tiempo estimado: {route_result['estimated_time_hours']} horas
        - Zonas de alta demanda: {route_result['total_zones']}
        
        Secuencia de visitas:
        """
        
        for zone in route:
            if zone.get('is_depot', False):
                report += f"\n  {zone['order']}. DEPOT (Inicio/Fin)"
            else:
                report += f"\n  {zone['order']}. Zona {zone['zone_id']} - Ocupación: {zone['predicted_occupation']:.1f}%"
        
        return report