import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = None

    def prepare_data(self, df, target_column='occupation'):
        """Preparar datos para entrenamiento"""
        df_processed = df.copy()

        # Crear características temporales si existe columna 'hour'
        if 'hour' in df_processed.columns:
            df_processed['hour_sin'] = np.sin(
                2 * np.pi * df_processed['hour'] / 24)
            df_processed['hour_cos'] = np.cos(
                2 * np.pi * df_processed['hour'] / 24)

        # Codificar variables categóricas
        categorical_cols = df_processed.select_dtypes(
            include=['object']).columns
        for col in categorical_cols:
            if col != target_column:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(
                    df_processed[col].astype(str))
                self.label_encoders[col] = le

        # Separar features y target
        feature_cols = [
            col for col in df_processed.columns if col != target_column]
        X = df_processed[feature_cols]
        y = df_processed[target_column]

        # Manejar valores faltantes
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        self.feature_names = feature_cols
        self.target_name = target_column

        return X, y

    def predict_all_zones_by_hour(self, hour, weather_data=None):
        """Predecir ocupación para todas las zonas en una hora específica"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecuta train_with_cv() primero.")
        
        # Obtener todas las zonas únicas del entrenamiento
        if not hasattr(self, 'unique_zones'):
            raise ValueError("No se tienen datos de zonas. Entrena el modelo primero.")
        
        predictions = []
        
        for zone_data in self.unique_zones:
            # Crear input para cada zona
            zone_input = zone_data.copy()
            zone_input['hour'] = hour
            
            # Agregar datos meteorológicos si se proporcionan
            if weather_data:
                zone_input.update(weather_data)
            
            # Hacer predicción
            pred = self.predict_custom([zone_input])
            
            predictions.append({
                'zone_id': zone_data['zone_id'],
                'lat': zone_data['lat'],
                'lon': zone_data['lon'],
                'ciudad': zone_data['ciudad'],
                'predicted_occupation': pred[0],
                'hour': hour
            })
        
        return sorted(predictions, key=lambda x: x['predicted_occupation'], reverse=True)
    
    def train_with_cv(self, df, target_column='occupation', n_splits=5, random_state=42):
        """Entrenar modelo con cross validation - MODIFICADO"""
        X, y = self.prepare_data(df, target_column)
        
        # Guardar información de zonas para predicciones futuras - MEJORADO
        print("Guardando información de zonas...")
        
        # Verificar qué ciudades tenemos
        ciudades_unicas = df['ciudad'].unique() if 'ciudad' in df.columns else ['CDMX']
        print(f"Ciudades encontradas: {ciudades_unicas}")
        
        # Agrupar zonas con información promedio
        zone_columns = ['zone_id', 'lat', 'lon']
        if 'ciudad' in df.columns:
            zone_columns.append('ciudad')
            
        agg_dict = {}
        for col in df.columns:
            if col not in zone_columns + [target_column, 'hour']:
                if col in ['tmin', 'tmax', 'prcp', 'wspd']:
                    agg_dict[col] = 'mean'
                elif col in ['weekday', 'holiday']:
                    agg_dict[col] = 'first'
        
        zone_info = df.groupby('zone_id').agg({
            'lat': 'first',
            'lon': 'first', 
            'ciudad': 'first' if 'ciudad' in df.columns else lambda x: 'CDMX',
            **agg_dict
        }).reset_index()
        
        self.unique_zones = zone_info.to_dict('records')
        print(f"Zonas guardadas: {len(self.unique_zones)}")
        print(f"Ejemplo de zona: {self.unique_zones[0] if self.unique_zones else 'Ninguna'}")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Configurar modelo XGBoost
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Cross validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=n_splits, scoring='r2'
        )
        
        # Entrenar modelo final
        self.model.fit(X_train_scaled, y_train)
        
        # Predicciones en test
        y_pred = self.model.predict(X_test_scaled)
        
        # Métricas
        results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_r2': r2_score(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"Cross Validation R² Score: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
        print(f"Test R² Score: {results['test_r2']:.4f}")
        print(f"Test RMSE: {results['test_rmse']:.4f}")
        print(f"Test MAE: {results['test_mae']:.4f}")
        
        return results

    def predict_custom(self, input_data):
        """Hacer predicciones personalizadas"""
        if self.model is None:
            raise ValueError(
                "Modelo no entrenado. Ejecuta train_with_cv() primero.")

        # Convertir a DataFrame si es necesario
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_data = pd.DataFrame(input_data)

        # Preparar datos
        df_processed = input_data.copy()

        # Aplicar mismas transformaciones que en entrenamiento
        if 'hour' in df_processed.columns:
            df_processed['hour_sin'] = np.sin(
                2 * np.pi * df_processed['hour'] / 24)
            df_processed['hour_cos'] = np.cos(
                2 * np.pi * df_processed['hour'] / 24)

        # Codificar variables categóricas
        for col, encoder in self.label_encoders.items():
            if col in df_processed.columns:
                df_processed[col] = encoder.transform(
                    df_processed[col].astype(str))

        # Seleccionar solo las características usadas en entrenamiento
        X_pred = df_processed[self.feature_names].fillna(0)

        # Escalar
        X_pred_scaled = self.scaler.transform(X_pred)

        # Predecir
        predictions = self.model.predict(X_pred_scaled)

        return predictions

    def plot_results(self, results):
        """Visualizar resultados"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # CV scores
        axes[0, 0].bar(range(len(results['cv_scores'])), results['cv_scores'])
        axes[0, 0].set_title('Cross Validation R² Scores')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('R² Score')

        # Predicciones vs reales
        axes[0, 1].scatter(results['y_test'], results['y_pred'], alpha=0.6)
        axes[0, 1].plot([results['y_test'].min(), results['y_test'].max()],
                        [results['y_test'].min(), results['y_test'].max()], 'r--')
        axes[0, 1].set_xlabel('Valores Reales')
        axes[0, 1].set_ylabel('Predicciones')
        axes[0, 1].set_title('Predicciones vs Valores Reales')

        # Residuos
        residuos = results['y_test'] - results['y_pred']
        axes[1, 0].scatter(results['y_pred'], residuos, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicciones')
        axes[1, 0].set_ylabel('Residuos')
        axes[1, 0].set_title('Gráfico de Residuos')

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.feature_names
            indices = np.argsort(importance)[::-1][:10]  # Top 10 features

            axes[1, 1].bar(range(len(indices)), importance[indices])
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Features')
            axes[1, 1].set_ylabel('Importance')
            axes[1, 1].set_xticks(range(len(indices)))
            axes[1, 1].set_xticklabels([feature_names[i]
                                       for i in indices], rotation=45)

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath='xgboost_model.pkl'):
        """Guardar modelo - MODIFICADO"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'unique_zones': getattr(self, 'unique_zones', [])
        }
        joblib.dump(model_data, filepath)
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath='xgboost_model.pkl'):
        """Cargar modelo - MODIFICADO"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.unique_zones = model_data.get('unique_zones', [])
        print(f"Modelo cargado desde: {filepath}")
