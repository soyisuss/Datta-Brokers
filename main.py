import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar muestras para evitar exceso de memoria
cdmx = pd.read_csv("./data/cdmx_data_series.csv", nrows=10000)
lyon = pd.read_csv("./data/lyon_data_series.csv", nrows=10000)

# Convertir a formato largo (occupation_0 a occupation_23)
occupation_cols = [f'occupation_{i}' for i in range(24)]
cdmx_long = cdmx.melt(
    id_vars=['zone_id', 'lat', 'lon', 'tmin', 'tmax',
             'prcp', 'wspd', 'weekday', 'holiday'],
    value_vars=occupation_cols,
    var_name='hour',
    value_name='occupation'
)
cdmx_long['hour'] = cdmx_long['hour'].str.extract('(\d+)').astype(int)

lyon_long = lyon.melt(
    id_vars=['zone_id', 'lat', 'lon', 'tmin', 'tmax',
             'prcp', 'wspd', 'weekday', 'holiday'],
    value_vars=occupation_cols,
    var_name='hour',
    value_name='occupation'
)
lyon_long['hour'] = lyon_long['hour'].str.extract('(\d+)').astype(int)

# Ocupación promedio por hora
cdmx_hour_avg = cdmx_long.groupby('hour')['occupation'].mean()
lyon_hour_avg = lyon_long.groupby('hour')['occupation'].mean()

# Correlaciones clima-ocupación
cdmx_corr = cdmx_long[['tmin', 'tmax', 'prcp', 'wspd', 'occupation']].corr()[
    'occupation'].drop('occupation')
lyon_corr = lyon_long[['tmin', 'tmax', 'prcp', 'wspd', 'occupation']].corr()[
    'occupation'].drop('occupation')

# Imprimir correlaciones
print("Correlación CDMX:\n", cdmx_corr)
print("\nCorrelación Lyon:\n", lyon_corr)

# Gráfica de ocupación promedio por hora
plt.figure(figsize=(12, 5))
sns.lineplot(data=cdmx_hour_avg, label="CDMX")
sns.lineplot(data=lyon_hour_avg, label="Lyon")
plt.title("Ocupación Promedio por Hora")
plt.xlabel("Hora del día")
plt.ylabel("Ocupación Promedio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
