# Open the cdmx flow file and print the first 10 lines
import sys
import os
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore

cdmx_flow_file = 'data/cdmx_data_flow.csv'
if not os.path.exists(cdmx_flow_file):
    print(f"File {cdmx_flow_file} does not exist.")
    sys.exit(1)

# Read the file and print the first 10 lines
rawDataAll = pd.read_csv(cdmx_flow_file)
print(rawDataAll.columns)
print(f"--- Total number of hourly records: {len(rawDataAll)}")

# Read the cdmx zones geo file
cdmx_zones_file = 'data/cdmx_zones.geojson'
if not os.path.exists(cdmx_zones_file):
    print(f"File {cdmx_zones_file} does not exist.")
    sys.exit(1)

# Read the file and print the first 10 lines
cdmx_zones_geo = gpd.read_file(cdmx_zones_file)
cdmx_zones_geo = cdmx_zones_geo.to_crs(epsg=3857)
cdmx_zones_data = pd.read_csv('data/cdmx_zones.csv')
cdmx_zones_geo['name'] = cdmx_zones_data['name'].astype('category')
print(f"--- Total number of zones: {len(cdmx_zones_geo)}")

# Read the Cdmx stations json
cdmx_stations = pd.read_json('data/cdmx_stations.json')
cdmx_stations = pd.DataFrame.from_records(cdmx_stations['data'].stations)
print(f"--- Total number of stations: {len(cdmx_stations)}")
cdmx_stations = cdmx_stations.drop(columns=['name','rental_methods','short_name','eightd_has_key_dispenser','is_charging','has_kiosk','electric_bike_surcharge_waiver','external_id'])
# Create a DataFrame with a geometry containing the Points
geo_cdmx_stations = gpd.GeoDataFrame(
    cdmx_stations, crs="EPSG:4326", geometry=gpd.points_from_xy(cdmx_stations["lon"], cdmx_stations["lat"])
)
geo_cdmx_stations = geo_cdmx_stations.to_crs(epsg=3857)
# Find which zone a station is in
geo_cdmx_stations['zone_id'] = geo_cdmx_stations.apply(lambda x: cdmx_zones_geo[cdmx_zones_geo.contains(x.geometry)].iloc[0].name if not cdmx_zones_geo[cdmx_zones_geo.contains(x.geometry)].empty else 'Unknown', axis=1)
print(f"--- Total number of stations: {len(geo_cdmx_stations)}")

# Remove records with station id > 676
rawDataAll = rawDataAll[rawDataAll['station_id'] <= 676]
rawDataAll = rawDataAll[rawDataAll['date']>='2022-01-01']
# Deduce the zone for each record in the raw data
rawDataAll['zone_id'] = pd.Series(geo_cdmx_stations.loc[rawDataAll['station_id']].zone_id.values,index=rawDataAll.index)
rawDataAll['capacity']= pd.Series(geo_cdmx_stations.loc[rawDataAll['station_id']].capacity.values,index=rawDataAll.index)
rawDataAll['lon']     = pd.Series(geo_cdmx_stations.loc[rawDataAll['station_id']].lon.values,index=rawDataAll.index).round(3)
rawDataAll['lat']     = pd.Series(geo_cdmx_stations.loc[rawDataAll['station_id']].lat.values,index=rawDataAll.index).round(3)
rawDataAll = rawDataAll.groupby(['station_id', 'date']).agg({'hour':lambda x: list(x),'flow': lambda x: list(x),'lon': lambda x: x.iloc[0],'lat': lambda x: x.iloc[0],'zone_id': lambda x: x.iloc[0],'tmin': lambda x: x.iloc[0],'tmax': lambda x: x.iloc[0],'prcp': lambda x: x.iloc[0],'wspd': lambda x: x.iloc[0],'weekday': lambda x: x.iloc[0],'holiday': lambda x: x.iloc[0],'capacity': lambda x: x.iloc[0]}).reset_index()
# Create a new column with a vector of 24 zeros
rawDataAll['hourly_flow'] = rawDataAll['flow'].apply(lambda x: [0]*24)
# Fill the hourly_flow vector with the flow values for each hour.
for i in range(len(rawDataAll)):
    for j in range(len(rawDataAll['hour'][i])):
        if rawDataAll['hour'][i][j] < 23:
            rawDataAll['hourly_flow'][i][rawDataAll['hour'][i][j]+1] = rawDataAll['flow'][i][j]
    # Compute the cumulative flow for each hour
    for j in range(1,24):
        rawDataAll['hourly_flow'][i][j] = rawDataAll['hourly_flow'][i][j]+rawDataAll['hourly_flow'][i][j-1]
    # Normalize the hourly flow by the capacity of the station
    for j in range(1,24):
        rawDataAll['hourly_flow'][i][j] = round(rawDataAll['hourly_flow'][i][j]/rawDataAll['capacity'][i],3)
rawDataAll.rename(columns={'hourly_flow': 'occupation'}, inplace=True)        
rawDataAll = rawDataAll.drop(columns=['flow','hour','date', 'station_id','capacity'])

# Remove any rows with NaN values
rawDataAll = rawDataAll.dropna()
# Reset the index
rawDataAll.reset_index(drop=True, inplace=True)
# Take 90 percent of the data for training and 10 percent for testing
train_size = int(len(rawDataAll) * 0.9)
rawDataAll_train = rawDataAll[:train_size]
rawDataAll_test = rawDataAll[train_size:]
# Save the training and testing data to CSV files
rawDataAll_train.to_csv('data/cdmx_data_series_train.csv', index=False)
rawDataAll_test.to_csv('data/cdmx_data_series_test.csv', index=False)

# A version of the testing data without the 'occupation' column
rawDataAll_test_no_occupation = rawDataAll_test.drop(columns=['occupation'])
# Save the testing data without the 'occupation' column to a CSV file
rawDataAll_test_no_occupation.to_csv('data/cdmx_data_series_test_no_occupation.csv', index=False)
