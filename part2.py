import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
from pathlib import Path
from shapely import wkt
from shapely.geometry import Polygon
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import math

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt

# Fonction pour ajouter les saisons
def add_seasons(data):
    data['times'] = pd.to_datetime(data['time'])
    
    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    data['season'] = data['times'].apply(get_season)
    data = data.drop(columns=['times'])
    return data

# Fonction pour regrouper par saisons
def aggregate_by_season(data):
    data = add_seasons(data)
    result = data.groupby(['season', 'lon', 'lat'])[['PSurf', 'Qair', 'Rainf', 'Snowf', 'Tair', 'Wind']].mean().reset_index()
    return result

# Fonction pour intégrer des données
def merge_data(climatic_data, soil_data):
    gdf_points = gpd.GeoDataFrame(
        climatic_data,
        geometry=gpd.points_from_xy(climatic_data.lon, climatic_data.lat),
        crs="EPSG:4326"
    )
    soil_data['geometry'] = soil_data['geometry'].apply(wkt.loads)
    gdf_polygons = gpd.GeoDataFrame(soil_data, geometry='geometry', crs="EPSG:4326")
    merged_data = gpd.sjoin(gdf_points, gdf_polygons, how="inner", predicate="within")
    return merged_data





def outlier(df, method, cols=None):
    df_out = df.copy()  # Work on a copy to avoid modifying the original DataFrame
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if method == "zscore":
        for feature in cols:
            # Process only numeric columns
            if np.issubdtype(df_out[feature].dtype, np.number):
                # Calculate Z-score for each feature
                z_scores = (df_out[feature] - df_out[feature].mean()) / df_out[feature].std()
                # Filter to keep only rows within Z-score threshold
                df_out = df_out[(z_scores < 3) & (z_scores > -3)]
        return df_out

    elif method == 'IQR':
        for feature in cols:
            # Process only numeric columns
            if np.issubdtype(df_out[feature].dtype, np.number):
                Q1 = df_out[feature].quantile(0.25)
                Q3 = df_out[feature].quantile(0.75)
                IQR = Q3 - Q1
                # Define lower and upper bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # Filter rows within the IQR range
                df_out = df_out[(df_out[feature] >= lower_bound) & (df_out[feature] <= upper_bound)]
        return df_out

    elif method == "Clipping":
        for feature in cols:
            # Process only numeric columns
            if np.issubdtype(df_out[feature].dtype, np.number):
                # Clip values at the specified quantiles
                df_out[feature] = df_out[feature].clip(lower=df_out[feature].quantile(0.05),
                                                       upper=df_out[feature].quantile(0.95))
        return df_out
    elif method == "log":
        for feature in cols:
            if np.issubdtype(df_out[feature].dtype, np.number):
                df_out[feature] = np.log1p(df_out[feature])
        return df_out


def normalize_data(df, method, cols=None):

    df_out = df.copy()  # Work on a copy to avoid modifying the original DataFrame
    
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if method == 'minmax':
        scaler = MinMaxScaler()  #
        df_out[cols] = scaler.fit_transform(df_out[cols])
        return df_out

    elif method == 'zscore':
        df_out[cols] = (df_out[cols] - df_out[cols].mean()) / df_out[cols].std()  
        return df_out

    else:
        raise ValueError("Method should be 'minmax' or 'zscore'.")


def discretization(df, cols, num_bins, method='equal_frequency', label_by_avg=False):
    df_out = df.copy()

    for col in cols:
        if method == 'equal_frequency':
            if label_by_avg:
                bin_edges = pd.qcut(df_out[col], num_bins, retbins=True)[1]
                bin_averages = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
                labels = bin_averages
            else:
                
                labels = [f'cat {i+1}' for i in range(num_bins)]
            df_out[f'{col}_EFD'] = pd.qcut(df_out[col], num_bins, labels=labels)

        elif method == 'equal_width':
            min_val = df_out[col].min()
            max_val = df_out[col].max()
            bin_edges = np.linspace(min_val, max_val, num_bins+1)  
            
            if label_by_avg:
                
                bin_averages = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
                labels = bin_averages
            else:
                # Default behavior: use labels like 'Bin 1', 'Bin 2', etc.
                labels = [f'Bin {i+1}' for i in range(num_bins)]
            
            df_out[f'{col}_EWD'] = pd.cut(df_out[col], bins=bin_edges, labels=labels, include_lowest=True)

        else:
            raise ValueError("Method must be 'equal_frequency' or 'equal_width'")

    return df_out


def eliminate_redundancies(df, method):
    reduced_df = df.copy()
    
    if method == 'horizontal':  
        reduced_df = reduced_df.drop_duplicates().reset_index(drop=True)
    elif method == 'vertical':  
        reduced_df = reduced_df.loc[:, ~reduced_df.T.duplicated()]
    else:
        raise ValueError("Method must be 'horizontal' or 'vertical'")
    
    return reduced_df
