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
import math



def central_tendency(df,attrs):
    mean = df[attrs].mean()
    median = df[attrs].median()
    mode = df[attrs].mode()
    symetric = False
    if mean == median and median == mode:
        symetric = True
    return mean , median , mode , symetric


def quantiles(df, attr):
    quantiles = np.quantile(df[attr], [0, 0.25, 0.5, 0.75, 1.0])
    lower = quantiles[0]
    upper = quantiles[4]
    IQR = quantiles[3] - quantiles[1]
    lower_bound = quantiles[1] - 1.5 * IQR
    upper_bound = quantiles[3] + 1.5 * IQR

    outliers = df[(df[attr] < lower_bound) | (df[attr] > upper_bound)]

    return quantiles, lower, upper, outliers
0

def missing_unique(df , attr):
    total_missing_values = df[attr].isnull().sum()
    unique = df[attr].unique()
    return total_missing_values , unique



def box_plots(df , Attr):
    plt.boxplot(df[Attr])
    plt.title(f"Boxplot of {Attr}")
    plt.ylabel("Values")
    plt.show()


def histogram(df , attr):
    plt.hist(df[attr], bins=10, color='skyblue', edgecolor='black')
    plt.title("Histogram of "+ attr)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def scatter(df , attr1 , attr2):
    sns.scatterplot(data = df ,x=attr1, y=attr2, color='skyblue', edgecolor='black')
    plt.show()


def correlation(df , cols):
    correlation_matrix = df[cols].corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.title("correlation")
    plt.show()






