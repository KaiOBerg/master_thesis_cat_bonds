import numpy as np
from pathlib import Path

#import CLIMADA modules:
from climada.hazard import Centroids
from climada.entity import LitPop

#define directories
EXPOSURE_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/EXPOSURE")

#define countries per tropical cyclone basin according to STORM dataset
NA = [28,44,52,84,132,192,212,214,308,624,328,332,388,659,662,670,740,780]
SI = [174,480,690,626]
SP = [184,242,296,520,570,598,882,90,626,798,548]
WP = [296,584,583,520,585]
EP = [296]
NI = [462]

#create dictionaries for countries per STORM basin
basins_countries = {
    'NA': NA,
    'SI': SI,
    'SP': SP,
    'WP': WP,
    'EP': EP,
    'NI': NI
}

#define variables for exposure
fin = 'gdp'
year = 2020
res = 30

def generate_exposure(country, fin, year, res, load_exp=False):

    exp_str = f"Exp_{country}_{fin}_{year}_{res}.hdf5"
    if load_exp and Path.is_file(EXPOSURE_DIR.joinpath(exp_str)):
        print("----------------------Loading Exposure----------------------")
        exp = LitPop.from_hdf5(EXPOSURE_DIR.joinpath(exp_str))
    else:
        print("----------------------Initiating Exposure----------------------")
        exp = LitPop.from_countries(country, fin_mode=fin, reference_year=year, res_arcsec=res)
        exp.write_hdf5(EXPOSURE_DIR.joinpath(exp_str))

    print("----------------------Generating Centroids----------------------")
    lat = exp.gdf['latitude'].values
    lon = exp.gdf['longitude'].values
    centrs = Centroids.from_lat_lon(lat, lon)

    print("----------------------Define STORM Basins----------------------")
    applicable_basins = []
    for basin, countries in basins_countries.items():
        if country in countries:
            applicable_basins.append(basin)

    #plot exposure and centroids
    exp.plot_raster()
    exp.plot_scatter()
    centrs.plot()
    
    if applicable_basins:
        return exp, centrs, applicable_basins
    else:
        return exp, centrs, "Country code not found in basin"
    

def construct_centroids(exp):
    lat = exp.gdf['latitude'].values
    lon = exp.gdf['longitude'].values
    centrs = Centroids.from_lat_lon(lat, lon)
    return centrs

def find_storm_basins(country_code):

    applicable_basins = []

    for basin, countries in basins_countries.items():
        if country_code in countries:
            applicable_basins.append(basin)
    return applicable_basins if applicable_basins else ["Country code not found in any basin"]