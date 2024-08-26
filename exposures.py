import numpy as np
from pathlib import Path

#import CLIMADA modules:
from climada.hazard import Centroids, tc_tracks
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
buffer = 1.0

def generate_exposure(country, tracks_dic, load_exp=False):

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

    print("----------------------Filter TC Tracks----------------------")
    storm_basin_sub = {}

    for basin in applicable_basins:
        sub = tracks_dic[basin].tracks_in_exp(exp, buffer)
        storm_basin_sub[basin] = sub

        print(f"Number of tracks in {basin} basin:",storm_basin_sub[basin].size)   

    #plot exposure, centroids, and state STORM basin(s)
    exp.plot_raster()
    exp.plot_scatter()
    centrs.plot()
    print('STORM basin of country: ', applicable_basins)

    
    if applicable_basins:
        return exp, centrs, applicable_basins, storm_basin_sub
    else:
        return exp, centrs, "Country code not found in basin", storm_basin_sub