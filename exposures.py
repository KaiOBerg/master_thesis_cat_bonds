import numpy as np
from pathlib import Path

#import CLIMADA modules:
from climada.hazard import Centroids, tc_tracks, TropCyclone
from climada.entity import LitPop

#define directories
EXPOSURE_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/exposure")
HAZARD_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard")

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
buffer = 1

#define variables for TC class
freq_corr_STORM = 1/10000


def init_TC_exp(track_dic, country, load_fls=False):

    exp_str = f"Exp_{country}_{fin}_{year}_{res}.hdf5"
    if load_fls and Path.is_file(EXPOSURE_DIR.joinpath(exp_str)):
        """Loading Exposure"""
        exp = LitPop.from_hdf5(EXPOSURE_DIR.joinpath(exp_str))
    else:
        """Initiating Exposure"""
        exp = LitPop.from_countries(country, fin_mode=fin, reference_year=year, res_arcsec=res)
        exp.write_hdf5(EXPOSURE_DIR.joinpath(exp_str))
        
    exp.plot_raster()

    """Generating Centroids"""
    lat = exp.gdf['latitude'].values
    lon = exp.gdf['longitude'].values
    centrs = Centroids.from_lat_lon(lat, lon)
    centrs.plot()

    """Define STORM Basins"""
    for basin, countries in basins_countries.items():
        if country in countries:
            applicable_basin = basin
    print('STORM basin of country: ', applicable_basin)


    """Filter TC Tracks"""
    storm_basin_sub = {}

    storm_basin_sub = track_dic[applicable_basin].tracks_in_exp(exp, buffer)

    print(f"Number of tracks in {applicable_basin} basin:",storm_basin_sub.size)   

    """initiate TC hazard from tracks and exposure"""
    # initiate new instance of TropCyclone(Hazard) class:
    haz_str = f"TC_sub_{applicable_basin}_{country}_{res}_STORM.hdf5"
    if load_fls and Path.is_file(HAZARD_DIR.joinpath(haz_str)):
        print("----------------------Loading Hazard----------------------")
        tc_storms = TropCyclone.from_hdf5(HAZARD_DIR.joinpath(haz_str))
    else:
        #generate TropCyclone class from previously loaded TC tracks for one storm data set
        tc_storms = TropCyclone.from_tracks(storm_basin_sub, centroids=centrs)
        tc_storms.frequency = np.ones(tc_storms.event_id.size) * freq_corr_STORM
        tc_storms.check()
        tc_storms.write_hdf5(HAZARD_DIR.joinpath(haz_str))    
    
    if applicable_basin:
        return exp, centrs, applicable_basin, storm_basin_sub, tc_storms
    else:
        return exp, centrs, "Country code not found in basin", storm_basin_sub, tc_storms