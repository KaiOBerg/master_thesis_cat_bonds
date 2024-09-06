import numpy as np
from pathlib import Path

#import CLIMADA modules:
from climada.hazard import Centroids, TropCyclone
from climada.hazard.tc_tracks import TCTracks
from climada.entity import LitPop

#define directories
EXPOSURE_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/exposure")
HAZARD_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard")
TC_TRACKS_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks/tracks_basins_climada")
STORM_dir = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks")

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
fin = 'gdp' #fin mode
year = 2020 #reference year
res = 30 #resolution in arcsec
buffer = 0.1 #buffer in arcsex

#define variables for TC class
r = 10000 #number of simulated years in tc dataset
freq_corr_STORM = 1 / r





def init_TC_exp(basins, country, load_fls=False):

    """Define STORM Basin"""
    for basin, countries in basins_countries.items():
        if country in countries:
            applicable_basin = basin
            print('STORM basin of country: ', applicable_basin)
    if 'applicable_basin' not in locals():
        print('Error: Applicable basin not found - Do not proceed.')
        return 0, 0, 0, 0
    else:
        pass
        

    exp_str = f"Exp_{country}_{fin}_{year}_{res}.hdf5"
    if load_fls and Path.is_file(EXPOSURE_DIR.joinpath(exp_str)):
        """Loading Exposure"""
        print("----------------------Loading Exposure----------------------")
        exp = LitPop.from_hdf5(EXPOSURE_DIR.joinpath(exp_str))
    else:
        """Initiating Exposure"""
        print("----------------------Initiating Exposure----------------------")
        exp = LitPop.from_countries(country, fin_mode=fin, reference_year=year, res_arcsec=res)
        exp.write_hdf5(EXPOSURE_DIR.joinpath(exp_str))
        
    exp.plot_raster()

    """initiate TC hazard from tracks and exposure"""
    # initiate new instance of TropCyclone(Hazard) class:
    haz_str = f"TC_sub_{applicable_basin}_{country}_{res}_STORM.hdf5"
    track_str = f"Track_sub_{applicable_basin}_{country}_{res}_STORM.hdf5"
    if load_fls and Path.is_file(HAZARD_DIR.joinpath(haz_str)):
        print("----------------------Loading Hazard----------------------")
        tc_storms = TropCyclone.from_hdf5(HAZARD_DIR.joinpath(haz_str))
        storm_basin_sub = TCTracks.from_hdf5(HAZARD_DIR.joinpath(track_str))

    else:
        print("----------------------Generate Hazard----------------------")
        """Generating Centroids"""
        lat = exp.gdf['latitude'].values
        lon = exp.gdf['longitude'].values
        centrs = Centroids.from_lat_lon(lat, lon)
        centrs.plot()

        track_dic = init_STORM_tracks(basins, load_fls)

        """Filter TC Tracks"""
        storm_basin_sub = {}

        storm_basin_sub = track_dic[applicable_basin].tracks_in_exp(exp, buffer)
        storm_basin_sub.write_hdf5(HAZARD_DIR.joinpath(track_str)) 

        print(f"Number of tracks in {applicable_basin} basin:",storm_basin_sub.size) 

        #generate TropCyclone class from previously loaded TC tracks for one storm data set
        tc_storms = TropCyclone.from_tracks(storm_basin_sub, centroids=centrs)
        tc_storms.frequency = np.ones(tc_storms.event_id.size) * freq_corr_STORM
        tc_storms.check()
        tc_storms.write_hdf5(HAZARD_DIR.joinpath(haz_str))    
    
    return exp, applicable_basin, storm_basin_sub, tc_storms




#Load all STORM tracks for the basin of interest.
def init_STORM_tracks(basins, load_fls=False):
    """Import TC Tracks"""
    #create empty dictionary for each basin 
    storms_basin = {}

    #loop through each basin and save tc_tracks
    for basin in basins:
        print("----------------------Initiating TC Tracks----------------------")
        storm_str = [f"STORM_DATA_IBTRACS_{basin}_1000_YEARS_{i}.txt" for i in range(10)]
        storm_paths = [STORM_dir.joinpath(storm_file) for storm_file in storm_str]

        storms = [TCTracks.from_simulations_storm(storm_path) for storm_path in storm_paths]
        
        storms_combined = []
    
        for storm in storms:
            storms_combined.append(storm.data)
                
        storms_basin[basin] = storms_combined
    
    return storms_basin