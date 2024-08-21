
import numpy as np
from pathlib import Path

#import CLIMADA modules:
from climada.hazard import Centroids, tc_tracks, TropCyclone
from climada.entity import LitPop


STORM_DIR = 'C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/storm_tc_tracks'
HAZARD_DIR = 'C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard'
CENT_STR = 'C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/storm_tc_tracks_centroids_0300as_global.hdf5'


#Load all STORM tracks for the basin of interest.
def init_STORM_tracks(basins):

    print("Starting function")

    storms_basin = {}

    for basin in basins:
        storms = {
            f"storm_{i:02}": tc_tracks.TCTracks.from_simulations_storm(
                f"C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/storm_tc_tracks/STORM_DATA_IBTRACS_{basin}_1000_YEARS_{i}.txt"
            )
            for i in range(10)
            }
        
        print(f"Loaded storms for basin {basin}")

        storms_combined = storms[next(iter(storms))]
    
        for key in list(storms.keys())[1:]:
            storms_combined.append(storms[key].data)
        
        print(f"Combined storms for basin {basin}")

        # Ensure the time steps are consistent  
        #storms_combined.equal_timestep(time_step_h=1.)
        
        #print('Equal time steps ensured')

        storms_basin[basin] = storms_combined

        print(f"Number of tracks in {basin} basin:",storms_basin[basin].size)
    
    return storms_basin

def filter_tc_tracks(tracks_dic, basins, exp, buffer):

    storm_basin_sub = {}

    for basin in basins:
        sub = tracks_dic[basin].tracks_in_exp(exp, buffer)
        storm_basin_sub[basin] = sub

        print(f"Number of tracks in {basin} basin:",storm_basin_sub[basin].size)    
        
    return storm_basin_sub

def generate_exposure(countries, fin, year, res):

    exp = {}

    for cty in countries:
        exp_var = LitPop.from_countries(cty, fin_mode=fin, reference_year=year, res_arcsec=res)
        exp[cty] = exp_var
    
    return exp

def construct_centroids(exp):
    lat = exp.gdf['latitude'].values
    lon = exp.gdf['longitude'].values
    centrs = Centroids.from_lat_lon(lat, lon)
    return centrs


def init_tc_hazard(tracks, cent, load_haz=False):
    """initiate TC hazard from tracks and exposure"""
     # initiate new instance of TropCyclone(Hazard) class:
    haz_str = f"TC_{reg}_0300as_STORM.hdf5"
    if load_haz and Path.is_file(HAZARD_DIR.joinpath(haz_str)):
        print("----------------------Loading Hazard----------------------")
        tc_hazard = TropCyclone.from_hdf5(HAZARD_DIR.joinpath(haz_str))
    else:
        print("----------------------Initiating Hazard----------------------")
        # hazard is initiated from tracks, windfield computed:
        tc_hazard = TropCyclone.from_tracks(tracks, centroids=cent)
        freq_corr_STORM = 1/10000
        tc_hazard.frequency = np.ones(tc_hazard.event_id.size) * freq_corr_STORM
        tc_hazard.check()
        tc_hazard.write_hdf5(HAZARD_DIR.joinpath(haz_str))
    return tc_hazard