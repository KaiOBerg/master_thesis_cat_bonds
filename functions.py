
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

def init_tc_hazard(tracks_dic, basins, frequ_corr):
    """initiate TC hazard from tracks and exposure"""
    tc_storms = {}

    for basin in basins:
        #generate TropCyclone class from previously loaded TC tracks for one storm data set
        tc_storms[basin] = TropCyclone.from_tracks(tracks_dic[basin], centroids=centrs)
        tc_storms[basin].frequency = np.ones(tc_storms[basin].event_id.size) * frequ_corr
        tc_storms[basin].check()

def generate_exposure(country, fin, year, res):

    exp = LitPop.from_countries(country, fin_mode=fin, reference_year=year, res_arcsec=res)
    exp.plot_raster()
    exp.plot_scatter()
    
    return exp

def construct_centroids(exp):
    lat = exp.gdf['latitude'].values
    lon = exp.gdf['longitude'].values
    centrs = Centroids.from_lat_lon(lat, lon)
    return centrs