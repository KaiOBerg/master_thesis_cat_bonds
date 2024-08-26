
import numpy as np
from pathlib import Path

#import CLIMADA modules:
from climada.hazard import Centroids, tc_tracks, TropCyclone

#define directories
HAZARD_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard")
TC_TRACKS_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks")



#Load all STORM tracks for the basin of interest.
def init_STORM_tracks(basins, load_haz=False):

    #create empty dictionary for each basin 
    storms_basin = {}

    #loop through each basin and save tc_tracks
    for basin in basins:
        tc_track_str = f"TC_tracks_{basin}_STORM.hdf5"
        if load_haz and Path.is_file(TC_TRACKS_DIR.joinpath(tc_track_str)):
            print("----------------------Loading Hazard----------------------")
            storms_basin[basin] = TropCyclone.from_hdf5(TC_TRACKS_DIR.joinpath(tc_track_str))
        else:
            print("----------------------Initiating Hazard----------------------")
            
            storms = {
                f"storm_{i:02}": tc_tracks.TCTracks.from_simulations_storm(
                    f"C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/storm_tc_tracks/STORM_DATA_IBTRACS_{basin}_1000_YEARS_{i}.txt"
                )                    
                for i in range(10)
                }
        
            storms_combined = storms[next(iter(storms))]
    
            for key in list(storms.keys())[1:]:
                storms_combined.append(storms[key].data)
                
            storms_combined.write_hdf5(TC_TRACKS_DIR.joinpath(tc_track_str))

            storms_basin[basin] = storms_combined
    
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
