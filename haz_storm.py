
from pathlib import Path

#import CLIMADA modules:
from climada.hazard.tc_tracks import TCTracks

#define directories
HAZARD_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard")
TC_TRACKS_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks/tracks_basins_climada")
STORM_dir = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks")


#Load all STORM tracks for the basin of interest.
def init_STORM_tracks(basins, load_fls=False):
    """Import TC Tracks"""
    #create empty dictionary for each basin 
    storms_basin = {}

    #loop through each basin and save tc_tracks
    for basin in basins:
        track_str = f"TC_tracks_{basin}_STORM.hdf5"
        if load_fls and Path.is_file(TC_TRACKS_DIR.joinpath(track_str)):
            print("----------------------Load TC Tracks----------------------")
            storms_basin[basin] = TCTracks.from_hdf5(TC_TRACKS_DIR.joinpath(track_str))
        else: 
            print("----------------------Initiating TC Tracks----------------------")
            storm_str = [f"STORM_DATA_IBTRACS_{basin}_1000_YEARS_{i}.txt" for i in range(10)]
            storm_paths = [STORM_dir.joinpath(storm_file) for storm_file in storm_str]

            storms = [TCTracks.from_simulations_storm(storm_path) for storm_path in storm_paths]
        
            storms_combined = []
    
            for storm in storms:
                storms_combined.append(storm.data)

            #storms_combined.write_hdf5(TC_TRACKS_DIR.joinpath(track_str)) 
                
            storms_basin[basin] = storms_combined
    
    return storms_basin