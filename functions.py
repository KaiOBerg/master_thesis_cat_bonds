
from pathlib import Path

#import CLIMADA modules:
from climada.hazard import tc_tracks

#define directories
HAZARD_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard")
TC_TRACKS_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks")


#Load all STORM tracks for the basin of interest.
def init_STORM_tracks(basins):
    """Generate TC Tracks"""
    #create empty dictionary for each basin 
    storms_basin = {}

    #loop through each basin and save tc_tracks
    for basin in basins:
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
                
        storms_basin[basin] = storms_combined
    
    return storms_basin