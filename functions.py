
import os
import numpy as np
from pathlib import Path

#import CLIMADA modules:
from climada.hazard import Centroids, TCTracks, TropCyclone
from climada.entity import LitPop


STORM_DIR = 'C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/storm_tc_tracks'
HAZARD_DIR = 'C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard'
CENT_STR = 'C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/storm_tc_tracks_centroids_0300as_global.hdf5'


def init_STORM_tracks(basin):
    """ Load all STORM tracks for the basin of interest."""
    all_tracks = []
    for i in range(10):
        # Construct the file path based on the basin and file index
        file_path = os.path.join(STORM_DIR, f"STORM_DATA_IBTRACS_{basin}_1000_YEARS_{i}.txt")
        # Load the storm tracks from the file
        try:
            tracks_STORM = TCTracks.from_simulations_storm(file_path)
            # Append the data from this file to the all_tracks list
            all_tracks.extend(tracks_STORM.data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    # Create a new TCTracks object to hold the combined data
    combined_tracks_STORM = TCTracks()
    combined_tracks_STORM.data = all_tracks
    
    # Ensure the time steps are consistent
    combined_tracks_STORM.equal_timestep(time_step_h=1.)

    return combined_tracks_STORM

def filter_tc_tracks(tracks, exp, buffer):
    tracks_storm_sub = tracks.tracks_in_exp(exp, buffer=buffer)
    return tracks_storm_sub

def generate_exposure(country, fin, year, res):
    exp = LitPop.from_countries(country, fin_mode=fin, reference_year=year, res_arcsec=res)
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