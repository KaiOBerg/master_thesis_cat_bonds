import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


#import CLIMADA modules:
from climada.hazard import Centroids, TropCyclone
from climada.hazard.tc_tracks import TCTracks
from climada.entity import LitPop

import grider as grd

#define directories
EXPOSURE_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/exposure")
HAZARD_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard")
TC_TRACKS_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks/tracks_basins_climada")
STORM_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks")
ADMIN_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/countries_admin")


#define countries per tropical cyclone basin according to STORM dataset
NA = [28,44,52,84,132,192,212,214,308,624,328,332,388,659,662,670,740,780]
SI = [174,480,690,626, 450]
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
fin = 'gdp' #fin mode for exposure
year = 2020 #reference year for exposure
res = 30 #resolution in arcsec for exposure
#define variables for grid and centroids
res_centrs = 150 #resolution in arcsec for centroids
buffer_distance_km = 40 
grid_cell_size_km = 30 
min_overlap_percent = 10 
#define variables for TC class
r = 10000 #number of simulated years in tc dataset
freq_corr_STORM = 1 / r





def init_TC_exp(country, grid_size=600, buffer_size=1, load_fls=False, plot_exp=True, plot_centrs=True, plt_grd=True):

    """Define STORM Basin"""
    for basin, countries in basins_countries.items():
        if country in countries:
            applicable_basin = basin
            #print('STORM basin of country: ', applicable_basin)
    if 'applicable_basin' not in locals():
        print('Error: Applicable basin not found - Do not proceed.')
        return 0, 0, 0, 0
    else:
        pass
        
    """Define Exposure"""
    exp_str = f"Exp_{country}_{fin}_{year}_{res}.hdf5"
    if load_fls and Path.is_file(EXPOSURE_DIR.joinpath(exp_str)):
        """Loading Exposure"""
        exp = LitPop.from_hdf5(EXPOSURE_DIR.joinpath(exp_str))
    else:
        """Initiating Exposure"""
        exp = LitPop.from_countries(country, fin_mode=fin, reference_year=year, res_arcsec=res)
        exp.write_hdf5(EXPOSURE_DIR.joinpath(exp_str))
    
    if plot_exp:
        exp.plot_raster(label= 'Exposure [log(tUSD)]', figsize=(10,5))

    """Divide Exposure set into admin/grid cells"""
    islands_gdf, buffered_islands, grid_gdf = grd.process_islands(exp, buffer_distance_km, grid_cell_size_km, min_overlap_percent, plt_grd)
    islands_split_gdf = grd.init_equ_pol(exp, grid_size, buffer_size)
    islands_split_gdf['admin_letter'] = [chr(65 + i) for i in range(len(islands_split_gdf))]

    if plt_grd:
        outer_boundary_grd = grid_gdf.dissolve()
        fig, ax = plt.subplots(figsize=(10, 5))
        islands_gdf.plot(ax=ax, color="green", label="Islands")
        islands_split_gdf.plot(ax=ax, facecolor="none", edgecolor="red", label="Admin")
        outer_boundary_grd.boundary.plot(ax=ax, facecolor="none", edgecolor="black", label="TC Track Boundary")
        handles = [
            plt.Line2D([0], [0], color="green", lw=4, label="Islands"),           
            plt.Line2D([0], [0], color="red", lw=2, label="Admin"),  
            plt.Line2D([0], [0], color="black", lw=2, label="TC Track Boundary")           
        ]
        ax.legend(handles=handles, loc="upper right")
        plt.show()

    """initiate TC hazard from tracks and exposure"""
    # initiate new instance of TropCyclone(Hazard) class:
    haz_str = f"TC_sub_{applicable_basin}_{country}_{res}_STORM.hdf5"
    track_str = f"Track_sub_{applicable_basin}_{country}_{res}_STORM.hdf5"
    if load_fls and Path.is_file(HAZARD_DIR.joinpath(haz_str)):
        tc_storms = TropCyclone.from_hdf5(HAZARD_DIR.joinpath(haz_str))
        storm_basin_sub = TCTracks.from_hdf5(HAZARD_DIR.joinpath(track_str))

    else:
        """Generating Centroids"""
        lat = exp.gdf['latitude'].values
        lon = exp.gdf['longitude'].values
        centrs = Centroids.from_lat_lon(lat, lon)
        if plot_centrs:
            centrs.plot()

        """Import TC Tracks"""
        track_dic = init_STORM_tracks(applicable_basin)

        """Filter TC Tracks"""
        tc_tracks_lines = track_dic[applicable_basin].to_geodataframe()
        intersected_tracks = gpd.sjoin(tc_tracks_lines, grid_gdf, how='inner', predicate='intersects')

        select_tracks = tc_tracks_lines.index.isin(intersected_tracks.index)
        tracks_in_exp = [track for j, track in enumerate(track_dic[applicable_basin].data) if select_tracks[j]]
        storm_basin_sub = TCTracks(tracks_in_exp) 
        storm_basin_sub.equal_timestep(time_step_h=1)
        storm_basin_sub.write_hdf5(HAZARD_DIR.joinpath(track_str)) 

        #generate TropCyclone class from previously loaded TC tracks for one storm data set
        tc_storms = TropCyclone.from_tracks(storm_basin_sub, centroids=centrs)
        tc_storms.frequency = np.ones(tc_storms.event_id.size) * freq_corr_STORM
        tc_storms.check()
        tc_storms.write_hdf5(HAZARD_DIR.joinpath(haz_str))   

    print(f"Number of tracks in {applicable_basin} basin:",storm_basin_sub.size) 

    return exp, applicable_basin, grid_gdf, islands_split_gdf, storm_basin_sub, tc_storms




#Load all STORM tracks for the basin of interest.
def init_STORM_tracks(basin, load_fls=False):
    """Import TC Tracks"""
    all_tracks = []
    storms_basin = {}
    print("----------------------Initiating TC Tracks----------------------")
    fname = lambda i: f"STORM_DATA_IBTRACS_{basin}_1000_YEARS_{i}.txt"
    for i in range(10):
        tracks_STORM = TCTracks.from_simulations_storm(STORM_DIR.joinpath(fname(i)))
        all_tracks.extend(tracks_STORM.data)
    tracks_STORM.data = all_tracks
            
    storms_basin[basin] = tracks_STORM

    return storms_basin



def init_centrs(grid_gdf, resolution_arcsec):
    points = []
    
    #Convert arcseconds to degrees
    resolution_degrees = resolution_arcsec / 3600.0
    
    for idx, row in grid_gdf.iterrows():

        geometry = row.geometry
        minx, miny, maxx, maxy = geometry.bounds
        
        #Calculate the number of points in x and y directions based on the resolution in degrees
        num_points_x = int((maxx - minx) / resolution_degrees)
        num_points_y = int((maxy - miny) / resolution_degrees)
        
        #Generate points
        for i in range(num_points_x + 1):  #+1 to include maxx
            for j in range(num_points_y + 1):  #+1 to include maxy
                x = minx + i * resolution_degrees
                y = miny + j * resolution_degrees
                points.append(Point(x, y))

    points_gdf = gpd.GeoDataFrame(geometry=points, crs=grid_gdf.crs)

    return points_gdf
