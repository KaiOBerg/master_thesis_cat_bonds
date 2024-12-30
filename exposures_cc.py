import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, box
import matplotlib.pyplot as plt
import shapely
import pandas as pd
from shapely.ops import unary_union


#import CLIMADA modules:
from climada.hazard import Centroids, TropCyclone
from climada.hazard.tc_tracks import TCTracks
from climada.entity import LitPop
import climada.util.coordinates as u_coord
from climada.util.constants import EARTH_RADIUS_KM, SYSTEM_DIR, DEF_CRS

import grider as grd

#define directories
EXPOSURE_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/exposure")
HAZARD_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard")
TC_TRACKS_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks/tracks_basins_climada")
ADMIN_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/countries_admin")


#define countries per tropical cyclone basin according to STORM dataset
NA = [28,44,52,84,132,192,212,214,308,624,328,332,388,659,662,670,740,780]
SI = [174,480,690,626, 450]
SP = [184,242,296,520,570,598,882,90,798,548,776]
WP = [584,583,520,585]
EP = []
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
#define variables for grid and centroids
res_centrs = 150 #resolution in arcsec for centroids
grid_cell_size_km = 30 
min_overlap_percent = 10 
#define variables for TC class
r = 10000 #number of simulated years in tc dataset
freq_corr_STORM = 1 / r





def init_TC_exp(country, cc_model, grid_specs, file_path, storm_path, buffer_grid_size=5, buffer_distance_km=105, min_pol_size=1000, res_exp=150, crs="EPSG:3857", load_fls=False, plot_exp=True, plot_centrs=True, plt_grd=True):

    STORM_DIR = storm_path.joinpath(cc_model)


    """Define STORM Basin"""
    for basin, countries in basins_countries.items():
        if country in countries:
            applicable_basin = basin
            #print('STORM basin of country: ', applicable_basin)
    if 'applicable_basin' not in locals():
        print('Error: Applicable basin not found - Do not proceed.')
        return 0, 0, 0, 0, 0, 0
    else:
        pass
        
    """Define Exposure"""
    exp_str = f"Exp_{country}_{fin}_{year}_{res_exp}.hdf5"
    if load_fls and Path.is_file(file_path.joinpath(exp_str)):
        """Loading Exposure"""
        exp = LitPop.from_hdf5(file_path.joinpath(exp_str))
    else:
        """Initiating Exposure"""
        exp = LitPop.from_countries(country, fin_mode=fin, reference_year=year, res_arcsec=res_exp)
        exp.write_hdf5(file_path.joinpath(exp_str))
    
    if plot_exp:
        exp.plot_raster(label= 'Exposure [log(mUSD)]', figsize=(10,5))

    """Divide Exposure set into admin/grid cells"""
    islands_gdf = grd.create_islands(exp, crs).explode(ignore_index=True, index_parts=True)
    buffered_geometries = islands_gdf.geometry.buffer(buffer_grid_size * 1000)
    islands_gdf = unary_union(buffered_geometries)
    islands_gdf = gpd.GeoDataFrame({'geometry': [islands_gdf]}, crs=crs).explode(index_parts=True)
    grid_gdf = crop_grid_cells_to_polygon(islands_gdf, grid_specs, min_pol_size)
    x, y, tc_bound = grd.process_islands(exp, buffer_distance_km, grid_cell_size_km, min_overlap_percent, crs, plt_grd)
    if crs == "EPSG:3857":
        exposure_crs = exp.crs
        islands_gdf = islands_gdf.to_crs(exposure_crs)
        grid_gdf = grid_gdf.to_crs(exposure_crs)
    grid_gdf['admin_letter'] = [chr(65 + i) for i in range(len(grid_gdf))]

    if plt_grd:
        outer_boundary_grd = grid_gdf.dissolve()
        fig, ax = plt.subplots(figsize=(10, 5))
        islands_gdf.plot(ax=ax, color="green", label="Islands")
        grid_gdf.plot(ax=ax, facecolor="none", edgecolor="red", label="Admin")
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
    haz_str = f"TC_sub_{applicable_basin}_{country}_{res_exp}_STORM_cc_{cc_model}.hdf5"
    track_str = f"Track_sub_{applicable_basin}_{country}_{res_exp}_STORMcc_{cc_model}.hdf5"
    if load_fls and Path.is_file(file_path.joinpath(haz_str)):
        tc_storms = TropCyclone.from_hdf5(file_path.joinpath(haz_str))
        storm_basin_sub = TCTracks.from_hdf5(file_path.joinpath(track_str))

    else:
        """Generating Centroids"""
        lat = exp.gdf['latitude'].values
        lon = exp.gdf['longitude'].values
        centrs = Centroids.from_lat_lon(lat, lon)
        if plot_centrs:
            centrs.plot()

        """Import TC Tracks"""
        track_dic = init_STORM_tracks(applicable_basin, STORM_DIR, cc_model)

        """Filter TC Tracks"""
        tc_tracks_lines = to_geodataframe(track_dic[applicable_basin])
        if crs != "EPSG:3857":
            tc_tracks_lines = tc_tracks_lines.to_crs(crs)
        intersected_tracks = gpd.sjoin(tc_tracks_lines, grid_gdf, how='inner', predicate='intersects')
        select_tracks = tc_tracks_lines.index.isin(intersected_tracks.index)
        tracks_in_exp = [track for j, track in enumerate(track_dic[applicable_basin].data) if select_tracks[j]]
        storm_basin_sub = TCTracks(tracks_in_exp) 
        storm_basin_sub.equal_timestep(time_step_h=1)
        storm_basin_sub.write_hdf5(file_path.joinpath(track_str)) 

        #generate TropCyclone class from previously loaded TC tracks for one storm data set
        tc_storms = TropCyclone.from_tracks(storm_basin_sub, centroids=centrs)
        tc_storms.frequency = np.ones(tc_storms.event_id.size) * freq_corr_STORM
        tc_storms.check()
        tc_storms.write_hdf5(file_path.joinpath(haz_str))   

    print(f"Number of tracks in {applicable_basin} basin:",storm_basin_sub.size) 

    return exp, applicable_basin, tc_bound, grid_gdf, storm_basin_sub, tc_storms




#Load all STORM tracks for the basin of interest.
def init_STORM_tracks(basin, STORM_path, cc_model, load_fls=False):
    """Import TC Tracks"""
    all_tracks = []
    storms_basin = {}
    print("----------------------Initiating TC Tracks----------------------")
    if cc_model == 'CMCC':
        fname = lambda i: f"STORM_DATA_CMCC-CM2-VHR4_{basin}_1000_YEARS_{i}_IBTRACSDELTA.txt"
    elif cc_model == 'CNRM':
        fname = lambda i: f"STORM_DATA_CNRM-CM6-1-HR_{basin}_1000_YEARS_{i}_IBTRACSDELTA.txt"
    elif cc_model == 'ECEARTH':
        fname = lambda i: f"STORM_DATA_EC-Earth3P-HR_{basin}_1000_YEARS_{i}_IBTRACSDELTA.txt"
    elif cc_model == 'HADGEM':
        fname = lambda i: f"STORM_DATA_HadGEM3-GC31-HM_{basin}_1000_YEARS_{i}_IBTRACSDELTA.txt"
    else:
        print('ERROR: No valid climate change model')
    for i in range(10):
        tracks_STORM = TCTracks.from_simulations_storm(STORM_path.joinpath(fname(i)))
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

def to_geodataframe(self):
    gdf = gpd.GeoDataFrame([dict(track.attrs) for track in self.data])

    # Normalize longitudes to the [-180, 180] range
    t_lons = [u_coord.lon_normalize(t.lon.values.copy()) for t in self.data]
    t_lats = [t.lat.values for t in self.data]
    # Create geometries
    gdf.geometry = gpd.GeoSeries([
                    LineString(np.c_[lons, lats]) if lons.size > 1
                    else Point(lons, lats)
                    for lons, lats in zip(t_lons, t_lats)
                ])

    gdf.crs = DEF_CRS

    # for splitting, restrict to tracks that come close to the antimeridian
    t_split_mask = np.asarray([
        (lon > 170).any() and (lon < -170).any() and lon.size > 1
        for lon in t_lons])

    antimeridian = LineString([(180, -90), (180, 90)]).buffer(1e-9)  # Tiny buffer to avoid exact overlap
    gdf.loc[t_split_mask, "geometry"] = gdf.geometry[t_split_mask] \
        .to_crs({"proj": "longlat", "lon_wrap": 180}) \
        .apply(lambda line: MultiLineString([
            LineString([(x - 360, y) for x, y in segment.coords])
            if any(x > 180 for x, y in segment.coords) else segment
            for segment in shapely.ops.split(line, antimeridian).geoms
        ]))
    
    return gdf



def crop_grid_cells_to_polygon(gdf, grid_cells_per_polygon, min_pol_size):
    cropped_cells = []
    
    # Loop through each polygon in the GeoDataFrame
    for idx, polygon in gdf.iterrows():
        polygon_area_km2 = polygon.geometry.area / 1e6 
        if polygon_area_km2 < min_pol_size:
            grid_gdf = gpd.GeoDataFrame({'geometry': [polygon.geometry]}, crs=gdf.crs)
            cropped_cells.append(grid_gdf)
        else:
            # Get the bounding box of the polygon
            minx, miny, maxx, maxy = polygon.geometry.bounds

            # Create a grid of the specified number of grid cells within the bounding box
            #num_cells_x = grid_cells_per_polygon[idx][0]
            #num_cells_y = grid_cells_per_polygon[idx][1]
            num_cells_x = grid_cells_per_polygon[0]
            num_cells_y = grid_cells_per_polygon[1]
            x_coords = np.linspace(minx, maxx, num_cells_x + 1)
            y_coords = np.linspace(miny, maxy, num_cells_y + 1)

            # Create the grid cells (rectangles)
            grid_cells = []
            for i in range(num_cells_x):
                for j in range(num_cells_y):
                    # Define the coordinates for each grid cell
                    grid_cell = box(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1])
                    cell_cropped = grid_cell.intersection(polygon.geometry)
                    grid_cells.append(cell_cropped)

            grid_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=gdf.crs)

            cropped_cells.append(grid_gdf)

    grids = gpd.GeoDataFrame(pd.concat(cropped_cells, ignore_index=True), crs=gdf.crs)
    # Reset index to ensure each geometry is on a new row
    grids.reset_index(drop=True, inplace=True)
    grids_clean = grids[~grids.is_empty]
    grids_clean = grids_clean.reset_index(drop=True)
    
    return grids_clean