import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
import matplotlib.pyplot as plt
import shapely


#import CLIMADA modules:
from climada.hazard import Centroids, TropCyclone
from climada.hazard.tc_tracks import TCTracks
from climada.entity import LitPop
import climada.util.coordinates as u_coord
from climada.util.constants import EARTH_RADIUS_KM, SYSTEM_DIR, DEF_CRS

import grider as grd

#define directories
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")
#define countries per tropical cyclone basin according to STORM dataset
NA = [28,44,52,84,132,192,212,214,308,624,328,332,388,659,662,670,740,780]
SI = [174,480,690,626, 450]
SP = [184,242,296,520,570,598,882,90,798,548]
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
res = 30 #resolution in arcsec for exposure
#define variables for grid and centroids
res_centrs = 150 #resolution in arcsec for centroids
grid_cell_size_km = 30 
min_overlap_percent = 10 
#define variables for TC class
r = 10000 #number of simulated years in tc dataset
freq_corr_STORM = 1 / r


def init_TC_exp(country, buffer_distance_km, grid_size=600, buffer_size=1):

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
        
    """Initiating Exposure"""
    exp = LitPop.from_countries(country, fin_mode=fin, reference_year=year, res_arcsec=res)

    """Divide Exposure set into admin/grid cells"""
    islands_gdf, buffered_islands, grid_gdf = grd.process_islands(exp, buffer_distance_km, grid_cell_size_km, min_overlap_percent, False)
    islands_split_gdf = grd.init_equ_pol(exp, grid_size, buffer_size)
    islands_split_gdf['admin_letter'] = [chr(65 + i) for i in range(len(islands_split_gdf))]

    """initiate TC hazard from tracks and exposure"""
    """Generating Centroids"""
    lat = exp.gdf['latitude'].values
    lon = exp.gdf['longitude'].values
    centrs = Centroids.from_lat_lon(lat, lon)
    """Import TC Tracks"""
    track_dic = init_STORM_tracks(applicable_basin)
    """Filter TC Tracks"""
    tc_tracks_lines = to_geodataframe(track_dic[applicable_basin])
    intersected_tracks = gpd.sjoin(tc_tracks_lines, grid_gdf, how='inner', predicate='intersects')
    select_tracks = tc_tracks_lines.index.isin(intersected_tracks.index)
    tracks_in_exp = [track for j, track in enumerate(track_dic[applicable_basin].data) if select_tracks[j]]
    storm_basin_sub = TCTracks(tracks_in_exp) 
    storm_basin_sub.equal_timestep(time_step_h=1)
    #generate TropCyclone class from previously loaded TC tracks for one storm data set
    tc_storms = TropCyclone.from_tracks(storm_basin_sub, centroids=centrs)
    tc_storms.frequency = np.ones(tc_storms.event_id.size) * freq_corr_STORM
    tc_storms.check()
    print('Job done')

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