from climada.hazard import TCTracks
from climada.hazard import Centroids, TropCyclone
import exposures_alt as exa
from pathlib import Path
import sys

OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")


def process_country(cty):
    """Wrapper function to process a single country."""
    print(f"Processing country: {cty}")
    #exp, applicable_basins, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = exa.init_TC_exp(country, 
    #                                                                                          grid_specs=[2,2], 
    #                                                                                          file_path=OUTPUT_DIR, 
    #                                                                                          storm_path=STORM_DIR, 
    #                                                                                          buffer_distance_km=105, 
    #                                                                                          res_exp=30, 
    #                                                                                          buffer_grid_size=1, 
    #                                                                                          load_fls=True)
    #lat = exp.gdf['latitude'].values
    #lon = exp.gdf['longitude'].values
    #centrs = Centroids.from_lat_lon(lat, lon)
    tracks = TCTracks.from_ibtracs_netcdf(basin='SP', year_range=(1980,2022))
    tracks.equal_timestep()
    tracks.write_hdf5(OUTPUT_DIR.joinpath("TC_tracks_IBTRACS_SP_1980_2022.hdf5"))
    #tracks.calc_perturbed_trajectories()
    #tc = TropCyclone.from_tracks(tracks, centroids=centrs)
    #tc.check()
    #tc.write_hdf5(OUTPUT_DIR.joinpath("IBTRACS_SP_1980_2022.hdf5")) 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python IBTRACS_import.py <country_code>")
        sys.exit(1)
    
    country = int(sys.argv[1])
    process_country(country)
    print(f"Finished processing country: {country}")