import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.io import MemoryFile
from rasterio.features import shapes, rasterize
from rasterio.transform import from_origin, from_bounds
from shapely.geometry import box, shape


# Define raster properties
pixel_size = 0.008333  # Size of each pixel in degrees (adjust this value as needed)
buffer_size = 0.5  # Buffer size in degrees to expand the raster bounds (adjust this value as needed)
grid_size = 0.3 # Size of each grid cell in degrees (adjust this value as needed)


def init_grid(exp, plot_rst=True):
    minx, miny, maxx, maxy = exp.gdf.total_bounds  # Get bounding box of the GeoDataFrame

    # Expand the bounds by the buffer size
    minx -= buffer_size
    miny -= buffer_size
    maxx += buffer_size
    maxy += buffer_size

    # Calculate the number of rows and columns for the raster
    nrows = int((maxy - miny) / pixel_size) + 1
    ncols = int((maxx - minx) / pixel_size) + 1

    # Define the transformation matrix
    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    # Initialize the raster grid with NaNs 
    raster = np.full((nrows, ncols), np.nan)

    # Loop through each point in the GeoDataFrame to assign values to the raster
    for _, row in exp.gdf.iterrows():
        # Extract x (longitude) and y (latitude) from the geometry
        x, y = row.geometry.x, row.geometry.y

        # Calculate the column and row index for each point
        col = int((x - minx) / pixel_size)
        row_idx = int((maxy - y) / pixel_size)

        # Assign the value to the corresponding cell in the raster
        raster[row_idx, col] = row['value']

    # Write the raster to a GeoTIFF file using rasterio
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=nrows,
            width=ncols,
            count=1,  # Number of bands
            dtype=rasterio.float32,  # Data type for the raster values
            crs='EPSG:4326',  # Coordinate reference system
            transform=transform,
        ) as dataset:
            dataset.write(raster, 1)  # Write raster data to the first band
            
            # Read back the raster data from the in-memory file
            raster_data = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
    
    # Generate grid cells over the bounding box
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)

    grid_cells = [box(x, y, x + grid_size, y + grid_size) for x in x_coords for y in y_coords]
    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
    
    # Extract the shapes of the non-zero areas (islands)
    mask = raster_data > 0  # Create a mask for non-zero values
    island_shapes = list(shapes(raster_data, mask=mask, transform=transform))

    # Create a GeoDataFrame of island polygons
    island_polygons = [shape(geom) for geom, value in island_shapes if value > 0]
    islands_gdf = gpd.GeoDataFrame({'geometry': island_polygons}, crs=crs)

    # Select grid cells that intersect with the islands
    intersecting_cells = gpd.sjoin(grid_gdf, islands_gdf, how='inner', predicate='intersects')
    intersecting_cells = intersecting_cells.drop_duplicates(subset='geometry') #remove duplicates
    intersecting_cells['grid_letter'] = [chr(65 + i) for i in range(len(intersecting_cells))]
    intersecting_cells = intersecting_cells.drop(columns=['index_right'])

    if plot_rst:
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot the original raster
        islands_gdf.plot(ax=ax, color='blue', legend=True, label='Islands')
        # Plot the intersecting grid cells
        intersecting_cells.plot(ax=ax, edgecolor='red', facecolor = 'none', label='Grid cells')
        plt.title("Grid Cells Over Islands")
        plt.legend()
        plt.show()

    return intersecting_cells








import numpy as np
from pathlib import Path

#import CLIMADA modules:
from climada.hazard import Centroids, TropCyclone
from climada.hazard.tc_tracks import TCTracks
from climada.entity import LitPop

#define directories
EXPOSURE_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/exposure")
HAZARD_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard")
TC_TRACKS_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks/tracks_basins_climada")
STORM_dir = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks")

#define countries per tropical cyclone basin according to STORM dataset
NA = [28,44,52,84,132,192,212,214,308,624,328,332,388,659,662,670,740,780]
SI = [174,480,690,626]
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
fin = 'gdp' #fin mode
year = 2020 #reference year
res = 30 #resolution in arcsec
buffer = 100000 #buffer in meter

#define variables for TC class
r = 10000 #number of simulated years in tc dataset
freq_corr_STORM = 1 / r





def init_TC_exp(country, load_fls=False, plot_exp=True):

    """Define STORM Basin"""
    for basin, countries in basins_countries.items():
        if country in countries:
            applicable_basin = basin
            print('STORM basin of country: ', applicable_basin)
    if 'applicable_basin' not in locals():
        print('Error: Applicable basin not found - Do not proceed.')
        return 0, 0, 0, 0
    else:
        pass
        

    exp_str = f"Exp_{country}_{fin}_{year}_{res}.hdf5"
    if load_fls and Path.is_file(EXPOSURE_DIR.joinpath(exp_str)):
        """Loading Exposure"""
        print("----------------------Loading Exposure----------------------")
        exp = LitPop.from_hdf5(EXPOSURE_DIR.joinpath(exp_str))
    else:
        """Initiating Exposure"""
        print("----------------------Initiating Exposure----------------------")
        exp = LitPop.from_countries(country, fin_mode=fin, reference_year=year, res_arcsec=res)
        exp.write_hdf5(EXPOSURE_DIR.joinpath(exp_str))
    
    if plot_exp:
        exp.plot_raster()

    """initiate TC hazard from tracks and exposure"""
    # initiate new instance of TropCyclone(Hazard) class:
    haz_str = f"TC_sub_{applicable_basin}_{country}_{res}_STORM.hdf5"
    track_str = f"Track_sub_{applicable_basin}_{country}_{res}_STORM.hdf5"
    if load_fls and Path.is_file(HAZARD_DIR.joinpath(haz_str)):
        print("----------------------Loading Hazard----------------------")
        tc_storms = TropCyclone.from_hdf5(HAZARD_DIR.joinpath(haz_str))
        storm_basin_sub = TCTracks.from_hdf5(HAZARD_DIR.joinpath(track_str))

    else:
        print("----------------------Generate Hazard----------------------")
        """Generating Centroids"""
        lat = exp.gdf['latitude'].values
        lon = exp.gdf['longitude'].values
        centrs = Centroids.from_lat_lon(lat, lon)
        if plot_exp:
            centrs.plot()

        track_dic = init_STORM_tracks(applicable_basin)

        """Filter TC Tracks"""
        #storm_basin_sub = {}
        #storm_basin_sub = track_dic[applicable_basin].tracks_in_exp(exp, buffer)
        utm_crs = "EPSG:3857" 
        exp_crs = exp.gdf.to_crs(utm_crs) 
        exp_buffer = exp_crs.buffer(distance=buffer, resolution=1)
        exp_buffer = exp_buffer.unary_union
        tc_tracks_lines_crs = track_dic[applicable_basin].to_geodataframe().to_crs(utm_crs)
        print('Next Buffer')
        tc_tracks_lines = tc_tracks_lines_crs.buffer(distance=buffer)
        print('Buffer finsihed')
        select_tracks = tc_tracks_lines.intersects(exp_buffer)
        tracks_in_exp = [track for j, track in enumerate(track_dic[applicable_basin].data) if select_tracks[j]]
        storm_basin_sub = TCTracks(tracks_in_exp)
        print('Filter finished')
        storm_basin_sub.write_hdf5(HAZARD_DIR.joinpath(track_str)) 

        print(f"Number of tracks in {applicable_basin} basin:",storm_basin_sub.size) 

        #generate TropCyclone class from previously loaded TC tracks for one storm data set
        tc_storms = TropCyclone.from_tracks(storm_basin_sub, centroids=centrs)
        tc_storms.frequency = np.ones(tc_storms.event_id.size) * freq_corr_STORM
        tc_storms.check()
        tc_storms.write_hdf5(HAZARD_DIR.joinpath(haz_str))    
    
    return exp, applicable_basin, storm_basin_sub, tc_storms


#Load all STORM tracks for the basin of interest.
def init_STORM_tracks(basin, load_fls=False):
    """Import TC Tracks"""
    all_tracks = []
    storms_basin = {}
    print("----------------------Initiating TC Tracks----------------------")
    fname = lambda i: f"STORM_DATA_IBTRACS_{basin}_1000_YEARS_{i}.txt"
    for i in range(10):
        tracks_STORM = TCTracks.from_simulations_storm(STORM_dir.joinpath(fname(i)))
        all_tracks.extend(tracks_STORM.data)
    tracks_STORM.data = all_tracks
            
    storms_basin[basin] = tracks_STORM

    return storms_basin













import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#import climada stuff
from climada.entity.impact_funcs import trop_cyclone
from climada.engine import ImpactCalc


def init_imp(exp, haz, grid, plot_frequ=True):
    #import regional calibrated impact function for TC
    # prepare impact calcuation - after Samuel Eberenz
    # The iso3n codes need to be consistent with the column “region_id” in the 
    # 1. Init impact functions:
    impact_func_set = trop_cyclone.ImpfSetTropCyclone()
    impf_set = impact_func_set.from_calibrated_regional_ImpfSet()
    impf_set.check()

    # get mapping: country ISO3n per region:
    iso3n_per_region = impf_id_per_region = impact_func_set.get_countries_per_region()[2]
    
    code_regions = {'NA1': 1, 'NA2': 2, 'NI': 3, 'OC': 4, 'SI': 5, 'WP1': 6, \
                    'WP2': 7, 'WP3': 8, 'WP4': 9, 'ROW': 10}

    # match exposure with correspoding impact function
    for calibration_region in impf_id_per_region:
        for country_iso3n in iso3n_per_region[calibration_region]:
            exp.gdf.loc[exp.gdf.region_id== country_iso3n, 'impf_TC'] = code_regions[calibration_region]

    #perform impact calcualtion
    imp = ImpactCalc(exp, impf_set, haz).impact(save_mat=True)

    #compute exceedance frequency curve
    frequ_curve = imp.calc_freq_curve()
    if plot_frequ:
        mask = frequ_curve.return_per < 500
        return_period_flt = frequ_curve.return_per[mask]
        impact_flt = frequ_curve.impact[mask]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(23, 6), gridspec_kw={'width_ratios': [10, 5, 5]})

        ax1.plot(return_period_flt, impact_flt, label='Filtered Data')

        # Add labels and title
        ax1.set_title("Impact Frequency Curve")
        ax1.set_xlabel("Return Period [Years]")
        ax1.set_ylabel("Impact [USD]")

        # Create an inset plot (overview of total data)
        inset_ax1 = inset_axes(ax1, width="30%", height="30%", loc='upper left', borderpad=3.0)  # adjust size and position
        inset_ax1.plot(frequ_curve.return_per, frequ_curve.impact, label='Overview Data')
        inset_ax1.set_xlabel("Return Period [Years]", fontsize=8)
        inset_ax1.set_ylabel("Impact [USD]", fontsize=8)

        ax2.plot(frequ_curve.return_per, frequ_curve.impact)
        ax2.set_xscale('log')
        ax2.set_xlabel('Return Period [Years]')
        ax2.set_ylabel('Impact [USD]')
        ax2.set_title('Impact Frequency Curve - Log')

        ax3.plot(frequ_curve.return_per, frequ_curve.impact)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Return Period [Years]')
        ax3.set_ylabel('Impact [USD]')
        ax3.set_title('Impact Frequency Curve - LogLog')

        # Show both plots
        plt.tight_layout()
        plt.show()

    #save impact per exposure point
    imp_per_exp = imp.imp_mat

    #Perform a spatial join to associate each exposure point with calculated impact with a grid cell
    exp_to_grid = exp.gdf.sjoin(grid, how='left', predicate="within")

    #group each exposure point according to grid cell letter
    agg_exp = exp_to_grid.groupby('grid_letter').apply(lambda x: x.index.tolist())

    #Dictionary to store the impacts for each grid cell
    imp_grid_csr = {}

    #Loop through each grid cell and its corresponding line numbers
    for letter, line_numbers in agg_exp.items():
        selected_values = imp_per_exp[:, line_numbers] #Select all impact values per grid cell
        imp_grid_csr[letter] = selected_values #Store them in dictionary per grid cell

    imp_grid_evt = {} #total damage for each event per grid cell

    #sum all impacts per grid cell
    for i in imp_grid_csr:
        imp_grid_evt[i] = imp_grid_csr[i].sum(axis=1) #calculate sum of impacts per grid cell
        imp_grid_evt[i] = [matrix.item() for matrix in imp_grid_evt[i]] #single values per event are stored in 1:1 matrix -> only save value

    #transform matrix to data frame
    imp_grid_evt = pd.DataFrame.from_dict(imp_grid_evt)

    return imp, imp_per_exp, agg_exp, imp_grid_evt




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re

from scipy.optimize import minimize

minimum_payout = 0.1

#Define bounds for minimum and maximum wind speeds
initial_guess = [20, 61]  # Example initial guess
bounds = [(20, 40), (51, 250)]  # Bounds for thresholds

def init_alt_payout(min_trig, max_trig, haz_int, nominal):
    payouts = []
    for i in range(len(haz_int)):
        payout = np.clip((haz_int[i] - min_trig) / (max_trig - min_trig), 0, 1) * nominal
        if payout < (minimum_payout * nominal):
            payout = 0
        payouts.append(payout)
    return payouts


def init_alt_objective_function(params, haz_int, damages, nominal):
    min_trig, max_trig = params
    payouts = init_alt_payout(min_trig, max_trig, haz_int, nominal)
    tot_payout = np.sum(payouts)
    tot_damages = np.sum(damages)
    basis_risk = ((tot_damages - tot_payout)**2)**0.5
    return basis_risk

def init_alt_optimization(haz_int, damages, nominal, print_params=True):
    # Define bounds and initial guesses for each grid cell
    grid_cells = range(len(haz_int.columns)-1)  # Assuming 10 grid cells
    grid_specific_results = {}

    for cell in grid_cells:

        # Perform optimization for each grid cell
        result = minimize(init_alt_objective_function, 
                          initial_guess, 
                          args=(haz_int.iloc[:,cell], damages.iloc[:,cell], nominal), 
                          bounds=bounds, 
                          method='L-BFGS-B')

        optimal_min_speed, optimal_max_speed = result.x
        grid_specific_results[cell] = (optimal_min_speed, optimal_max_speed)

    if print_params:
        print(grid_specific_results)

    #Reshape parameters into a more interpretable form if needed
    optimized_xs = np.array([values[0] for values in grid_specific_results.values()])  #minimum threshold of windspeeds
    optimized_ys = np.array([values[1] for values in grid_specific_results.values()])  #maximum threshold of windspeeds

    return result, optimized_xs, optimized_ys

def alt_pay_vs_damage(damages, optimized_xs, optimized_ys, haz_int, nominal, include_plot=False):
    b = len(damages)
    payout_evt_grd = pd.DataFrame({letter: [None] * b for letter in haz_int.columns[:-1]})
    pay_dam_df = pd.DataFrame({'pay': [0.0] * b, 'damage': [0.0] * b, 'year': [0] * b})

    for i in range(len(damages)):
        tot_dam = np.sum(damages.iloc[i, :])
        pay_dam_df.loc[i,"damage"] = tot_dam
        pay_dam_df.loc[i,"year"] = int(haz_int['year'][i])
        for j in range(len(haz_int.columns)-1):
            grid_hazint = haz_int.iloc[:,j] 
            payouts = init_alt_payout(optimized_xs[j], optimized_ys[j], grid_hazint, nominal)
            payout_evt_grd.iloc[:,j] = payouts
        tot_pay = np.sum(payout_evt_grd.iloc[i, :])
        if tot_pay > nominal:
                tot_pay = nominal
        else: 
            pass
        pay_dam_df.loc[i,"pay"] = tot_pay

    if include_plot:

        mask = pay_dam_df['damage'] <= nominal
        damage_flt = pay_dam_df['damage'][mask]
        payout_flt = pay_dam_df['pay'][mask]

        #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(23, 6), gridspec_kw={'width_ratios': [10, 5, 5]})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        ax1.scatter(damage_flt, payout_flt, marker='o', color='blue', label='Events')
        ax1.plot([payout_flt.min(), payout_flt.max()], [payout_flt.min(), payout_flt.max()], color='red', linestyle='--', label='Trendline')
        ax1.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 

        # Add labels and title
        ax1.set_title("Damage vs. Payout for each Event")
        ax1.set_xlabel("Damage [USD]")
        ax1.set_ylabel("Payout [USD]")
        ax1.legend(loc='upper left', borderpad=2.0)

        # Create an inset plot (overview of total data)
        inset_ax1 = inset_axes(ax1, width="30%", height="30%", loc='lower right', borderpad=3.0)  # adjust size and position
        inset_ax1.scatter(pay_dam_df['damage'], pay_dam_df['pay'], label='Overview Data', marker='o', color='blue')
        inset_ax1.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 
        #inset_ax1.set_xscale('log')
        #inset_ax1.set_yscale('log')
        inset_ax1.set_xlabel("Damage [USD]", fontsize=8)
        inset_ax1.set_ylabel("Payout [USD]", fontsize=8)

        ax2.scatter(damage_flt, payout_flt, marker='o', color='blue', label='Events')
        ax2.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 
        ax2.set_xscale('log')
        # Add labels and title
        ax2.set_title("Damage vs. Payout for each Event - Low Damages")
        ax2.set_xlabel("Damage [USD]")
        ax2.set_ylabel("Payout [USD]")
        ax2.legend()


        # Create an inset plot (overview of total data)
        inset_ax2 = inset_axes(ax2, width="30%", height="30%", loc='upper left', borderpad=4.0)  # adjust size and position
        inset_ax2.scatter(damage_flt, payout_flt, label='Overview Data', marker='o', color='blue')
        inset_ax2.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 
        inset_ax2.set_xlabel("Damage [USD]", fontsize=8)
        inset_ax2.set_ylabel("Payout [USD]", fontsize=8)
        # Show both plots
        plt.tight_layout()
        plt.show()

    else: 
        pass

    return pay_dam_df